from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.networks import MLP


class PointNetEncoder(nn.Module):
    """Plain-PointNet exteroceptive encoder for DreamWaQ++-style perception.

    Consumes a (SE(3)-accumulated) point cloud and produces a permutation-invariant latent. The design
    goals are deliberate:

    - **Rotation-variant**: points are consumed in the sensor/body frame with *no* T-Net / spatial
      transformer, so the latent reacts to the cloud's orientation relative to the sensor.
    - **Sequence-invariant**: a symmetric max-pool over points makes the output independent of point
      ordering, so points may be fed in any order.
    - **Deployment-friendly**: only static ops (Linear / activation / sigmoid gate / masked ``amax``),
      no farthest-point-sampling or ``topk``, so it traces/scripts and exports to ONNX cleanly and runs
      real-time on edge GPUs (e.g. Jetson Orin NX).

    Each point carries ``point_feat_dim`` channels; the first three are always ``xyz`` (used for the
    origin/validity mask), with any extra channels (e.g. a recency ``Δt``) appended.

    A learned **confidence filter** down-weights unreliable/stale points before aggregation (per the
    DreamWaQ++ paper). Invalid points -- those at/near the origin, i.e. missed lidar rays mapped to
    ``(0, 0, 0)`` -- are hard-masked out of the pool.

    The VAE heads return a reparameterised latent ``z_pe`` plus its ``mean``/``logvar`` so the (later)
    reconstruction + KL loss can be wired in.
    """

    def __init__(
        self,
        latent_dim: int,
        point_feat_dim: int = 4,
        shared_mlp_dims: list[int] = [64, 128, 256],
        confidence_hidden_dims: list[int] = [64],
        activation: str = "elu",
        use_confidence_filter: bool = True,
        mask_origin: bool = True,
        origin_eps: float = 1e-3,
        logvar_min: float = -30.0,
        logvar_max: float = 3.22,
        **kwargs,
    ):
        if kwargs:
            print(
                "PointNetEncoder.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        self.point_feat_dim = point_feat_dim
        self.use_confidence_filter = use_confidence_filter
        self.mask_origin = mask_origin
        self.origin_eps = origin_eps
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max

        feat_dim = shared_mlp_dims[-1]

        # shared per-point MLP (applied over the last dim of (B, M, point_feat_dim)); every layer,
        # including the last, is activated so the pooled features are activations rather than raw logits
        self.shared_mlp = MLP(point_feat_dim, feat_dim, shared_mlp_dims[:-1], activation, last_activation=activation)
        # learned confidence gate -> per-point logit in (B, M, 1)
        if use_confidence_filter:
            self.confidence_head = MLP(feat_dim, 1, confidence_hidden_dims, activation)
        # VAE heads on the pooled global feature
        self.encode_mean = nn.Linear(feat_dim, latent_dim)
        self.encode_logvar = nn.Linear(feat_dim, latent_dim)

        self.latent_dim = latent_dim
        self.feat_dim = feat_dim

    def reparameterise(self, mean, logvar):
        std = torch.exp(logvar * 0.5)
        eps = torch.randn_like(std)
        return mean + std * eps

    def forward(self, points, mask=None):
        """Encode a point cloud into an exteroceptive latent.

        Args:
            points: ``(B, M, point_feat_dim)`` or flattened ``(B, M * point_feat_dim)``.
            mask: optional ``(B, M)`` bool validity mask. Combined with the origin mask if enabled.

        Returns:
            ``(z_pe, mean, logvar, global_feat)`` -- the sampled latent, its distribution params, and
            the pooled pre-VAE feature ``(B, feat_dim)``.
        """
        if points.dim() == 2:
            points = points.view(points.shape[0], -1, self.point_feat_dim)

        xyz = points[..., :3]

        # validity mask: invalid points are excluded from the pool
        valid = torch.ones(points.shape[:2], dtype=torch.bool, device=points.device)
        if self.mask_origin:
            valid = valid & (xyz.norm(dim=-1) > self.origin_eps)
        if mask is not None:
            valid = valid & mask.bool()

        feats = self.shared_mlp(points)  # (B, M, feat_dim)

        if self.use_confidence_filter:
            gate = torch.sigmoid(self.confidence_head(feats))  # (B, M, 1)
            feats = feats * gate

        # masked symmetric max-pool over points (permutation invariant)
        neg = torch.finfo(feats.dtype).min
        feats = feats.masked_fill(~valid.unsqueeze(-1), neg)
        global_feat = feats.amax(dim=1)  # (B, feat_dim)
        # rows with no valid point pool to ``neg`` everywhere -> zero them out
        any_valid = valid.any(dim=1, keepdim=True)  # (B, 1)
        global_feat = torch.where(any_valid, global_feat, torch.zeros_like(global_feat))

        mean = self.encode_mean(global_feat)
        logvar = self.encode_logvar(global_feat).clamp(self.logvar_min, self.logvar_max)
        z_pe = self.reparameterise(mean, logvar)
        return z_pe, mean, logvar, global_feat


class HeightmapDecoder(nn.Module):
    """Reconstructs the privileged robot-centric heightmap from the exteroceptive latent ``z_pe``.

    An auxiliary head only: the policy consumes the latent, not this output. Supervising the latent to
    reconstruct the privileged heightmap is what forces it to encode foot-relevant terrain. The loss
    (MSE against the ``height_scanner`` ground truth) is wired into ``PPODreamWAQPerception`` later.
    """

    def __init__(
        self,
        latent_dim: int,
        out_dim: int = 187,
        hidden_dims: list[int] = [128, 128],
        activation: str = "elu",
    ):
        super().__init__()
        self.decoder = MLP(latent_dim, out_dim, hidden_dims, activation)
        self.out_dim = out_dim

    def forward(self, z_pe):
        return self.decoder(z_pe)
