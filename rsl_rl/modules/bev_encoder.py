from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.utils import resolve_nn_activation


class BEVGridEncoder(nn.Module):
    """Tiny 2D-CNN exteroceptive encoder over a body-frame BEV heightmap grid.

    A drop-in replacement for :class:`PointNetEncoder` in ``ActorCriticDWAQPerception``: it consumes a
    flattened BEV grid ``(B, C*H*W)``, reshapes to ``(B, C, H, W)``, runs a small conv stack + linear
    head, and exposes the same VAE interface returning ``(z_pe, mean, logvar, global_feat)``.

    Motivation: the BEV grid already aggregates the lidar spatially, so a few cheap convolutions replace
    the PointNet's per-point MLP over hundreds of points (and the env-side FPS), cutting both collection
    and forward cost while keeping the DreamWaQ++ mixer + heightmap-reconstruction structure intact.

    Like the PointNet, the VAE heads return a reparameterised ``z_pe`` plus its ``mean``/``logvar`` for
    the reconstruction + KL loss; the actor consumes the deterministic ``mean`` when
    ``extero_use_mean`` is set on the actor-critic (avoids injecting sampled noise under collapse).
    """

    def __init__(
        self,
        latent_dim: int,
        in_channels: int = 2,
        grid_hw: tuple[int, int] = (17, 11),
        cnn_channels: list[int] = [16, 32],
        head_dim: int = 128,
        activation: str = "elu",
        logvar_min: float = -30.0,
        logvar_max: float = 3.22,
        **kwargs,
    ):
        if kwargs:
            print(
                "BEVGridEncoder.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        self.in_channels = in_channels
        self.grid_h, self.grid_w = grid_hw
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max

        layers: list[nn.Module] = []
        c_prev = in_channels
        for c in cnn_channels:
            layers += [nn.Conv2d(c_prev, c, kernel_size=3, padding=1), resolve_nn_activation(activation)]
            c_prev = c
        self.conv = nn.Sequential(*layers)

        feat_dim = head_dim
        self.head = nn.Sequential(
            nn.Linear(c_prev * self.grid_h * self.grid_w, feat_dim), resolve_nn_activation(activation)
        )
        # VAE heads on the pooled global feature (mirrors PointNetEncoder)
        self.encode_mean = nn.Linear(feat_dim, latent_dim)
        self.encode_logvar = nn.Linear(feat_dim, latent_dim)

        self.latent_dim = latent_dim
        self.feat_dim = feat_dim

    def reparameterise(self, mean, logvar):
        std = torch.exp(logvar * 0.5)
        eps = torch.randn_like(std)
        return mean + std * eps

    def forward(self, grid, mask=None):
        """Encode a BEV grid into an exteroceptive latent.

        Args:
            grid: ``(B, C, H, W)`` or flattened ``(B, C*H*W)`` (channel-major).
            mask: ignored (kept for interface parity with ``PointNetEncoder``).

        Returns:
            ``(z_pe, mean, logvar, global_feat)``.
        """
        b = grid.shape[0]
        if grid.dim() == 2:
            grid = grid.view(b, self.in_channels, self.grid_h, self.grid_w)

        feat = self.conv(grid).reshape(b, -1)
        global_feat = self.head(feat)
        mean = self.encode_mean(global_feat)
        logvar = self.encode_logvar(global_feat).clamp(self.logvar_min, self.logvar_max)
        z_pe = self.reparameterise(mean, logvar)
        return z_pe, mean, logvar, global_feat
