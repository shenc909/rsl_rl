from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.networks import MLP


class _MixerBlock(nn.Module):
    """One MLP-Mixer block: token-mixing then channel-mixing, each pre-normed with a residual."""

    def __init__(self, num_tokens: int, token_dim: int, hidden_dim: int, activation: str):
        super().__init__()
        self.token_norm = nn.LayerNorm(token_dim)
        # token-mixing MLP acts across the token axis (applied to the transposed (B, D, T))
        self.token_mlp = MLP(num_tokens, num_tokens, [hidden_dim, hidden_dim], activation)
        self.channel_norm = nn.LayerNorm(token_dim)
        # channel-mixing MLP acts across the feature axis
        self.channel_mlp = MLP(token_dim, token_dim, [hidden_dim, hidden_dim], activation)

    def forward(self, x):  # x: (B, T, D)
        y = self.token_norm(x).transpose(1, 2)  # (B, D, T)
        x = x + self.token_mlp(y).transpose(1, 2)
        x = x + self.channel_mlp(self.channel_norm(x))
        return x


class MLPMixerFusion(nn.Module):
    """Fuses the proprioceptive (CENet) latent and the exteroceptive (PointNet) latent.

    Following DreamWaQ++: layer-norm each modality latent separately, project both to a common token
    dimension, treat them as a 2-token sequence, and mix with token-mixing + channel-mixing MLP blocks.
    The flattened output is the fused feature consumed by the actor.
    """

    def __init__(
        self,
        proprio_dim: int,
        extero_dim: int,
        token_dim: int = 128,
        hidden_dim: int = 256,
        num_blocks: int = 1,
        activation: str = "gelu",
    ):
        super().__init__()
        self.proprio_norm = nn.LayerNorm(proprio_dim)
        self.extero_norm = nn.LayerNorm(extero_dim)
        self.proprio_proj = nn.Linear(proprio_dim, token_dim)
        self.extero_proj = nn.Linear(extero_dim, token_dim)

        self.blocks = nn.ModuleList(
            [
                _MixerBlock(num_tokens=2, token_dim=token_dim, hidden_dim=hidden_dim, activation=activation)
                for _ in range(num_blocks)
            ]
        )

        self.token_dim = token_dim
        self.out_dim = 2 * token_dim

    def forward(self, proprio_latent, extero_latent):
        # normalize each modality separately, then project to the shared token dim
        p = self.proprio_proj(self.proprio_norm(proprio_latent))  # (B, D)
        e = self.extero_proj(self.extero_norm(extero_latent))  # (B, D)
        x = torch.stack((p, e), dim=1)  # (B, 2, D)
        for block in self.blocks:
            x = block(x)
        return x.flatten(start_dim=1)  # (B, 2 * D)
