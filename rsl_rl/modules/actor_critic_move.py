"""PS-Net actor-critic for MOVE (arXiv 2412.03353).

Implements the pseudo-siamese network of "MOVE: Multi-skill Omnidirectional Legged Locomotion
with Limited View in 3D Environments":

- standard input encoder: proprio history MLP + front-depth CNN -> self-attention transformer ->
  GRU -> latents (v_hat, z VAE, z^f vision latent, z^c contrastive latent);
- surroundings encoder (training only): privileged proprio query cross-attending into cube-map /
  foot-scan visual tokens -> z^c_s;
- critic encoder (training only): privileged proprio + cube-map/foot-scan tokens -> transformer ->
  value MLP;
- decoders: [v_hat, z] -> next-step proprio, z^f -> front cube-map face; shared SimSiam predictor
  for the contrastive latents.

Observation-group conventions follow ActorCriticDWAQ: the actor's current proprio frame and the
encoder history are sliced out of the flattened, term-major "history" obs group via obs_hist_dict.
Unlike DWAQ, the MOVE encoder consumes ALL history_length frames (including the current one) since
the reconstruction target is the next-step observation, obtained by a time shift in the recurrent
minibatches.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.networks import MLP, EmpiricalNormalization, Memory
from rsl_rl.utils import unpad_trajectories
from rsl_rl.utils.safety_utils import check_safe


class DepthEncoderCNN(nn.Module):
    """Small conv stack turning a depth image (or stacked depth views) into embed-dim tokens."""

    def __init__(self, in_channels: int, image_hw: tuple[int, int], embed_dim: int, channels=(16, 32, 64)):
        super().__init__()
        layers = []
        prev = in_channels
        for ch in channels:
            layers.append(nn.Conv2d(prev, ch, kernel_size=3, stride=2, padding=1))
            layers.append(nn.ELU())
            prev = ch
        self.conv = nn.Sequential(*layers)
        # 2x2 spatial tokens regardless of input resolution
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.proj = nn.Linear(channels[-1], embed_dim)
        self.num_tokens = 4
        self.image_hw = image_hw
        self.in_channels = in_channels

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        # x_flat: [B, C*H*W] -> tokens [B, num_tokens, embed_dim]
        batch = x_flat.shape[0]
        x = x_flat.view(batch, self.in_channels, *self.image_hw)
        feat = self.pool(self.conv(x))  # [B, ch, 2, 2]
        tokens = feat.flatten(2).transpose(1, 2)  # [B, 4, ch]
        return self.proj(tokens)


class CrossAttentionBlock(nn.Module):
    """Cross-attention with the visual embedding in the skip connection (MOVE Fig. 4, right)."""

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(nn.Linear(embed_dim, ff_dim), nn.ELU(), nn.Linear(ff_dim, embed_dim))
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, query: torch.Tensor, visual_tokens: torch.Tensor) -> torch.Tensor:
        # query: [B, 1, D]; visual_tokens: [B, T, D]
        attn_out, _ = self.attn(query, visual_tokens, visual_tokens)
        # skip connection carries only the visual embedding (mean-pooled), per the paper: the
        # surroundings encoder must prioritize its visual input over the proprioceptive query.
        visual_skip = visual_tokens.mean(dim=1, keepdim=True)
        x = self.norm1(attn_out + visual_skip)
        x = self.norm2(x + self.ff(x))
        return x  # [B, 1, D]


class ActorCriticMove(nn.Module):
    is_recurrent = True

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        # observation layout
        obs_hist_dict: dict | None = None,
        history_length: int = 10,
        depth_shape: tuple[int, int] = (48, 64),
        cube_num_views: int = 5,
        cube_shape: tuple[int, int] = (24, 24),
        foot_scan_dim: int = 36,
        privileged_proprio_dim: int = 48,
        # architecture
        embed_dim: int = 128,
        num_attn_heads: int = 4,
        num_transformer_layers: int = 2,
        transformer_ff_dim: int = 256,
        gru_hidden_dim: int = 256,
        gru_num_layers: int = 1,
        vae_latent_dim: int = 16,
        vision_latent_dim: int = 32,
        contrast_latent_dim: int = 16,
        proprio_encoder_hidden_dims=(256,),
        obs_decoder_hidden_dims=(64, 128),
        depth_decoder_hidden_dims=(128, 256),
        predictor_hidden_dims=(64,),
        actor_hidden_dims=(512, 256, 128),
        critic_hidden_dims=(512, 256, 128),
        activation: str = "elu",
        # ppo interface
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        cenet_beta: float = 1.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticMove.__init__ got unexpected arguments, which will be ignored: "
                + str(list(kwargs.keys()))
            )
        super().__init__()

        self.obs_groups = obs_groups
        self.obs_hist_dict = dict(obs_hist_dict) if obs_hist_dict else {}
        self.history_length = history_length
        self.depth_shape = tuple(depth_shape)
        self.cube_num_views = cube_num_views
        self.cube_shape = tuple(cube_shape)
        self.foot_scan_dim = foot_scan_dim
        self.cenet_beta = cenet_beta

        # -- observation dimensions
        self.proprio_dim = sum(self.obs_hist_dict.values())
        num_actor_obs = sum(obs[g].shape[-1] for g in obs_groups["policy"])
        assert num_actor_obs == self.proprio_dim, (
            f"policy group dim ({num_actor_obs}) must equal sum(obs_hist_dict dims) ({self.proprio_dim})"
        )
        self.history_dim = sum(obs[g].shape[-1] for g in obs_groups["history"])
        assert self.history_dim == self.proprio_dim * history_length, (
            f"history group dim ({self.history_dim}) must be proprio_dim * history_length "
            f"({self.proprio_dim}*{history_length})"
        )
        num_critic_obs = sum(obs[g].shape[-1] for g in obs_groups["critic"])
        assert num_critic_obs == privileged_proprio_dim, (
            f"critic group dim ({num_critic_obs}) != privileged_proprio_dim ({privileged_proprio_dim})"
        )
        depth_dim = sum(obs[g].shape[-1] for g in obs_groups["depth"])
        assert depth_dim == depth_shape[0] * depth_shape[1], (
            f"depth group dim ({depth_dim}) != depth_shape product ({depth_shape[0] * depth_shape[1]})"
        )
        self.cube_dim = cube_num_views * cube_shape[0] * cube_shape[1]
        self.cube_face_dim = cube_shape[0] * cube_shape[1]
        surround_dim = sum(obs[g].shape[-1] for g in obs_groups["surroundings"])
        assert surround_dim == self.cube_dim + foot_scan_dim, (
            f"surroundings group dim ({surround_dim}) != cube ({self.cube_dim}) + foot scan ({foot_scan_dim})"
        )

        # -- term-major history slicing (AAABBBCCC layout, oldest first, current frame last)
        self.current_indices = []
        offset = 0
        for _key, dim in self.obs_hist_dict.items():
            self.current_indices += list(range(offset + (history_length - 1) * dim, offset + history_length * dim))
            offset += dim * history_length

        # === standard input encoder ===
        self.proprio_encoder = MLP(self.history_dim, embed_dim, list(proprio_encoder_hidden_dims), activation)
        self.depth_encoder = DepthEncoderCNN(1, self.depth_shape, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_attn_heads,
            dim_feedforward=transformer_ff_dim,
            batch_first=True,
            dropout=0.0,
        )
        self.fusion_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        self.memory = Memory(embed_dim, type="gru", num_layers=gru_num_layers, hidden_size=gru_hidden_dim)

        # latent heads off the GRU output
        self.head_vel = nn.Linear(gru_hidden_dim, 3)
        self.head_z_mean = nn.Linear(gru_hidden_dim, vae_latent_dim)
        self.head_z_logvar = nn.Linear(gru_hidden_dim, vae_latent_dim)
        self.head_zf = nn.Linear(gru_hidden_dim, vision_latent_dim)
        self.head_zc = nn.Linear(gru_hidden_dim, contrast_latent_dim)

        # decoders (MOVE Eq. 6)
        self.obs_decoder = MLP(3 + vae_latent_dim, self.proprio_dim, list(obs_decoder_hidden_dims), activation)
        self.depth_decoder = MLP(vision_latent_dim, self.cube_face_dim, list(depth_decoder_hidden_dims), activation)
        # shared SimSiam predictor for both contrastive branches (MOVE Eq. 7)
        self.contrast_predictor = MLP(
            contrast_latent_dim, contrast_latent_dim, list(predictor_hidden_dims), activation
        )

        # === surroundings encoder (training only) ===
        self.surround_proprio_mlp = MLP(privileged_proprio_dim, embed_dim, [embed_dim], activation)
        self.surround_cube_encoder = DepthEncoderCNN(cube_num_views, self.cube_shape, embed_dim)
        self.surround_foot_mlp = MLP(foot_scan_dim, embed_dim, [embed_dim], activation)
        self.surround_cross_attn = CrossAttentionBlock(embed_dim, num_attn_heads, transformer_ff_dim)
        surround_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_attn_heads,
            dim_feedforward=transformer_ff_dim,
            batch_first=True,
            dropout=0.0,
        )
        self.surround_self_attn = nn.TransformerEncoder(surround_layer, num_layers=1)
        self.surround_head_zc = nn.Linear(embed_dim, contrast_latent_dim)

        # === critic encoder (privileged, no weight sharing with the surroundings encoder) ===
        self.critic_proprio_mlp = MLP(privileged_proprio_dim, embed_dim, [embed_dim], activation)
        self.critic_cube_encoder = DepthEncoderCNN(cube_num_views, self.cube_shape, embed_dim)
        self.critic_foot_mlp = MLP(foot_scan_dim, embed_dim, [embed_dim], activation)
        critic_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_attn_heads,
            dim_feedforward=transformer_ff_dim,
            batch_first=True,
            dropout=0.0,
        )
        self.critic_transformer = nn.TransformerEncoder(critic_layer, num_layers=1)
        self.critic = MLP(embed_dim, 1, list(critic_hidden_dims), activation)

        # === actor ===
        self.latent_dim = 3 + vae_latent_dim + vision_latent_dim + contrast_latent_dim
        self.actor = MLP(self.proprio_dim + self.latent_dim, num_actions, list(actor_hidden_dims), activation)

        # === normalization ===
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(self.proprio_dim)
            self.history_obs_normalizer = EmpiricalNormalization(self.history_dim)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()
            self.history_obs_normalizer = torch.nn.Identity()
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(privileged_proprio_dim)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        # action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = None
        # latents cached by the last batch-mode act() call, consumed by PPOMove for the aux losses
        self.last_encoding: dict[str, torch.Tensor] | None = None
        self.estimated_vel = None
        Normal.set_default_validate_args(False)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

    # ------------------------------------------------------------------
    # observation helpers
    # ------------------------------------------------------------------

    def _cat_groups(self, obs, set_name: str) -> torch.Tensor:
        return torch.cat([obs[g] for g in self.obs_groups[set_name]], dim=-1)

    def get_actor_obs(self, obs, normalize: bool = True) -> torch.Tensor:
        """Current proprio frame o_t, sliced out of the (term-major) history group."""
        current = self._cat_groups(obs, "history")[..., self.current_indices]
        return self.actor_obs_normalizer(current) if normalize else current

    def get_history_obs(self, obs, normalize: bool = True) -> torch.Tensor:
        hist = self._cat_groups(obs, "history")
        return self.history_obs_normalizer(hist) if normalize else hist

    def get_critic_obs(self, obs, normalize: bool = True) -> torch.Tensor:
        critic_obs = self._cat_groups(obs, "critic")
        return self.critic_obs_normalizer(critic_obs) if normalize else critic_obs

    def get_depth_obs(self, obs) -> torch.Tensor:
        return self._cat_groups(obs, "depth")

    def get_surroundings_obs(self, obs) -> tuple[torch.Tensor, torch.Tensor]:
        surround = self._cat_groups(obs, "surroundings")
        return surround[..., : self.cube_dim], surround[..., self.cube_dim :]

    def get_vel_target(self, obs) -> torch.Tensor:
        """Ground-truth base linear velocity (raw m/s) at critic-obs indices 3:6."""
        return self.get_critic_obs(obs, normalize=False)[..., 3:6]

    # ------------------------------------------------------------------
    # encoders
    # ------------------------------------------------------------------

    @staticmethod
    def reparameterise(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mean + std * torch.randn_like(std)

    def _fuse_tokens(self, hist_flat: torch.Tensor, depth_flat: torch.Tensor) -> torch.Tensor:
        """MLP + CNN + self-attention transformer over [flat_batch, ...] inputs -> pooled [flat_batch, D]."""
        proprio_token = self.proprio_encoder(hist_flat).unsqueeze(1)  # [B, 1, D]
        depth_tokens = self.depth_encoder(depth_flat)  # [B, 4, D]
        seq = torch.cat([proprio_token, depth_tokens], dim=1)
        fused = self.fusion_transformer(seq)
        return fused.mean(dim=1)

    def _heads(self, gru_out: torch.Tensor, deterministic: bool = False) -> dict[str, torch.Tensor]:
        z_mean = self.head_z_mean(gru_out)
        # clamp as in the DWAQ CENet so the KL term cannot blow up (HF-VAE convention)
        z_logvar = self.head_z_logvar(gru_out).clamp(-30.0, 3.22)
        # deployment/eval uses the latent mean so the policy is deterministic
        z = z_mean if deterministic else self.reparameterise(z_mean, z_logvar)
        return {
            "vel": self.head_vel(gru_out),
            "z_mean": z_mean,
            "z_logvar": z_logvar,
            "z": z,
            "zf": self.head_zf(gru_out),
            "zc": self.head_zc(gru_out),
        }

    def encode_standard(self, obs, masks=None, hidden_states=None, deterministic: bool = False) -> dict[str, torch.Tensor]:
        """Standard input encoder.

        In rollout/inference mode (masks is None) obs tensors are [N, D]; the internal GRU state is
        advanced. In batch (update) mode obs tensors are padded trajectories [T, n_traj, D]; the
        returned latents are PADDED [T, n_traj, ...] (PPOMove masks them; the actor path unpads).
        """
        hist = self.get_history_obs(obs)
        depth = self.get_depth_obs(obs)
        if not check_safe(hist):
            print("[ActorCriticMove] history obs has nan/inf")
        if not check_safe(depth):
            print("[ActorCriticMove] depth obs has nan/inf")

        batch_mode = masks is not None
        if batch_mode:
            if hidden_states is None:
                raise ValueError("Hidden states must be passed for batch-mode encoding")
            t_dim, n_traj = hist.shape[0], hist.shape[1]
            pooled = self._fuse_tokens(hist.flatten(0, 1), depth.flatten(0, 1))
            pooled = pooled.view(t_dim, n_traj, -1)
            gru_out, _ = self.memory.rnn(pooled, hidden_states)  # padded [T, n_traj, H]
        else:
            pooled = self._fuse_tokens(hist, depth)
            gru_out = self.memory(pooled).squeeze(0)  # [N, H]
        return self._heads(gru_out, deterministic=deterministic)

    def encode_surroundings(self, obs) -> torch.Tensor:
        """Privileged surroundings encoder -> z^c_s. Stateless; accepts [..., D] obs."""
        critic_obs = self.get_critic_obs(obs)
        cube, foot = self.get_surroundings_obs(obs)
        lead_shape = critic_obs.shape[:-1]
        critic_flat = critic_obs.reshape(-1, critic_obs.shape[-1])
        cube_flat = cube.reshape(-1, cube.shape[-1])
        foot_flat = foot.reshape(-1, foot.shape[-1])

        query = self.surround_proprio_mlp(critic_flat).unsqueeze(1)  # [B, 1, D]
        visual_tokens = torch.cat(
            [self.surround_cube_encoder(cube_flat), self.surround_foot_mlp(foot_flat).unsqueeze(1)], dim=1
        )
        fused = self.surround_cross_attn(query, visual_tokens)  # [B, 1, D]
        seq = torch.cat([fused, visual_tokens], dim=1)
        out = self.surround_self_attn(seq)[:, 0]
        zc_s = self.surround_head_zc(out)
        return zc_s.reshape(*lead_shape, -1)

    # ------------------------------------------------------------------
    # rsl-rl interface
    # ------------------------------------------------------------------

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def reset(self, dones=None):
        self.memory.reset(dones)

    def forward(self):
        raise NotImplementedError

    def update_distribution(self, actor_input: torch.Tensor):
        mean = self.actor(actor_input)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}")
        if not check_safe(mean):
            print("[ActorCriticMove] action mean has nan/inf")
        self.distribution = Normal(mean, std)

    @staticmethod
    def _actor_input(actor_obs: torch.Tensor, enc: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([actor_obs, enc["vel"], enc["z"], enc["zf"], enc["zc"]], dim=-1)

    def act(self, obs, masks=None, hidden_states=None):
        enc = self.encode_standard(obs, masks=masks, hidden_states=hidden_states)
        actor_obs = self.get_actor_obs(obs)
        if masks is not None:
            # cache padded latents for the PPOMove auxiliary losses; unpad for the actor path
            self.last_encoding = enc
            enc_u = {k: unpad_trajectories(v, masks) for k, v in enc.items()}
            actor_obs = unpad_trajectories(actor_obs, masks)
        else:
            self.last_encoding = None
            enc_u = enc
            self.estimated_vel = enc["vel"].detach()
        self.update_distribution(self._actor_input(actor_obs, enc_u))
        return self.distribution.sample()

    def act_inference(self, obs):
        enc = self.encode_standard(obs, deterministic=True)
        self.estimated_vel = enc["vel"].detach()
        return self.actor(self._actor_input(self.get_actor_obs(obs), enc))

    def evaluate(self, obs, masks=None, hidden_states=None):
        critic_obs = self.get_critic_obs(obs)
        cube, foot = self.get_surroundings_obs(obs)
        lead_shape = critic_obs.shape[:-1]
        critic_flat = critic_obs.reshape(-1, critic_obs.shape[-1])
        cube_flat = cube.reshape(-1, cube.shape[-1])
        foot_flat = foot.reshape(-1, foot.shape[-1])

        tokens = torch.cat(
            [
                self.critic_proprio_mlp(critic_flat).unsqueeze(1),
                self.critic_cube_encoder(cube_flat),
                self.critic_foot_mlp(foot_flat).unsqueeze(1),
            ],
            dim=1,
        )
        pooled = self.critic_transformer(tokens).mean(dim=1)
        values = self.critic(pooled).reshape(*lead_shape, 1)
        if masks is not None:
            values = unpad_trajectories(values, masks)
        return values

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def get_hidden_states(self):
        # the critic is feed-forward: duplicate the actor GRU state so the rollout storage's
        # (hid_a, hid_c) bookkeeping stays intact.
        return self.memory.hidden_states, self.memory.hidden_states

    def update_normalization(self, obs):
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(self.get_actor_obs(obs, normalize=False))
            self.history_obs_normalizer.update(self.get_history_obs(obs, normalize=False))
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs, normalize=False))

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True
