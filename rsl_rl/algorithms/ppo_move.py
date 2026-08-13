"""PPO for MOVE (arXiv 2412.03353): PPO + PS-Net reconstruction and contrastive losses.

Extends the PPODreamWAQ pattern (auxiliary estimator losses folded into the joint PPO loss with a
single optimizer) to the MOVE PS-Net:

    L = L_ppo + L_reconstruction + L_contrast                       (MOVE Eq. 5)
    L_reconstruction = beta * KL(q(z) || N(0,1)) + MSE(o_hat_{t+1}, o_{t+1})
                       + MSE(v_hat, v) + MSE(c_hat_front, c_front)  (MOVE Eq. 6)
    L_contrast = -cos(p_s, sg(z_hat^c)) - cos(p_hat, sg(z_s^c))     (MOVE Eq. 7, SimSiam)

The policy is recurrent (GRU), so updates run on the recurrent minibatch generator: trajectories
are split at episode boundaries and padded. The o_{t+1} reconstruction target is the next-step
actor observation obtained by shifting inside each padded trajectory — the split guarantees the
shift never crosses a reset. Losses on padded latents are masked by the trajectory masks.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
import tensordict

from rsl_rl.modules.actor_critic_move import ActorCriticMove
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils.safety_utils import check_safe


class _MoveRolloutStorage(RolloutStorage):
    """Rollout storage that de-duplicates the critic hidden-state buffer.

    ActorCriticMove's critic is feed-forward: get_hidden_states() returns the actor GRU state in
    both slots. Alias the critic-side buffer to the actor's so the duplicate is never allocated
    (the recurrent generator still yields a (hid_a, hid_c) pair; the critic slot is ignored).
    """

    def _save_hidden_states(self, hidden_states):
        super()._save_hidden_states(hidden_states)
        if self.saved_hidden_states_c is not self.saved_hidden_states_a:
            self.saved_hidden_states_c = self.saved_hidden_states_a


class PPOMove:
    policy: ActorCriticMove

    def __init__(
        self,
        policy,
        num_learning_epochs=5,
        num_mini_batches=4,
        clip_param=0.2,
        gamma=0.99,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.01,
        vel_loss_coef=1.0,
        obs_recon_coef=1.0,
        depth_recon_coef=1.0,
        contrast_loss_coef=1.0,
        learning_rate=0.001,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="adaptive",
        desired_kl=0.01,
        device="cpu",
        normalize_advantage_per_mini_batch=False,
        # accepted for runner compatibility; not supported by this algorithm
        rnd_cfg: dict | None = None,
        symmetry_cfg: dict | None = None,
        multi_gpu_cfg: dict | None = None,
    ):
        if rnd_cfg is not None:
            raise NotImplementedError("PPOMove does not support RND.")
        if symmetry_cfg is not None:
            raise NotImplementedError(
                "PPOMove does not support symmetry augmentation (depth images need a dedicated mirror function)."
            )
        self.rnd = None

        # device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # PPO components
        self.policy = policy
        self.policy.to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.storage: RolloutStorage = None  # type: ignore
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.vel_loss_coef = vel_loss_coef
        self.obs_recon_coef = obs_recon_coef
        self.depth_recon_coef = depth_recon_coef
        self.contrast_loss_coef = contrast_loss_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

    def init_storage(self, training_type, num_envs, num_transitions_per_env, obs, actions_shape):
        self.storage = _MoveRolloutStorage(
            training_type, num_envs, num_transitions_per_env, obs, actions_shape, self.device
        )

    def test_mode(self):
        self.policy.eval()

    def train_mode(self):
        self.policy.train()

    def act(self, obs: tensordict.TensorDict):
        # per-step NaN guard on the small (1D state) groups only — scanning the image groups every
        # step forces large GPU syncs; images are sanitized in their obs terms and the storage's
        # compute_returns() non-finite scan still covers them once per iteration.
        for key in obs.keys():
            if obs[key].shape[-1] <= 512 and not check_safe(obs[key]):
                raise ValueError(f"Observation group '{key}' contains NaN/Inf values")
        # hidden states BEFORE stepping the recurrent encoder
        self.transition.hidden_states = self.policy.get_hidden_states()
        self.transition.actions = self.policy.act(obs).detach()
        self.transition.values = self.policy.evaluate(obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        self.transition.observations = obs
        return self.transition.actions

    def process_env_step(self, obs, rewards, dones, extras):
        # update the normalizers
        self.policy.update_normalization(obs)

        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # bootstrapping on time outs
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device), 1
            )

        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, obs):
        last_values = self.policy.evaluate(obs).detach()
        self.storage.compute_returns(
            last_values, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        )

    @staticmethod
    def _masked_mean(per_sample: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mean of per-sample losses [T, n_traj] over valid trajectory steps."""
        mask = mask.float()
        return (per_sample * mask).sum() / mask.sum().clamp(min=1.0)

    def update(self):  # noqa: C901
        beta = self.policy.cenet_beta
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_vel_loss = 0
        mean_obs_recon_loss = 0
        mean_depth_recon_loss = 0
        mean_kld_loss = 0
        mean_contrast_loss = 0

        generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for (
            obs_batch,  # padded trajectories [T, n_traj, ...]
            actions_batch,  # [T, env_slice, A]
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,  # [T, n_traj] bool
        ) in generator:

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # -- actor: recompute log-probs (unpadded) and cache padded latents for the aux losses
            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            # -- critic
            value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            # -- entropy
            mu_batch = self.policy.action_mean
            sigma_batch = self.policy.action_std
            entropy_batch = self.policy.entropy

            # adaptive learning rate (KL between old and new action distributions)
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # === PS-Net auxiliary losses (padded space, masked) ===
            enc = self.policy.last_encoding  # padded [T, n_traj, ...]
            mask = masks_batch

            # velocity estimation: v_hat vs ground-truth base lin vel (raw m/s)
            vel_target = self.policy.get_vel_target(obs_batch).detach()
            vel_loss = self._masked_mean(((enc["vel"] - vel_target) ** 2).sum(dim=-1), mask)

            # next-obs reconstruction: decode [v_hat, z] -> o_{t+1}; shift inside trajectories
            obs_decode = self.policy.obs_decoder(torch.cat([enc["vel"], enc["z"]], dim=-1))
            actor_obs_target = self.policy.get_actor_obs(obs_batch).detach()
            obs_recon_loss = self._masked_mean(
                ((obs_decode[:-1] - actor_obs_target[1:]) ** 2).sum(dim=-1), mask[1:]
            )

            # front-view reconstruction: decode z^f -> clean front cube-map face at time t
            cube_target, _ = self.policy.get_surroundings_obs(obs_batch)
            front_target = cube_target[..., : self.policy.cube_face_dim].detach()
            depth_decode = self.policy.depth_decoder(enc["zf"])
            depth_recon_loss = self._masked_mean(((depth_decode - front_target) ** 2).sum(dim=-1), mask)

            # KL divergence of the VAE latent
            kld = -0.5 * torch.sum(1 + enc["z_logvar"] - enc["z_mean"].pow(2) - enc["z_logvar"].exp(), dim=-1)
            kld_loss = self._masked_mean(kld, mask)

            # contrastive loss (MOVE Eq. 7): weight-sharing predictor + stop-gradient
            zc = enc["zc"]
            zc_s = self.policy.encode_surroundings(obs_batch)
            p_hat = self.policy.contrast_predictor(zc)  # standard-encoder branch
            p_s = self.policy.contrast_predictor(zc_s)  # surroundings-encoder branch
            cos = nn.functional.cosine_similarity
            contrast_per_sample = -cos(p_s, zc.detach(), dim=-1) - cos(p_hat, zc_s.detach(), dim=-1)
            contrast_loss = self._masked_mean(contrast_per_sample, mask)

            aux_loss = (
                self.vel_loss_coef * vel_loss
                + self.obs_recon_coef * obs_recon_loss
                + self.depth_recon_coef * depth_recon_loss
                + beta * kld_loss
                + self.contrast_loss_coef * contrast_loss
            )

            # === PPO losses (unpadded space) ===
            # note: squeeze only the trailing singleton — torch.squeeze() would also drop the
            # env dimension when a recurrent minibatch contains exactly one env
            ratio = torch.exp(actions_log_prob_batch - old_actions_log_prob_batch.squeeze(-1))
            surrogate = -advantages_batch.squeeze(-1) * ratio
            surrogate_clipped = -advantages_batch.squeeze(-1) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
                + aux_loss
            )

            if not check_safe(loss):
                raise ValueError(
                    "[PPOMove] non-finite total loss: "
                    f"surrogate={surrogate_loss.item():.4g} value={value_loss.item():.4g} "
                    f"vel={vel_loss.item():.4g} obs_recon={obs_recon_loss.item():.4g} "
                    f"depth_recon={depth_recon_loss.item():.4g} kld={kld_loss.item():.4g} "
                    f"contrast={contrast_loss.item():.4g}"
                )

            # gradient step
            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_vel_loss += vel_loss.item()
            mean_obs_recon_loss += obs_recon_loss.item()
            mean_depth_recon_loss += depth_recon_loss.item()
            mean_kld_loss += kld_loss.item()
            mean_contrast_loss += contrast_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_vel_loss /= num_updates
        mean_obs_recon_loss /= num_updates
        mean_depth_recon_loss /= num_updates
        mean_kld_loss /= num_updates
        mean_contrast_loss /= num_updates
        self.storage.clear()

        return {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "vel_reconstruction": mean_vel_loss,
            "obs_reconstruction": mean_obs_recon_loss,
            "depth_reconstruction": mean_depth_recon_loss,
            "kld": mean_kld_loss,
            "contrastive": mean_contrast_loss,
        }

    """
    Helper functions (multi-GPU), mirrored from PPODreamWAQ.
    """

    def broadcast_parameters(self):
        model_params = [self.policy.state_dict()]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.policy.load_state_dict(model_params[0])

    def reduce_parameters(self):
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        offset = 0
        for param in self.policy.parameters():
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                offset += numel
