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

# health logging: the PPO-vs-auxiliary gradient-norm split costs two extra backward passes through
# the retained graph, so it is sampled once every N update() calls instead of every minibatch.
GRAD_NORM_LOG_INTERVAL = 50
# failure budget for that split: a transient failure (an OOM under ``retain_graph=True``) must not
# cost the whole run's diagnostic, so it is retried; a persistent one is given up on after this many
# tries rather than re-raised on every update.
GRAD_NORM_MAX_FAILURES = 3


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

        # health logging state: one update() call == one learning iteration
        self._update_count = 0
        self._grad_norm_sampling_enabled = True
        self._grad_norm_failures = 0

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

    @staticmethod
    def _grad_norm_of(loss: torch.Tensor, params: list[torch.nn.Parameter]) -> float:
        """L2 norm of d(loss)/d(params), WITHOUT touching param.grad.

        Uses torch.autograd.grad(retain_graph=True): the gradients are returned as a fresh tuple and
        discarded here, so the real optimizer step still sees exactly the gradients produced by the
        single loss.backward() below -- no double-counting, no change to the update. Peak extra
        memory is one parameter-sized buffer (~7.9 MB fp32 for ActorCriticMove's 1.97 M params).
        """
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        sq_sum = torch.zeros((), device=loss.device)
        for grad in grads:
            if grad is not None:
                sq_sum += grad.pow(2).sum()
        return sq_sum.sqrt().item()

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
        # -- health metrics
        mean_clip_fraction = 0
        mean_explained_var = 0
        mean_grad_norm = 0
        sum_grad_norm_ppo = 0.0
        sum_grad_norm_aux = 0.0
        kl_per_minibatch: list[float] = []
        sample_grad_norms = self._grad_norm_sampling_enabled and self._update_count % GRAD_NORM_LOG_INTERVAL == 0

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

            # KL between old and new action distributions. Computed for every minibatch (not only
            # under the adaptive schedule) and recorded per (epoch, minibatch), so the KL's
            # evolution WITHIN an update is visible and not just its mean; the adaptive-lr branch
            # below consumes the very same value, so the schedule is unchanged.
            with torch.no_grad():
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
            kl_per_minibatch.append(kl_mean.item())

            # adaptive learning rate
            if self.desired_kl is not None and self.schedule == "adaptive":
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

            # velocity estimation: v_hat vs ground-truth base lin vel (raw m/s).
            #
            # 2026-09-21: MEAN over the feature axis, and plain MSE rather than Huber. MOVE Eq. 6 and
            # PIE both state the reconstruction losses are MSE ("mean-square error"); summing over
            # the feature axis computes n x MSE, which silently weighted these three terms by their
            # dimension counts (576 / 45 / 3) behind coefficients that all read 1.0. A measured
            # per-term gradient split at ck25000 found depth_recon owning 96% of the whole-model
            # gradient norm (229.6 of 238.8) and out-gradienting the surrogate 320-425x on the shared
            # trunk, so the pre-clip norm sat at 238.8 against max_grad_norm=1.0 and the policy
            # gradient was attenuated ~240x. Taking the mean restores Eq. 6's unit-weight balance and
            # makes it resolution-independent -- neither paper states the cube-face resolution, so
            # under the sum convention the term balance depended on an unspecified number.
            #
            # HUBER RETAINED, only the reduction changed. Gap-trench falls produce multi-m/s targets
            # no estimator can predict; squared error on those spikes 20-40x and, through the joint
            # clip, wastes the whole update. Measured 2026-09-21: swapping this term to plain MSE
            # took its gradient norm to 1.758 vs 0.85 predicted from the pure 1/3 rescale -- i.e. the
            # squared kernel DOUBLED it on outliers, on a fresh level-0 rollout that contains few
            # falls. Under the real curriculum it would be worse, and vel would outrank obs_recon.
            # So the kernel stays Huber (delta=1.0, linear above 1 m/s) and only the feature-axis
            # reduction becomes a mean, which is the part that fixes the dimension-count weighting.
            # This is a knowing, documented departure from Eq. 6's stated MSE -- the only one left in
            # the reconstruction group.
            vel_target = self.policy.get_vel_target(obs_batch).detach()
            vel_loss = self._masked_mean(
                torch.nn.functional.huber_loss(
                    enc["vel"], vel_target, reduction="none", delta=1.0
                ).mean(dim=-1),
                mask,
            )

            # next-obs reconstruction: decode [v_hat, z] -> o_{t+1}; shift inside trajectories
            obs_decode = self.policy.obs_decoder(torch.cat([enc["vel"], enc["z"]], dim=-1))
            actor_obs_target = self.policy.get_actor_obs(obs_batch).detach()
            obs_recon_loss = self._masked_mean(
                ((obs_decode[:-1] - actor_obs_target[1:]) ** 2).mean(dim=-1), mask[1:]
            )

            # front-view reconstruction: decode z^f -> clean front cube-map face at time t
            cube_target, _ = self.policy.get_surroundings_obs(obs_batch)
            front_target = cube_target[..., : self.policy.cube_face_dim].detach()
            depth_decode = self.policy.depth_decoder(enc["zf"])
            depth_recon_loss = self._masked_mean(((depth_decode - front_target) ** 2).mean(dim=-1), mask)

            # KL divergence of the VAE latent. Deliberately still a SUM over the 16 latent dims:
            # the analytic KL of a diagonal Gaussian is a per-dimension sum by definition, and Eq. 6
            # writes D_KL as a single term, not a per-dimension average. Meaning it here would make
            # beta 16x weaker than the paper's unit weight. NOTE this makes kld the largest aux
            # gradient after the change (~12.1 at ck25000, vs surrogate 1.47) -- the imbalance moves
            # from depth_recon to kld rather than disappearing.
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

            ppo_loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()
            loss = ppo_loss + aux_loss

            # -- health metrics. NOTE: "ppo_clip_fraction" is the share of samples whose importance
            # ratio leaves the trust region; it is unrelated to the reward-side Episode/clip_fraction
            # (share of steps whose total reward is clipped to zero) and must never be conflated.
            with torch.no_grad():
                mean_clip_fraction += ((ratio - 1.0).abs() > self.clip_param).float().mean().item()
                # explained variance of the ROLLOUT-time value (``target_values_batch``), not of
                # ``value_batch``: the latter has already taken gradient steps on the very returns it
                # would be scored against, so it measures training-set fit and its bias grows with
                # the effective critic step size, making two runs incomparable on this tag.
                returns_var = returns_batch.var()
                mean_explained_var += (
                    1.0 - (returns_batch - target_values_batch).var() / returns_var.clamp(min=1e-8)
                ).item()

            if not check_safe(loss):
                raise ValueError(
                    "[PPOMove] non-finite total loss: "
                    f"surrogate={surrogate_loss.item():.4g} value={value_loss.item():.4g} "
                    f"vel={vel_loss.item():.4g} obs_recon={obs_recon_loss.item():.4g} "
                    f"depth_recon={depth_recon_loss.item():.4g} kld={kld_loss.item():.4g} "
                    f"contrast={contrast_loss.item():.4g}"
                )

            # -- sampled gradient-norm split: does the auxiliary/VAE gradient dwarf the policy
            # gradient inside the single shared Adam + single grad clip? Rank-local and pre-reduce
            # (the ratio, not the absolute scale, is the quantity of interest).
            if sample_grad_norms:
                params = [p for p in self.policy.parameters() if p.requires_grad]
                try:
                    sum_grad_norm_ppo += self._grad_norm_of(ppo_loss, params)
                    sum_grad_norm_aux += self._grad_norm_of(aux_loss, params)
                except RuntimeError as err:
                    # a second backward through the retained cuDNN GRU graph, or a transient OOM
                    # (``torch.cuda.OutOfMemoryError`` is a ``RuntimeError``), are the plausible
                    # failures; instrumentation must never take down a 30k-iteration run. A transient
                    # OOM must not cost the whole run either, so this retries at the next sampling
                    # opportunity and only gives up after ``GRAD_NORM_MAX_FAILURES`` of them. Whether
                    # sampling is still alive is echoed to TensorBoard every iteration below -- a
                    # console print alone is invisible in a 30k-line log.
                    self._grad_norm_failures += 1
                    sample_grad_norms = False
                    if self._grad_norm_failures >= GRAD_NORM_MAX_FAILURES:
                        self._grad_norm_sampling_enabled = False
                        print(
                            f"[PPOMove] WARNING gradient-norm sampling PERMANENTLY DISABLED after"
                            f" {self._grad_norm_failures} failures: {err}",
                            flush=True,
                        )
                    else:
                        print(
                            f"[PPOMove] WARNING gradient-norm sampling failed"
                            f" ({self._grad_norm_failures}/{GRAD_NORM_MAX_FAILURES}), will retry: {err}",
                            flush=True,
                        )

            # gradient step
            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            # clip_grad_norm_ returns the PRE-clip total norm of the (already reduced) gradient
            mean_grad_norm += nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm).item()
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
        mean_clip_fraction /= num_updates
        mean_explained_var /= num_updates
        mean_grad_norm /= num_updates
        self.storage.clear()

        # the runner logs every key of this dict as TensorBoard tag "Loss/<key>"
        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "vel_reconstruction": mean_vel_loss,
            "obs_reconstruction": mean_obs_recon_loss,
            "depth_reconstruction": mean_depth_recon_loss,
            "kld": mean_kld_loss,
            "contrastive": mean_contrast_loss,
            "ppo_clip_fraction": mean_clip_fraction,
            "explained_variance": mean_explained_var,
            "grad_norm": mean_grad_norm,
            "kl": sum(kl_per_minibatch) / max(len(kl_per_minibatch), 1),
        }
        # per-(epoch, minibatch) KL -- the recurrent generator is epoch-major, so the i-th minibatch
        # of the update is epoch i // num_mini_batches, minibatch i % num_mini_batches
        for i, kl_value in enumerate(kl_per_minibatch):
            loss_dict[f"kl_e{i // self.num_mini_batches}m{i % self.num_mini_batches}"] = kl_value
        # 0/1 heartbeat: the three grad-norm tags below are sampled every GRAD_NORM_LOG_INTERVAL
        # updates and vanish entirely once sampling is disabled, which is indistinguishable in
        # TensorBoard from "wrong run open". This tag is always present, so the absence is legible.
        loss_dict["grad_norm_sampling_enabled"] = float(self._grad_norm_sampling_enabled)
        if sample_grad_norms:
            grad_norm_ppo = sum_grad_norm_ppo / num_updates
            grad_norm_aux = sum_grad_norm_aux / num_updates
            loss_dict["grad_norm_ppo"] = grad_norm_ppo
            loss_dict["grad_norm_aux"] = grad_norm_aux
            loss_dict["grad_norm_aux_over_ppo"] = grad_norm_aux / max(grad_norm_ppo, 1e-12)

        self._update_count += 1
        return loss_dict

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
