from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
from tensordict import TensorDict

from rsl_rl.networks import MLP, EmpiricalNormalization
from rsl_rl.utils.safety_utils import check_safe

class ActorCriticDWAQ(nn.Module):
    is_recurrent = False
    def __init__(
        self, 
        obs,
        obs_groups,
        num_actions,
        cenet_in_dim, 
        cenet_out_dim,
        cenet_encoder_hidden_dims=[128],
        cenet_decoder_hidden_dims=[128,64],
        cenet_decoder_out_dim=46,
        actor_obs_normalization=False,
        critic_obs_normalization=False, 
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu", 
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        history_length=3,
        obs_hist_dict=dict(),
        use_height_scan=False,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        # get the observation dimensions
        self.obs_groups = obs_groups
        num_actor_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_actor_obs += obs[obs_group].shape[-1]
        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_critic_obs += obs[obs_group].shape[-1]
        
        self.use_height_scan = use_height_scan
        self.history_length = history_length
        self.obs_hist_dict = obs_hist_dict
        
        # generate history indices since obs history stacks using AAABBBCCC instead of ABCABCABC
        # assume obs_history is a history of obs of length n, including the current obs
        # history is implemented as a circular buffer with first element being the oldest, last element being the latest
        self.history_indices = []
        sum = 0
        for key, dim in self.obs_hist_dict.items():
            for h in range(self.history_length - 1):
                if key != "height_scan":
                    self.history_indices += list(range(sum + h * dim, sum + (h + 1) * dim))
            sum += dim * self.history_length
        
        self.current_indices = []
        sum = 0
        for key, dim in self.obs_hist_dict.items():
            if key != "height_scan":
                self.current_indices += list(range(sum + (self.history_length - 1) * dim, sum + self.history_length * dim))
            sum += dim * self.history_length
        # print(self.history_indices)

        # actor
        self.actor = MLP(num_actor_obs + cenet_out_dim, num_actions, actor_hidden_dims, activation)
        # actor observation normalization
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            if use_height_scan:
                self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs - obs_hist_dict["height_scan"])
            else:
                self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()
        print(f"Actor MLP: {self.actor}")

        # critic
        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
        # critic observation normalization
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()
        print(f"Critic MLP: {self.critic}")
        
        self.history_obs_normalization = actor_obs_normalization
        if self.history_obs_normalization:
            self.history_obs_normalizer = EmpiricalNormalization(cenet_in_dim)
        else:
            self.history_obs_normalizer = torch.nn.Identity()

        # CENet
        # self.encoder = nn.Sequential(
        #     nn.Linear(cenet_in_dim,128),
        #     self.activation,
        #     nn.Linear(128,64),
        #     self.activation,
        # )
        self.encoder = MLP(cenet_in_dim,64,cenet_encoder_hidden_dims,activation)
        
        self.encode_mean_latent = nn.Linear(64,cenet_out_dim-3)
        self.encode_logvar_latent = nn.Linear(64,cenet_out_dim-3)
        self.encode_mean_vel = nn.Linear(64,3)
        self.encode_logvar_vel = nn.Linear(64,3)

        self.decoder = MLP(cenet_out_dim,cenet_decoder_out_dim,cenet_decoder_hidden_dims,activation)
        # self.decoder = nn.Sequential(
        #     nn.Linear(cenet_out_dim,64),
        #     self.activation,
        #     nn.Linear(64,128),
        #     self.activation,
        #     nn.Linear(128,45)
        # )

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    def reparameterise(self,mean,logvar):
        #clamp logvar to avoid inf or nan, based on VAE implementation by huggingface
        logvar = torch.clamp(logvar, min=-30.0, max=20.0)
        std = torch.exp(logvar*0.5)
        code_temp = torch.randn_like(std)
        code = mean + std*code_temp
        return code
    
    def cenet_forward(self,history_obs):
        distribution = self.encoder(history_obs)
        if not check_safe(distribution):
            print("cenet distribution has nan or inf")
        mean_latent = self.encode_mean_latent(distribution)
        if not check_safe(mean_latent):
            print("cenet mean_latent has nan or inf")
        logvar_latent = self.encode_logvar_latent(distribution)
        if not check_safe(logvar_latent):
            print("cenet logvar has nan or inf")
        # var = torch.exp(logvar_latent*0.5)
        # code_temp = torch.randn_like(var)
        # code = mean_latent + var*code_temp
        # print("latent : ",code[0])
        mean_vel = self.encode_mean_vel(distribution)
        if not check_safe(mean_vel):
            print("cenet mean_vel has nan or inf")
        logvar_vel = self.encode_logvar_vel(distribution)
        if not check_safe(logvar_vel):
            print("cenet logvar_vel has nan or inf")
        code_latent = self.reparameterise(mean_latent,logvar_latent)
        if not check_safe(code_latent):
            print("cenet code_latent has nan or inf")
        code_vel = self.reparameterise(mean_vel,logvar_vel)
        if not check_safe(code_vel):
            print("cenet code_vel has nan or inf")
        code = torch.cat((code_vel,code_latent),dim=-1)
        decode = self.decoder(code)
        if not check_safe(decode):
            print("cenet decode has nan or inf")
        return code,code_vel,decode,mean_vel,logvar_vel,mean_latent,logvar_latent

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    # def update_distribution(self, observations):
    #     mean = self.actor(observations)
    #     self.distribution = Normal(mean, mean * 0.0 + self.std)
    
    def update_distribution(self, observations):
        # compute mean
        mean = self.actor(observations)
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
            # std = torch.clamp(std, 0.001, 1.0)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        if torch.isnan(mean).any():
            print("action mean has nan")
            print(mean)
        if torch.isnan(std).any():
            print("action std has nan")
            print(std)
        if torch.isinf(mean).any():
            print("action mean has inf")
            print(mean)
        if torch.isinf(std).any():
            print("action std has inf")
            print(std)
        self.distribution = Normal(mean, std)

    def act(self, obs, **kwargs):
        actor_obs = self.get_actor_obs(obs)
        
        if not check_safe(actor_obs):
            print("actor obs has nan or inf")
        history_obs = self.get_history_obs(obs)
        if not check_safe(history_obs):
            print("history obs has nan or inf")
        code,_,decode,_,_,_,_ = self.cenet_forward(history_obs)
        if not check_safe(code):
            print("code has nan or inf")
        if self.use_height_scan:
            height_scan_obs = self.get_curr_height_scan_obs(obs)
            if not check_safe(height_scan_obs):
                print("height scan obs has nan or inf")
            observations = torch.cat((code,actor_obs,height_scan_obs),dim=-1)
        else:
            observations = torch.cat((code,actor_obs),dim=-1)
        if not check_safe(observations):
            print("observations has nan or inf")
            print(observations)
        self.update_distribution(observations)
        return self.distribution.sample()

    def act_inference(self, obs, **kwargs):
        actor_obs = self.get_actor_obs(obs)
        history_obs = self.get_history_obs(obs)
        code,_,decode,_,_,_,_ = self.cenet_forward(history_obs)
        if self.use_height_scan:
            height_scan_obs = self.get_curr_height_scan_obs(obs)
            observations = torch.cat((code,actor_obs,height_scan_obs),dim=-1)
        else:
            observations = torch.cat((code,actor_obs),dim=-1)
        return self.actor(observations)

    def evaluate(self, obs, **kwargs):
        obs = self.get_critic_obs(obs)
        value = self.critic(obs)
        return value
    
    def get_actor_obs(self, obs):
        obs_list = []
        # for obs_group in self.obs_groups["policy"]:
        #     obs_list.append(obs[obs_group])
        for obs_group in self.obs_groups["history"]:
            obs_list.append(obs[obs_group][:,self.current_indices])
        return self.actor_obs_normalizer(torch.cat(obs_list, dim=-1))

    def get_critic_obs(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["critic"]:
            obs_list.append(obs[obs_group])
        return self.critic_obs_normalizer(torch.cat(obs_list, dim=-1))
    
    def get_history_obs(self, obs):
        obs_list = []
        # print(self.history_indices.__len__())
        # print(obs[self.obs_groups["history"][0]].shape)
        for obs_group in self.obs_groups["history"]:
            obs_list.append(obs[obs_group][:,self.history_indices])
        return self.history_obs_normalizer(torch.cat(obs_list, dim=-1))
    
    def get_curr_height_scan_obs(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["history"]:
            obs_list.append(obs[obs_group][:,-self.obs_hist_dict["height_scan"]:])
        return torch.cat(obs_list, dim=-1)

    def get_zero_actor_obs(self, obs):
        zero_obs = TensorDict()
        for obs_group in self.obs_groups["policy"]:
            zero_obs[obs_group] = torch.zeros_like(obs[obs_group])
        return zero_obs

    def get_vel_target(self, obs):
        critic_obs = self.get_critic_obs(obs)
        critic_obs = self.critic_obs_normalizer(critic_obs)
        lin_vel = critic_obs[:,3:6]
        return lin_vel
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def update_normalization(self, obs):
        if self.actor_obs_normalization:
            actor_obs = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
            history_obs = self.get_history_obs(obs)
            self.history_obs_normalizer.update(history_obs)
        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """

        super().load_state_dict(state_dict, strict=strict)
        return True