# Copyright (c) 2025, Istituto Italiano di Tecnologia
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
from torch import autograd
from rsl_rl.utils import utils


class Discriminator(nn.Module):
    """Discriminator implements the discriminator network for the AMP algorithm.

    This network is trained to distinguish between expert and policy-generated data.
    It also provides reward signals for the policy through adversarial learning.

    Args:
        input_dim (int): Dimension of the concatenated input state (state + next state).
        hidden_layer_sizes (list): List of hidden layer sizes.
        reward_scale (float): Scale factor for the computed reward.
        device (str): Device to run the model on ('cpu' or 'cuda').
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layer_sizes: list[int],
        reward_scale: float,
        reward_clamp_epsilon: float = 0.0001,
        device: str = "cpu",
        task_reward_lerp: float = 0.0
    ):
        super(Discriminator, self).__init__()

        self.device = device
        self.input_dim = input_dim
        self.reward_scale = reward_scale
        self.reward_clamp_epsilon = reward_clamp_epsilon
        layers = []
        curr_in_dim = input_dim

        for hidden_dim in hidden_layer_sizes:
            layers.append(nn.Linear(curr_in_dim, hidden_dim))
            layers.append(nn.ReLU())
            curr_in_dim = hidden_dim

        self.trunk = nn.Sequential(*layers).to(device)
        self.linear = nn.Linear(hidden_layer_sizes[-1], 1).to(device)

        self.trunk.train()
        self.linear.train()
        self.task_reward_lerp = task_reward_lerp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the discriminator.

        Args:
            x (Tensor): Input tensor (batch_size, input_dim).

        Returns:
            Tensor: Discriminator output logits.
        """
        h = self.trunk(x)
        d = self.linear(h)
        return d

    def predict_reward(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        normalizer=None,
    ) -> torch.Tensor:
        """Predicts reward based on discriminator output using a log-style formulation.

        Args:
            state (Tensor): Current state tensor.
            next_state (Tensor): Next state tensor.
            normalizer (Optional): Optional state normalizer.

        Returns:
            Tensor: Computed adversarial reward.
        """
        with torch.no_grad():
            self.eval()
            if normalizer is not None:
                state = normalizer.normalize(state)
                next_state = normalizer.normalize(next_state)

            d = self.linear(self.trunk(torch.cat([state, next_state], dim=-1)))
            reward = self.reward_scale * torch.clamp(1 - (1/4) * torch.square(d - 1), min=0)
            # Avoid log(0) by clamping the input to a minimum threshold
            # reward = -torch.log(
            #     torch.maximum(
            #         1 - prob,
            #         torch.tensor(self.reward_clamp_epsilon, device=self.device),
            #     )
            # )

            # reward = self.reward_scale * reward
            self.train()
            return reward.squeeze()

    def compute_grad_pen(
        self,
        expert_state: torch.Tensor,
        expert_next_state: torch.Tensor,
        lambda_: float = 10,
    ) -> torch.Tensor:
        """Computes the gradient penalty used to regularize the discriminator.

        Args:
            expert_state (Tensor): A batch of expert states.
            expert_next_state (Tensor): A batch of expert next states.
            lambda_ (float): Penalty coefficient.

        Returns:
            Tensor: Gradient penalty value.
        """
        expert_data = torch.cat([expert_state, expert_next_state], dim=-1)
        expert_data.requires_grad = True

        disc = self.linear(self.trunk(expert_data))
        ones = torch.ones(disc.size(), device=disc.device)
        grad = autograd.grad(
            outputs=disc, inputs=expert_data,
            grad_outputs=ones, create_graph=True,
            retain_graph=True, only_inputs=True)[0]

        # Enforce that the grad norm approaches 0.
        grad_pen = lambda_ * (grad.norm(2, dim=1) - 0).pow(2).mean()
        return grad_pen

    def wgan_loss(self, policy_d, expert_d):
        """
        This loss function computes a modified Wasserstein loss for the discriminator.
        The original Wasserstein loss is D(policy) - D(expert), but here we apply a tanh
        transformation to the discriminator outputs scaled by eta_wgan. This helps in stabilizing the training.
        Args:
            policy_d (Tensor): Discriminator output for policy data.
            expert_d (Tensor): Discriminator output for expert data.
        """
        policy_d = torch.tanh(self.eta_wgan * policy_d)
        expert_d = torch.tanh(self.eta_wgan * expert_d)
        return policy_d.mean() - expert_d.mean()