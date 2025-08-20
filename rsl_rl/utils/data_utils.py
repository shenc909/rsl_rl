# Copyright (c) 2025, Istituto Italiano di Tecnologia
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Tuple, Union

import torch


class RunningMeanStd:
    """
    Calculates the running mean and standard deviation of a data stream.
    Based on the parallel algorithm for calculating variance:
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

    Args:
        epsilon (float): Small constant to initialize the count for numerical stability.
        shape (Tuple[int, ...]): Shape of the data (e.g., observation shape).
    """

    def __init__(
        self,
        epsilon: float = 1e-4,
        shape: Tuple[int, ...] = (),
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.mean = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.var = torch.ones(shape, dtype=torch.float32, device=self.device)
        self.count = torch.tensor(epsilon, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def update(self, arr: torch.Tensor) -> None:
        """
        Updates the running statistics using a new batch of data.

        Args:
            arr (torch.Tensor): Batch of data (batch_size, *shape).
        """
        batch = arr.to(self.device, dtype=torch.float32)
        batch_mean = batch.mean(dim=0)
        batch_var = batch.var(dim=0, unbiased=False)
        batch_count = torch.tensor(
            batch.shape[0], dtype=torch.float32, device=self.device
        )
        self._update_from_moments(batch_mean, batch_var, batch_count)

    @torch.no_grad()
    def _update_from_moments(
        self,
        batch_mean: torch.Tensor,
        batch_var: torch.Tensor,
        batch_count: torch.Tensor,
    ) -> None:
        """
        Updates statistics using precomputed batch mean, variance, and count.

        Args:
            batch_mean (torch.Tensor): Mean of the batch.
            batch_var (torch.Tensor): Variance of the batch.
            batch_count (torch.Tensor): Number of samples in the batch.
        """
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total_count
        new_var = m2 / total_count

        self.mean.copy_(new_mean)
        self.var.copy_(new_var)
        self.count.copy_(total_count)


class Normalizer(RunningMeanStd):
    """
    A normalizer that uses running statistics to normalize inputs, with optional clipping.

    Args:
        input_dim (Tuple[int, ...]): Shape of the input observations.
        epsilon (float): Small constant added to variance to avoid division by zero.
        clip_obs (float): Maximum absolute value to clip the normalized observations.
    """

    def __init__(
        self,
        input_dim: Union[int, Tuple[int, ...]],
        epsilon: float = 1e-4,
        clip_obs: float = 10.0,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        shape = (input_dim,) if isinstance(input_dim, int) else tuple(input_dim)
        super().__init__(epsilon=epsilon, shape=shape, device=device)
        self.epsilon = epsilon
        self.clip_obs = clip_obs

    def normalize(self, input: torch.Tensor) -> torch.Tensor:
        """
        Normalizes input using running mean and std, and clips the result.

        Args:
            input (torch.Tensor): Input tensor to normalize.

        Returns:
            torch.Tensor: Normalized and clipped tensor.
        """
        x = input.to(self.device, dtype=torch.float32)
        std = (self.var + self.epsilon).sqrt()
        y = (x - self.mean) / std
        return torch.clamp(y, -self.clip_obs, self.clip_obs)

    @torch.no_grad()
    def update_normalizer(self, rollouts, expert_loader) -> None:
        """
        Updates running statistics using samples from both policy and expert trajectories.

        Args:
            rollouts: Object with method `feed_forward_generator_amp(...)`.
            expert_loader: Dataloader or similar object providing expert batches.
        """
        policy_generator = rollouts.feed_forward_generator_amp(
            None, mini_batch_size=expert_loader.batch_size
        )
        expert_generator = expert_loader.dataset.feed_forward_generator_amp(
            expert_loader.batch_size
        )

        for expert_batch, policy_batch in zip(expert_generator, policy_generator):
            batch = torch.cat((*expert_batch, *policy_batch), dim=0)
            self.update(batch)