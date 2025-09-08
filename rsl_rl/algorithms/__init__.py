# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implementation of different RL agents."""

from .distillation import Distillation
from .ppo import PPO
from .ppo_amp import PPOAMP
from .ppo_dreamwaq import PPODreamWAQ

__all__ = ["PPO", "Distillation", "PPOAMP", "PPODreamWAQ"]
