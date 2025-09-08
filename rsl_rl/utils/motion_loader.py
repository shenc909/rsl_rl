# Copyright (c) 2025, Istituto Italiano di Tecnologia
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path
from typing import List, Union, Tuple, Generator
from dataclasses import dataclass

import torch
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d


@dataclass
class MotionData:
    """
    Data class representing motion data for humanoid agents.

    This class stores joint positions and velocities, base velocities (both in local
    and mixed/world frames), and base orientation (as quaternion). It offers utilities
    for preparing data in AMP-compatible format, as well as environment reset states.

    Attributes:
        - joint_positions: shape (T, 12)
        - joint_velocities: shape (T, 12)
        - base_lin_velocities_local: linear velocity in local (body) frame (T, 3)
        - base_ang_velocities_local: angular velocity in local (body) frame (T, 3)
        - proj_gravity: projected gravity vector in local (body) frame (T, 3)

    Notes:
        - All data is converted to torch.Tensor on the specified device during initialization.
    """

    joint_positions: Union[torch.Tensor, np.ndarray]
    joint_velocities: Union[torch.Tensor, np.ndarray]
    base_lin_velocities_local: Union[torch.Tensor, np.ndarray]
    base_ang_velocities_local: Union[torch.Tensor, np.ndarray]
    projected_gravity: Union[torch.Tensor, np.ndarray]
    device: str = "cpu"

    def __post_init__(self) -> None:
        # Convert numpy arrays (or SciPy Rotations) to torch tensors
        def to_tensor(x):
            return torch.tensor(x, device=self.device, dtype=torch.float32)

        if isinstance(self.joint_positions, np.ndarray):
            self.joint_positions = to_tensor(self.joint_positions)
        if isinstance(self.joint_velocities, np.ndarray):
            self.joint_velocities = to_tensor(self.joint_velocities)
        if isinstance(self.base_lin_velocities_local, np.ndarray):
            self.base_lin_velocities_local = to_tensor(self.base_lin_velocities_local)
        if isinstance(self.base_ang_velocities_local, np.ndarray):
            self.base_ang_velocities_local = to_tensor(self.base_ang_velocities_local)
        if isinstance(self.projected_gravity, np.ndarray):
            self.projected_gravity = to_tensor(self.projected_gravity)

    def __len__(self) -> int:
        return self.joint_positions.shape[0]

    def get_amp_dataset_obs(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Returns the AMP observation tensor for given indices.

        Args:
            indices: indices of samples to retrieve

        Returns:
            Concatenated observation tensor
        """
        return torch.cat(
            (
                self.joint_positions[indices],
                self.joint_velocities[indices],
                self.base_lin_velocities_local[indices],
                self.base_ang_velocities_local[indices],
                self.projected_gravity[indices],
            ),
            dim=1,
        )

    def get_state_for_reset(self, indices: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Returns the full state needed for environment reset.

        Args:
            indices: indices of samples to retrieve

        Returns:
            Tuple of (joint_positions, joint_velocities, base_lin_velocities, base_ang_velocities, projected_gravity)
        """
        return (
            self.joint_positions[indices],
            self.joint_velocities[indices],
            self.base_lin_velocities_local[indices],
            self.base_ang_velocities_local[indices],
            self.projected_gravity[indices],
        )

    def get_random_sample_for_reset(self, items: int = 1) -> Tuple[torch.Tensor, ...]:
        indices = torch.randint(0, len(self), (items,), device=self.device)
        return self.get_state_for_reset(indices)


class AMPLoader:
    """
    Loader and processor for humanoid motion capture datasets in AMP format.

    Responsibilities:
      - Loading .npy files containing motion data
      - Building a unified joint ordering across all datasets
      - Resampling trajectories to match the simulator's timestep
      - Computing derived quantities (velocities, local-frame motion)
      - Returning torch-friendly MotionData instances

    Dataset format:
        Each .npy contains a dict with keys:
          - "joints_list": List[str]
          - "joint_positions": List[np.ndarray]
          - "root_position": List[np.ndarray]
          - "root_quaternion": List[np.ndarray] (xyzw)
          - "fps": float (frames/sec)

    Args:
        device: Target torch device ('cpu' or 'cuda')
        dataset_path_root: Directory containing the .npy motion files
        dataset_names: List of dataset filenames (no extension)
        dataset_weights: List of sampling weights (for minibatch sampling)
        simulation_dt: Timestep used by the simulator
        slow_down_factor: Integer factor to slow down original data
    """

    def __init__(
        self,
        device: str,
        dataset_path_root: Path,
        dataset_names: List[str],
        # dataset_weights: List[float],
        # simulation_dt: float,
        # slow_down_factor: int = 1,
    ) -> None:
        self.device = device
        if isinstance(dataset_path_root, str):
            dataset_path_root = Path(dataset_path_root)

        # Load and process each dataset into MotionData
        self.motion_data: List[MotionData] = []
        for dataset_name in dataset_names:
            dataset_path = dataset_path_root / f"{dataset_name}.npy"
            md = self.load_data(
                dataset_path,
                # simulation_dt,
                # slow_down_factor,
            )
            self.motion_data.extend(md)

        # Normalize dataset-level sampling weights
        dataset_weights = [1.0] * len(self.motion_data)
        weights = torch.tensor(dataset_weights, dtype=torch.float32, device=self.device)
        self.dataset_weights = weights / weights.sum()

        # Precompute flat buffers for fast sampling
        obs_list, next_obs_list, reset_states = [], [], []
        for data, w in zip(self.motion_data, self.dataset_weights):
            T = len(data)
            idx = torch.arange(T, device=self.device)
            obs = data.get_amp_dataset_obs(idx)
            next_idx = torch.clamp(idx + 1, max=T - 1)
            next_obs = data.get_amp_dataset_obs(next_idx)

            obs_list.append(obs)
            next_obs_list.append(next_obs)

            jp, jv, blv, bav, pg = data.get_state_for_reset(idx)
            reset_states.append(torch.cat([jp, jv, blv, bav, pg], dim=1))

        self.all_obs = torch.cat(obs_list, dim=0)
        self.all_next_obs = torch.cat(next_obs_list, dim=0)
        self.all_states = torch.cat(reset_states, dim=0)

        # Build per-frame sampling weights: weight_i / length_i
        lengths = [len(d) for d in self.motion_data]
        per_frame = torch.cat(
            [
                torch.full((L,), w / L, device=self.device)
                for w, L in zip(self.dataset_weights, lengths)
            ]
        )
        self.per_frame_weights = per_frame / per_frame.sum()

    # def _resample_data_Rn(
    #     self,
    #     data: List[np.ndarray],
    #     original_keyframes,
    #     target_keyframes,
    # ) -> np.ndarray:
    #     f = interp1d(original_keyframes, data, axis=0)
    #     return f(target_keyframes)

    # def _resample_data_SO3(
    #     self,
    #     raw_quaternions: List[np.ndarray],
    #     original_keyframes,
    #     target_keyframes,
    # ) -> Rotation:

    #     # the quaternion is expected in the dataset as `xyzw` format (SciPy default)
    #     tmp = Rotation.from_quat(raw_quaternions)
    #     slerp = Slerp(original_keyframes, tmp)
    #     return slerp(target_keyframes)

    # def _compute_ang_vel(
    #     self,
    #     data: List[Rotation],
    #     dt: float,
    #     local: bool = False,
    # ) -> np.ndarray:
    #     R_prev = data[:-1]
    #     R_next = data[1:]

    #     if local:
    #         # Exp = R_i⁻¹ · R_{i+1}
    #         rel = R_prev.inv() * R_next
    #     else:
    #         # Exp = R_{i+1} · R_i⁻¹
    #         rel = R_next * R_prev.inv()

    #     # Log-map to rotation vectors and divide by Δt
    #     rotvec = rel.as_rotvec() / dt

    #     return np.vstack((rotvec, rotvec[-1]))

    def _compute_raw_derivative(self, data: np.ndarray, dt: float) -> np.ndarray:
        d = (data[1:] - data[:-1]) / dt
        return np.vstack([d, d[-1:]])

    def load_data(
        self,
        dataset_path: Path,
        # simulation_dt: float,
        # slow_down_factor: int = 1,
    ) -> List[MotionData]:
        """
        Loads and processes one motion dataset.

        dataset_path: Path to the dataset file
        
        Note: Each dataset file is a dictionary containing a torch.Tensor of shape (T, N, D)
        T: timesteps recorded
        N: number of environments/parallel simulations
        D: dimension of observation

        Returns:
            MotionData instance
        """
        
        md_arr: List[MotionData] = []

        data: dict[str, torch.Tensor] = np.load(str(dataset_path), allow_pickle=True).item()

        # num_timesteps_in_data = (data["velocity_commands"]).shape[0]
        num_envs = (data["velocity_commands"]).shape[1]

        for i in range(num_envs):
            jp_list = data["joint_pos"][:, i, :]
            jv_list = data["joint_vel"][:, i, :]
            blv_list = data["base_lin_vel"][:, i, :]
            bav_list = data["base_ang_vel"][:, i, :]
            pg_list = data["projected_gravity"][:, i, :]
            
            md_arr.append(
                MotionData(
                    joint_positions=jp_list.to(self.device),
                    joint_velocities=jv_list.to(self.device),
                    base_lin_velocities_local=blv_list.to(self.device),
                    base_ang_velocities_local=bav_list.to(self.device),
                    projected_gravity=pg_list.to(self.device),
                    device=self.device
                )
            )

        # dt = 1.0 / data["fps"] / float(slow_down_factor)
        # T = len(jp_list)
        # t_orig = np.linspace(0, T * dt, T)
        # T_new = int(T * dt / simulation_dt)
        # t_new = np.linspace(0, T * dt, T_new)

        # resampled_joint_positions = self._resample_data_Rn(jp_list, t_orig, t_new)
        # resampled_joint_velocities = self._compute_raw_derivative(
        #     resampled_joint_positions, simulation_dt
        # )

        # resampled_base_positions = self._resample_data_Rn(
        #     data["root_position"], t_orig, t_new
        # )
        # resampled_base_orientations = self._resample_data_SO3(
        #     data["root_quaternion"], t_orig, t_new
        # )

        # resampled_base_lin_vel_mixed = self._compute_raw_derivative(
        #     resampled_base_positions, simulation_dt
        # )

        # resampled_base_ang_vel_mixed = self._compute_ang_vel(
        #     resampled_base_orientations, simulation_dt, local=False
        # )

        # resampled_base_lin_vel_local = np.stack(
        #     [
        #         R.as_matrix().T @ v
        #         for R, v in zip(
        #             resampled_base_orientations, resampled_base_lin_vel_mixed
        #         )
        #     ]
        # )
        # resampled_base_ang_vel_local = self._compute_ang_vel(
        #     resampled_base_orientations, simulation_dt, local=True
        # )

        return md_arr

    def feed_forward_generator(
        self, num_mini_batch: int, mini_batch_size: int
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        """
        Yields mini-batches of (state, next_state) pairs for training,
        sampled directly from precomputed buffers.

        Args:
            num_mini_batch: Number of mini-batches to yield
            mini_batch_size: Size of each mini-batch
        Yields:
            Tuple of (state, next_state) tensors
        """
        for _ in range(num_mini_batch):
            idx = torch.multinomial(
                self.per_frame_weights, mini_batch_size, replacement=True
            )
            yield self.all_obs[idx], self.all_next_obs[idx]

    def get_state_for_reset(self, number_of_samples: int) -> Tuple[torch.Tensor, ...]:
        """
        Randomly samples full states for environment resets,
        sampled directly from the precomputed state buffer.

        Args:
            number_of_samples: Number of samples to retrieve
        Returns:
            Tuple of (joint_positions, joint_velocities, base_lin_velocities, base_ang_velocities, projected_gravity)
        """
        idx = torch.multinomial(
            self.per_frame_weights, number_of_samples, replacement=True
        )
        full = self.all_states[idx]
        joint_dim = self.motion_data[0].joint_positions.shape[1]

        # The dimensions of the full state are:
        #   - joint_dim (joint_positions) + joint_dim (joint_velocities)
        #   + 3 (base_lin_velocities) + 3 (base_ang_velocities) + 3 (projected_gravity)
        #   = joint_dim + joint_dim + 3 + 3 + 3
        dims = [joint_dim, joint_dim, 3, 3, 3]
        return torch.split(full, dims, dim=1)