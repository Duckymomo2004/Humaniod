# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, wrap_to_pi, yaw_quat

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_target_l2(
    env: ManagerBasedRLEnv,
    target: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""

    asset: Articulation = env.scene[asset_cfg.name]

    joint_pos = wrap_to_pi(
        asset.data.joint_pos[:, asset_cfg.joint_ids]
    )

    return torch.sum(
        torch.square(joint_pos - target),
        dim=1,
    )


def pose(
    env: ManagerBasedRLEnv,
    pose_weights: list[float],
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize deviation from the default joint pose.

    Port of HoloSoma locomotion pose reward.
    """

    asset: Articulation = env.scene[asset_cfg.name]

    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    default_joint_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]

    weights = torch.tensor(
        pose_weights,
        device=env.device,
        dtype=joint_pos.dtype,
    )

    if weights.numel() != joint_pos.shape[1]:
        raise ValueError(
            f"pose_weights has {weights.numel()} values, "
            f"but {joint_pos.shape[1]} joints were selected."
        )

    error = torch.square(joint_pos - default_joint_pos)

    return torch.sum(
        error * weights.unsqueeze(0),
        dim=1,
    )


def penalty_close_feet_xy(
    env: ManagerBasedRLEnv,
    close_feet_threshold: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize feet that are too close in the lateral direction."""

    asset: Articulation = env.scene[asset_cfg.name]

    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]

    if foot_pos_w.shape[1] != 2:
        raise ValueError(
            "penalty_close_feet_xy requires exactly two foot bodies."
        )

    left_pos = foot_pos_w[:, 0, :]
    right_pos = foot_pos_w[:, 1, :]

    # Vector from right foot to left foot
    feet_delta_w = left_pos - right_pos

    # Convert only according to base yaw.
    base_yaw_quat = yaw_quat(asset.data.root_quat_w.torch)

    feet_delta_b = quat_apply_inverse(
        base_yaw_quat,
        feet_delta_w,
    )

    # Lateral distance = Y direction in robot frame
    lateral_distance = torch.abs(feet_delta_b[:, 1])

    return (
        lateral_distance < close_feet_threshold
    ).float()


def penalty_feet_ori(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize feet that are not parallel to the ground."""

    asset: Articulation = env.scene[asset_cfg.name]

    foot_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids, :]

    if foot_quat_w.shape[1] != 2:
        raise ValueError(
            "penalty_feet_ori requires exactly two foot bodies."
        )

    num_envs = foot_quat_w.shape[0]

    gravity_w = torch.tensor(
        [0.0, 0.0, -1.0],
        device=env.device,
        dtype=foot_quat_w.dtype,
    )

    gravity_w = gravity_w.view(1, 1, 3).expand(
        num_envs,
        2,
        3,
    )

    gravity_foot = quat_apply_inverse(
        foot_quat_w.reshape(-1, 4),
        gravity_w.reshape(-1, 3),
    ).reshape(num_envs, 2, 3)

    left_error = torch.norm(
        gravity_foot[:, 0, :2],
        dim=1,
    )

    right_error = torch.norm(
        gravity_foot[:, 1, :2],
        dim=1,
    )

    return left_error + right_error

def _expected_foot_height(
    phase: torch.Tensor,
    swing_height: float,
) -> torch.Tensor:
    """Desired foot height from gait phase."""

    def cubic_bezier(
        y_start: torch.Tensor,
        y_end: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:

        y_diff = y_end - y_start

        bezier = (
            x**3
            + 3.0 * x**2 * (1.0 - x)
        )

        return y_start + y_diff * bezier

    x = (phase + torch.pi) / (2.0 * torch.pi)

    up = cubic_bezier(
        torch.zeros_like(x),
        torch.full_like(x, swing_height),
        2.0 * x,
    )

    down = cubic_bezier(
        torch.full_like(x, swing_height),
        torch.zeros_like(x),
        2.0 * x - 1.0,
    )

    return torch.where(
        x <= 0.5,
        up,
        down,
    )


def feet_phase(
    env: ManagerBasedRLEnv,
    swing_height: float,
    tracking_sigma: float,
    asset_cfg: SceneEntityCfg,
    command_name: str = "gait_phase",
) -> torch.Tensor:
    """Track desired foot height using gait phase."""

    asset: Articulation = env.scene[asset_cfg.name]

    phase = env.command_manager.get_command(command_name)

    # [num_envs, 2]
    foot_z = asset.data.body_pos_w[
        :, asset_cfg.body_ids, 2
    ]

    # Flat terrain:
    # height relative to ground
    foot_z = (
        foot_z
        - env.scene.env_origins[:, 2].unsqueeze(1)
    )

    expected_z = _expected_foot_height(
        phase,
        swing_height,
    )

    error = torch.square(
        foot_z - expected_z
    ).sum(dim=1)

    return torch.exp(
        -error / tracking_sigma
    )
