from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def sin_phase(
    env: ManagerBasedRLEnv,
    command_name: str = "gait_phase",
) -> torch.Tensor:
    phase = env.command_manager.get_command(command_name)

    return torch.sin(phase)


def cos_phase(
    env: ManagerBasedRLEnv,
    command_name: str = "gait_phase",
) -> torch.Tensor:
    phase = env.command_manager.get_command(command_name)

    return torch.cos(phase)