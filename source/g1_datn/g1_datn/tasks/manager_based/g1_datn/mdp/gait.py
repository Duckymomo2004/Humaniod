from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class GaitPhaseCommand(CommandTerm):
    """Gait phase dùng cho locomotion kiểu HoloSoma."""

    cfg: GaitPhaseCommandCfg

    def __init__(self, cfg: GaitPhaseCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.phase = torch.zeros(
            self.num_envs,
            2,
            device=self.device,
        )

        self.phase_offset = torch.zeros(
            self.num_envs,
            2,
            device=self.device,
        )

        self.gait_freq = (
            torch.ones(
                self.num_envs,
                1,
                device=self.device,
            )
            / cfg.gait_period
        )

    @property
    def command(self) -> torch.Tensor:
        return self.phase

    def reset(
        self,
        env_ids: Sequence[int] | None = None,
    ) -> dict[str, float]:

        if env_ids is None or isinstance(env_ids, slice):
            ids = torch.arange(
                self.num_envs,
                device=self.device,
                dtype=torch.long,
            )
        else:
            ids = torch.as_tensor(
                env_ids,
                device=self.device,
                dtype=torch.long,
            )

        # HoloSoma randomizes initial gait phase
        if self.cfg.randomize_phase:
            left_phase = torch.empty(
                len(ids),
                device=self.device,
            ).uniform_(
                -math.pi,
                math.pi,
            )
        else:
            left_phase = torch.zeros(
                len(ids),
                device=self.device,
            )

        self.phase_offset[ids, 0] = left_phase

        # Hai chân lệch phase pi
        self.phase_offset[ids, 1] = wrap_to_pi(
            left_phase - math.pi
        )

        self.phase[ids] = self.phase_offset[ids]

        return super().reset(env_ids)

    def _resample_command(
        self,
        env_ids: Sequence[int],
    ) -> None:

        ids = torch.as_tensor(
            env_ids,
            device=self.device,
            dtype=torch.long,
        )

        mean_freq = 1.0 / self.cfg.gait_period
        width = self.cfg.gait_frequency_randomization_width

        if width > 0.0:
            self.gait_freq[ids] = torch.empty(
                len(ids),
                1,
                device=self.device,
            ).uniform_(
                mean_freq - width,
                mean_freq + width,
            )
        else:
            self.gait_freq[ids] = mean_freq

    def _update_command(self) -> None:

        # Giống logic HoloSoma:
        # phase = episode_step * phase_dt + initial_phase
        phase = (
            self._env.episode_length_buf.unsqueeze(1)
            * (2.0 * math.pi * self._env.step_dt)
            * self.gait_freq
            + self.phase_offset
        )

        self.phase[:] = wrap_to_pi(phase)

        # Nếu command = 0 thì robot đứng
        velocity_command = self._env.command_manager.get_command(
            self.cfg.velocity_command_name
        )

        stand_mask = (
            torch.linalg.norm(
                velocity_command[:, :2],
                dim=1,
            )
            < 0.01
        ) & (
            torch.abs(
                velocity_command[:, 2]
            )
            < 0.01
        )

        self.phase[stand_mask] = self.cfg.stand_phase_value

    def _update_metrics(self) -> None:
        pass


@configclass
class GaitPhaseCommandCfg(CommandTermCfg):
    """Configuration của gait phase."""

    class_type: type = GaitPhaseCommand

    gait_period: float = 1.0

    # HoloSoma: 1 Hz ± 0.2 Hz
    gait_frequency_randomization_width: float = 0.2

    randomize_phase: bool = True

    stand_phase_value: float = math.pi

    velocity_command_name: str = "base_velocity"