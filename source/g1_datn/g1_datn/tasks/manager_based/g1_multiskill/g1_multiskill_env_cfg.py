from importlib import import_module
import os
from pathlib import Path

from isaaclab.utils.configclass import configclass

# Import baseline G1-29DoF từ Unitree RL Lab.
# Dùng import_module vì tên package "29dof"
# không sử dụng được trong cú pháp from ... import.
base_module = import_module(
    "unitree_rl_lab.tasks.locomotion.robots.g1."
    "29dof.velocity_env_cfg"
)

BaseG1EnvCfg = base_module.RobotEnvCfg


@configclass
class G1MultiSkillEnvCfg(BaseG1EnvCfg):
    """G1 environment - Phase 0: Walking Baseline."""

    def __post_init__(self):
        super().__post_init__()

        # Resolve the upstream asset placeholder without changing Unitree's checkout.
        model_dir = os.environ.get("UNITREE_MODEL_DIR")
        if model_dir:
            self.scene.robot.spawn.usd_path = str(
                Path(model_dir) / "G1/29dof/usd/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd"
            )

        # Bắt đầu nhỏ để kiểm tra môi trường
        self.scene.num_envs = 64

        # Giữ nguyên:
        # - Robot G1-29DoF
        # - Observation và Action
        # - Reward và Termination
        # - PPO control frequency
