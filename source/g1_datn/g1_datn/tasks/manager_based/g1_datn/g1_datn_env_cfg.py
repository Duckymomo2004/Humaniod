from isaaclab.utils.configclass import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import (
    G1FlatEnvCfg,
)


@configclass
class G1DatnEnvCfg(G1FlatEnvCfg):
    """Unitree G1 locomotion environment for DATN."""

    def __post_init__(self):
        super().__post_init__()

        # Scene

        # Debug với ít robot trước
        self.scene.num_envs = 4
        self.scene.env_spacing = 2.5

        # Episode length
        self.episode_length_s = 20.0

        # Commands

        # Bài toán ban đầu dễ hơn baseline:
        # đi tới vừa phải, đi ngang ít, quay chậm hơn
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)

        # ============================================================
        # Actions
        # ============================================================
        # q_des = q_default + scale * action
        self.actions.joint_pos.scale = 0.5

        # ============================================================
        # Rewards
        # ============================================================
        # Ưu tiên tracking vận tốc tiến
        self.rewards.track_lin_vel_xy_exp.weight = 1.5

        # Giảm ưu tiên quay trong giai đoạn đầu
        self.rewards.track_ang_vel_z_exp.weight = 0.5

        # Giữ torso thẳng
        self.rewards.flat_orientation_l2.weight = -1.0

        # Giữ action mượt
        self.rewards.action_rate_l2.weight = -0.005

        # Hạn chế trượt chân
        self.rewards.feet_slide.weight = -0.1