from isaaclab.utils.configclass import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg import (
    G1FlatPPORunnerCfg,
)


@configclass
class PPORunnerCfg(G1FlatPPORunnerCfg):
    """PPO configuration for the G1 DATN project."""

    def __post_init__(self):
        super().__post_init__()

        # Tách log/checkpoint của DATN khỏi task G1 chính thức
        self.experiment_name = "g1_datn"