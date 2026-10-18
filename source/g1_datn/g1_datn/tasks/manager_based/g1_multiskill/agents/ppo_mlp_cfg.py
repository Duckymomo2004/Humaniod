from isaaclab.utils.configclass import configclass

from unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg import (
    BasePPORunnerCfg,
)


@configclass
class G1MultiSkillPPORunnerCfg(BasePPORunnerCfg):
    """PPO configuration for G1 multi-skill research."""

    experiment_name = "g1_multiskill_baseline"
    max_iterations = 3000
    save_interval = 100