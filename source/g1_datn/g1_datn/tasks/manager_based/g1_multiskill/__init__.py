import gymnasium as gym

gym.register(
    id="G1-MultiSkill-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.g1_multiskill_env_cfg:"
            "G1MultiSkillEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.agents.ppo_mlp_cfg:"
            "G1MultiSkillPPORunnerCfg"
        ),
    },
)