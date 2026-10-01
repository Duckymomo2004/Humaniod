from isaaclab_fast_sac import FastSacAlgorithmCfg, FastSacRunnerCfg

from isaaclab.utils.configclass import configclass


@configclass
class G1FastSacRunnerCfg(FastSacRunnerCfg):
    seed: int = 42
    device: str = "cuda:0"

    max_iterations: int = 10000
    save_interval: int = 200

    experiment_name: str = "g1_datn_fast_sac"
    run_name: str = ""

    logger: str = "tensorboard"
    wandb_project: str = "g1_datn_fast_sac"

    obs_groups: dict = {
        "policy": ["policy"],
        "critic": ["policy"],
    }

    clip_actions: float | None = None

    resume: bool = False
    load_run: str = ".*"
    load_checkpoint: str = "model_.*.pt"

    algorithm: FastSacAlgorithmCfg = FastSacAlgorithmCfg(
        critic_learning_rate=3e-4,
        actor_learning_rate=3e-4,
        alpha_learning_rate=3e-4,

        buffer_size=1024,
        num_steps=1,

        gamma=0.97,
        tau=0.125,

        batch_size=8192,
        learning_starts=10,
        policy_frequency=4,
        num_updates=8,

        target_entropy_ratio=0.0,
        alpha_init=0.001,
        use_autotune=True,

        num_atoms=101,
        v_min=-20.0,
        v_max=20.0,

        critic_hidden_dim=768,
        actor_hidden_dim=512,

        use_tanh=True,
        log_std_max=0.0,
        log_std_min=-5.0,

        use_layer_norm=True,
        num_q_networks=2,

        max_grad_norm=0.0,
        weight_decay=0.001,

        compile=True,
        amp=True,
        amp_dtype="bf16",

        obs_normalization=True,


        use_symmetry=False,

        logging_interval=100,

        action_scale=None,
        action_bias=None,
    )
