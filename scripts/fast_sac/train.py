"""Train G1 with FastSAC."""

import argparse
import os
import sys
from datetime import datetime

# local imports
import cli_args

# Register custom G1 task
import g1_datn.tasks  # noqa: F401
import gymnasium as gym
import torch
from isaaclab_fast_sac import (
    FastSacRunner,
    FastSacRunnerCfg,
    FastSacVecEnvWrapper,
)

from isaaclab.envs import (
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import (
    add_launcher_args,
    get_checkpoint_path,
    launch_simulation,
    setup_preset_cli,
)
from isaaclab_tasks.utils.hydra import hydra_task_config

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


parser = argparse.ArgumentParser(
    description="Train G1 with FastSAC."
)

parser.add_argument(
    "--video",
    action="store_true",
    default=False,
)

parser.add_argument(
    "--video_length",
    type=int,
    default=200,
)

parser.add_argument(
    "--video_interval",
    type=int,
    default=2000,
)

parser.add_argument(
    "--num_envs",
    type=int,
    default=None,
)

parser.add_argument(
    "--task",
    type=str,
    default=None,
)

parser.add_argument(
    "--seed",
    type=int,
    default=None,
)

parser.add_argument(
    "--max_iterations",
    type=int,
    default=None,
)

parser.add_argument(
    "--distributed",
    action="store_true",
    default=False,
)

cli_args.add_fast_sac_args(parser)
add_launcher_args(parser)

args_cli, remaining_args = setup_preset_cli(parser)

if args_cli.video:
    args_cli.enable_cameras = True

# Hydra chỉ nhận phần arguments còn lại
sys.argv = [sys.argv[0]] + remaining_args


@hydra_task_config(
    args_cli.task,
    "fast_sac_cfg_entry_point",
)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: FastSacRunnerCfg,
):
    """Train FastSAC."""

    with launch_simulation(env_cfg, args_cli):

        # CLI overrides
        agent_cfg = cli_args.update_fast_sac_cfg(
            agent_cfg,
            args_cli,
        )

        if args_cli.num_envs is not None:
            env_cfg.scene.num_envs = args_cli.num_envs

        if args_cli.max_iterations is not None:
            agent_cfg.max_iterations = args_cli.max_iterations

        if args_cli.seed is not None:
            agent_cfg.seed = args_cli.seed

        env_cfg.seed = agent_cfg.seed

        if not args_cli.distributed:
            if args_cli.device is not None:
                env_cfg.sim.device = args_cli.device
                agent_cfg.device = args_cli.device
        else:
            local_rank = int(os.getenv("LOCAL_RANK", "0"))
            global_rank = int(os.getenv("RANK", "0"))

            env_cfg.sim.device = f"cuda:{local_rank}"
            agent_cfg.device = f"cuda:{local_rank}"

            seed = agent_cfg.seed + global_rank
            agent_cfg.seed = seed
            env_cfg.seed = seed

        # Logs
        log_root = os.path.abspath(
            os.path.join(
                "logs",
                "fast_sac",
                agent_cfg.experiment_name,
            )
        )

        log_dir = datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S"
        )

        if agent_cfg.run_name:
            log_dir += f"_{agent_cfg.run_name}"

        log_dir = os.path.join(
            log_root,
            log_dir,
        )

        print(
            f"[INFO] FastSAC logs: {log_dir}"
        )

        env_cfg.log_dir = log_dir

        # Create Isaac Lab env
        env = gym.make(
            args_cli.task,
            cfg=env_cfg,
            render_mode="rgb_array"
            if args_cli.video
            else None,
        )

        # MARL -> single agent if needed
        if isinstance(
            env.unwrapped.cfg,
            DirectMARLEnvCfg,
        ):
            env = multi_agent_to_single_agent(env)

        # Video
        if args_cli.video:
            video_kwargs = {
                "video_folder": os.path.join(
                    log_dir,
                    "videos",
                    "train",
                ),
                "step_trigger":
                    lambda step:
                    step % args_cli.video_interval == 0,
                "video_length":
                    args_cli.video_length,
                "disable_logger": True,
            }

            print_dict(
                video_kwargs,
                nesting=4,
            )

            env = gym.wrappers.RecordVideo(
                env,
                **video_kwargs,
            )

        # FastSAC adapter
        env = FastSacVecEnvWrapper(
            env,
            clip_actions=agent_cfg.clip_actions,
        )

        # FastSAC runner
        runner = FastSacRunner(
            env,
            agent_cfg.to_dict(),
            log_dir=log_dir,
            device=agent_cfg.device,
        )

        # Resume
        if agent_cfg.resume:
            resume_path = get_checkpoint_path(
                log_root,
                agent_cfg.load_run,
                agent_cfg.load_checkpoint,
            )

            print(
                f"[INFO] Loading: {resume_path}"
            )

            runner.load(resume_path)

        # Save configs
        dump_yaml(
            os.path.join(
                log_dir,
                "params",
                "env.yaml",
            ),
            env_cfg,
        )

        dump_yaml(
            os.path.join(
                log_dir,
                "params",
                "agent.yaml",
            ),
            agent_cfg,
        )

        try:
            runner.learn(
                num_learning_iterations=
                    agent_cfg.max_iterations,

                init_at_random_ep_len=True,
            )

        finally:
            runner.close()
            env.close()


if __name__ == "__main__":
    main()
