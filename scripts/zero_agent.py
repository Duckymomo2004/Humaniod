# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

import argparse
import contextlib
import sys

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401
from isaaclab_tasks.utils import (
    add_launcher_args,
    launch_simulation,
    resolve_task_config,
    setup_preset_cli,
)

# add argparse arguments
parser = argparse.ArgumentParser(description="Zero agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--max_steps", type=int, default=100, help="Stop after this many steps; 0 runs until closed.")
# append AppLauncher cli args
add_launcher_args(parser)
# simple agents should open Kit visualizer by default
parser.set_defaults(visualizer=["kit"])
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args

import g1_datn.tasks  # noqa: F401
MAX_STEPS = 100


def main():
    """Zero actions agent with Isaac Lab environment."""

    torch.manual_seed(42)

    # parse configuration via Hydra (supports preset selection, e.g. env.sim.physics=newton_mjwarp)
    env_cfg, _ = resolve_task_config(args_cli.task, "")

    with launch_simulation(env_cfg, args_cli):
        # override with CLI arguments
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        if args_cli.disable_fabric:
            env_cfg.sim.use_fabric = False

        # create environment
        env = gym.make(args_cli.task, cfg=env_cfg)

        # print info (this is vectorized environment)
        print(f"[INFO]: Gym observation space: {env.observation_space}")
        print(f"[INFO]: Gym action space: {env.action_space}")
        # reset environment
        env.reset()
        # simulate environment
        # keep running while any visualizer is open, otherwise fall back to MAX_STEPS
        sim = env.unwrapped.sim
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        step = 0
        while args_cli.max_steps == 0 or step < args_cli.max_steps:
            if sim.visualizers:
                # visualizer mode: run until the visualizer window is closed
                if not any(v.is_running() and not v.is_closed for v in sim.visualizers):
                    break
            # run everything in inference mode
            with torch.inference_mode():
                # apply actions
                obs, reward, terminated, truncated, info = env.step(actions)
                assert torch.isfinite(reward).all(), "Non-finite simulation rewards"
                step += 1
        print(f"[SUCCESS] Simulated {step} steps on {env.unwrapped.device} with {env.unwrapped.num_envs} G1 robots.", flush=True)
        # close the simulator
        env.close()


if __name__ == "__main__":
    # run the main function
    main()
