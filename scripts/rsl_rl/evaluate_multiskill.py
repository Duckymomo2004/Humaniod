"""Evaluate saved G1 policies without changing training rewards, curriculum, or weights.

Uses the checkpoint's saved agent/env configuration. Fixed velocity commands are
evaluation inputs; noise, randomizations, terminations, and curriculum stay active.
Measurements are taken before auto-reset so fallen states are not lost.
"""

import argparse
import hashlib
import importlib.metadata as metadata
import json
import math
from pathlib import Path
import sys
import traceback

import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoints", nargs="+", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--seconds", type=float, default=20.0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--scenarios", nargs="+", default=["training_commands", "standing", "forward_slow", "turn_left", "turn_right", "forward_0p5"])
parser.add_argument("--zero_reference", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
output = Path(args.output).resolve()
output.mkdir(parents=True, exist_ok=True)
original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
launcher = AppLauncher(args)
sys.argv = original_argv


def evaluate():
    import gymnasium as gym
    import numpy as np
    from rsl_rl.runners import OnPolicyRunner
    from tensordict import TensorDict

    from isaaclab.utils.io import dump_yaml
    from isaaclab.utils.seed import configure_seed
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    import g1_datn  # noqa: F401
    from evaluation_io import load_snapshot

    commands = {
        "training_commands": None,
        "standing": (0.0, 0.0, 0.0),
        "forward_slow": (0.1, 0.0, 0.0),
        "turn_left": (0.0, 0.0, 0.1),
        "turn_right": (0.0, 0.0, -0.1),
        "forward_0p5": (0.5, 0.0, 0.0),
    }
    if set(args.scenarios) - commands.keys():
        raise ValueError("Unknown scenario")
    results = []
    manifest = {"versions": {n: metadata.version(n) for n in ["isaacsim", "isaaclab", "torch", "rsl-rl-lib"]},
                "seed": args.seed, "num_envs": args.num_envs, "seconds_per_scenario": args.seconds,
                "actor_mode": "deterministic", "measurement": "post-physics, before auto-reset",
                "changes_from_training": ["num_envs", "log_dir", "fixed command inputs for named scenarios", "debug markers disabled"],
                "checkpoint_sha256": {}}
    runs = [(Path(p).resolve(), False) for p in args.checkpoints]
    if args.zero_reference:
        runs.append((Path(args.checkpoints[-1]).resolve(), True))
    for checkpoint, zero in runs:
        label = "zero_actions" if zero else checkpoint.stem
        saved_env = load_snapshot(checkpoint.parent / "params/env.yaml")
        saved_agent = load_snapshot(checkpoint.parent / "params/agent.yaml")
        cfg = load_cfg_from_registry("G1-MultiSkill-v0", "env_cfg_entry_point")
        # Restore the actual saved training config, including migrated PhysX and PPO fields.
        # Lab's strict updater does not accept replacing an optional None with a
        # scalar. Materialize these runtime-resolved fields from the snapshot first.
        cfg.seed = saved_env["seed"]
        cfg.log_dir = saved_env["log_dir"]
        cfg.scene.terrain.env_spacing = saved_env["scene"]["terrain"]["env_spacing"]
        for name in ["robot", "height_scanner", "sky_light"]:
            getattr(cfg.scene, name).spawn.spawn_path = saved_env["scene"][name]["spawn"]["spawn_path"]
        cfg.from_dict(saved_env)
        cfg.scene.num_envs = args.num_envs
        cfg.seed = args.seed
        cfg.log_dir = str(output / label)
        cfg.commands.base_velocity.debug_vis = False
        cfg.sim.visualizer_cfgs = []
        dump_yaml(str(output / label / "eval_env.yaml"), cfg)
        manifest["checkpoint_sha256"][str(checkpoint)] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        configure_seed(args.seed, False)
        raw = gym.make("G1-MultiSkill-v0", cfg=cfg)
        env = raw.unwrapped
        wrapped = RslRlVecEnvWrapper(raw, clip_actions=saved_agent.get("clip_actions"))
        assert env.scene["robot"].num_joints == wrapped.num_actions == 29
        assert type(cfg.sim.physics).__name__ == "PhysxCfg" and env.device.startswith("cuda")
        if zero:
            policy = None
        else:
            runner = OnPolicyRunner(wrapped, saved_agent, log_dir=None, device=env.device)
            runner.load(str(checkpoint), load_cfg={"actor": True, "critic": True, "optimizer": False, "iteration": True}, map_location=env.device)
            policy = runner.get_inference_policy(device=env.device)
            loaded = torch.load(checkpoint, map_location=env.device, weights_only=True)
            assert all(torch.equal(v, loaded["actor_state_dict"][k]) for k, v in runner.alg.actor.state_dict().items())

        robot = env.scene["robot"]
        command_term = env.command_manager.get_term("base_velocity")
        original_resample = command_term._resample_command
        original_update = command_term._update_command
        target = None

        def resample(ids):
            original_resample(ids)
            if target is not None:
                command_term.vel_command_b[ids] = target

        def update():
            original_update()
            if target is not None:
                command_term.vel_command_b[:] = target

        command_term._resample_command = resample
        command_term._update_command = update
        original_compute = env.termination_manager.compute
        context = None
        step_index = 0
        last_actions = None

        def measured_compute():
            dones = original_compute()
            if context is not None:
                # Read terminal physics state now, before env._reset_idx changes it.
                data = robot.data
                f = context["features"][step_index]
                f[:, :3] = data.root_lin_vel_b.torch
                f[:, 3] = data.root_ang_vel_b.torch[:, 2]
                f[:, 4] = torch.acos((-data.projected_gravity_b.torch[:, 2]).clamp(-1, 1))
                f[:, 5] = data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
                f[:, 6] = torch.linalg.vector_norm(data.root_ang_vel_b.torch[:, :2], dim=1)
                f[:, 7] = last_actions.square().mean(dim=1).sqrt()
                f[:, 8] = data.joint_vel.torch.square().mean(dim=1).sqrt()
                f[:, 9] = (data.joint_pos.torch - data.default_joint_pos.torch).square().mean(dim=1).sqrt()
                context["commands"][step_index] = command_term.command
                context["episode_steps"][step_index] = torch.where(dones, env.episode_length_buf, 0)
                for i, name in enumerate(context["term_names"]):
                    context["terms"][step_index, :, i] = env.termination_manager.get_term(name)
                context["first_failure"] = torch.where((context["first_failure"] < 0) & dones, step_index + 1, context["first_failure"])
            return dones

        env.termination_manager.compute = measured_compute
        for scenario_index, scenario in enumerate(["standing"] if zero else args.scenarios):
            target = None if commands[scenario] is None else torch.tensor(commands[scenario], device=env.device)
            context = None
            # History buffers can be allocated lazily during inference. Reset in
            # the same mode so PyTorch permits their in-place clear operation.
            with torch.inference_mode():
                obs_dict, _ = env.reset(seed=args.seed + scenario_index)
            obs = TensorDict(obs_dict, batch_size=[env.num_envs])
            if policy is not None:
                policy.reset()
            steps = int(round(args.seconds / env.step_dt))
            names = list(env.termination_manager.active_terms)
            context = {"features": torch.zeros((steps, env.num_envs, 10), device=env.device),
                       "commands": torch.zeros((steps, env.num_envs, 3), device=env.device),
                       "episode_steps": torch.zeros((steps, env.num_envs), dtype=torch.long, device=env.device),
                       "terms": torch.zeros((steps, env.num_envs, len(names)), dtype=torch.bool, device=env.device),
                       "term_names": names,
                       "first_failure": torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)}
            with torch.inference_mode():
                for step_index in range(steps):
                    last_actions = torch.zeros((env.num_envs, wrapped.num_actions), device=env.device) if zero else policy(obs)
                    assert torch.isfinite(last_actions).all(), "Non-finite actions"
                    obs, reward, done, _ = wrapped.step(last_actions)
                    assert torch.isfinite(reward).all(), "Non-finite rewards"
                    if policy is not None:
                        policy.reset(done)
            features = context["features"].cpu().numpy()
            cmd = context["commands"].cpu().numpy()
            lengths = context["episode_steps"].cpu().numpy()
            terms = context["terms"].cpu().numpy()
            first = context["first_failure"].cpu().numpy()
            assert np.isfinite(features).all()
            if target is not None:
                assert np.allclose(cmd, target.cpu().numpy(), atol=1e-6), "Command was not held fixed"
            episode_seconds = lengths[lengths > 0] * env.step_dt
            done_mask = lengths > 0
            # Episode-level per-env tracking means, excluding right-censored tails.
            episode_xy, episode_yaw, tracked_seconds = [], [], []
            err_xy = np.linalg.norm(features[:, :, :2] - cmd[:, :, :2], axis=2)
            err_yaw = np.abs(features[:, :, 3] - cmd[:, :, 2])
            for index in range(env.num_envs):
                begin = 0
                for end in np.flatnonzero(done_mask[:, index]):
                    ex, ew = float(err_xy[begin:end+1, index].mean()), float(err_yaw[begin:end+1, index].mean())
                    episode_xy.append(ex)
                    episode_yaw.append(ew)
                    if ex < 0.1 and ew < 0.2:
                        tracked_seconds.append((end - begin + 1) * env.step_dt)
                    begin = end + 1
            ended = int(done_mask.sum())
            summary = {"checkpoint": label, "scenario": scenario, "command": commands[scenario],
                       "num_envs": env.num_envs, "duration_s": args.seconds, "transitions": steps * env.num_envs,
                       "completed_episodes": ended, "censored_episodes": env.num_envs,
                       "episode_mean_s": float(episode_seconds.mean()) if ended else None,
                       "episode_median_s": float(np.median(episode_seconds)) if ended else None,
                       "episode_p95_s": float(np.quantile(episode_seconds, .95)) if ended else None,
                       "episode_max_s": float(episode_seconds.max()) if ended else None,
                       "first_episode_survival_2s": float(np.mean((first < 0) | (first * env.step_dt >= 2))),
                       "first_episode_survival_5s": float(np.mean((first < 0) | (first * env.step_dt >= 5))),
                       "first_episode_survival_horizon": float(np.mean(first < 0)),
                       "velocity_xy_rmse_mps": float(np.sqrt(np.mean(err_xy**2))),
                       "velocity_xy_mae_mps": float(err_xy.mean()),
                       "yaw_rate_rmse_radps": float(np.sqrt(np.mean(err_yaw**2))),
                       "yaw_rate_mae_radps": float(err_yaw.mean()),
                       "mean_vx_mps": float(features[:, :, 0].mean()), "mean_vy_mps": float(features[:, :, 1].mean()),
                       "mean_wz_radps": float(features[:, :, 3].mean()),
                       "tilt_mean_deg": float(np.rad2deg(features[:, :, 4]).mean()),
                       "tilt_p95_deg": float(np.quantile(np.rad2deg(features[:, :, 4]), .95)),
                       "upright_under_20deg_fraction": float(np.mean(features[:, :, 4] < math.radians(20))),
                       "base_height_mean_m": float(features[:, :, 5].mean()),
                       "angular_xy_rms_radps": float(np.sqrt(np.mean(features[:, :, 6]**2))),
                       "action_rms": float(np.sqrt(np.mean(features[:, :, 7]**2))),
                       "joint_velocity_rms_radps": float(np.sqrt(np.mean(features[:, :, 8]**2))),
                       "joint_pose_deviation_rms_rad": float(np.sqrt(np.mean(features[:, :, 9]**2))),
                       "episode_tracking_pass_fraction": float(np.mean((np.array(episode_xy) < .1) & (np.array(episode_yaw) < .2))) if ended else None,
                       "tracking_pass_episode_mean_s": float(np.mean(tracked_seconds)) if tracked_seconds else None,
                       "termination_counts": {name: int(terms[:, :, i].sum()) for i, name in enumerate(names)},
                       "bad_orientation_episode_fraction": float(terms[:, :, names.index("bad_orientation")].sum() / ended) if ended else None}
            directory = output / label
            directory.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(directory / f"{scenario}.npz", features=features, commands=cmd,
                                terminal_episode_steps=lengths, terminations=terms, first_failure_steps=first,
                                feature_names=np.array(["vx", "vy", "vz", "wz", "tilt_rad", "height", "omega_xy", "action_rms", "joint_vel_rms", "joint_pose_error"]),
                                termination_names=np.array(names), dt=env.step_dt)
            (directory / f"{scenario}.json").write_text(json.dumps(summary, indent=2))
            results.append(summary)
            (output / "results.json").write_text(json.dumps(results, indent=2))
            print("[EVALUATION] " + json.dumps(summary), flush=True)
            context = None
        wrapped.close()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[PASS] {len(results)} scenarios saved to {output}", flush=True)


exit_code = 0
try:
    evaluate()
except BaseException:
    exit_code = 1
    traceback.print_exc()
finally:
    launcher.app.close(wait_for_replicator=False, exit_code=exit_code)
