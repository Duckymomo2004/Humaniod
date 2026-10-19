"""Native PhysX / returned-observation regression; inference only, no PPO update."""

import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import sys
import traceback
from types import MethodType

import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--num_envs", type=int, required=True)
parser.add_argument("--seconds", type=float, default=4.0)
parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
parser.add_argument("--baseline", action="store_true", help="Test-only bypass of project reset fix; expected to reproduce violations.")
parser.add_argument("--nonzero_reset_probe", action="store_true", help="Inject nonzero reset state through a test-only event hook and audit returned observations.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
sys.argv = [sys.argv[0]]
launcher = AppLauncher(args)


def main():
    import gymnasium as gym
    import numpy as np
    import warp as wp
    from rsl_rl.runners import OnPolicyRunner
    from tensordict import TensorDict
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.utils.math import quat_apply_inverse, quat_from_euler_xyz
    from isaaclab.utils.seed import configure_seed
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    import g1_datn
    from g1_datn.tasks.manager_based.g1_multiskill.g1_multiskill_env import G1MultiSkillEnv
    from evaluation_io import load_snapshot

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    checkpoint = Path(args.checkpoint)
    sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    saved = load_snapshot(checkpoint.parent / "params/env.yaml")
    agent = load_snapshot(checkpoint.parent / "params/agent.yaml")
    cfg = load_cfg_from_registry("G1-MultiSkill-v0", "env_cfg_entry_point")
    cfg.seed, cfg.log_dir = saved["seed"], saved["log_dir"]
    cfg.scene.terrain.env_spacing = saved["scene"]["terrain"]["env_spacing"]
    for name in ["robot", "height_scanner", "sky_light"]:
        getattr(cfg.scene, name).spawn.spawn_path = saved["scene"][name]["spawn"]["spawn_path"]
    cfg.from_dict(saved)
    cfg.seed, cfg.scene.num_envs, cfg.log_dir = 42, args.num_envs, str(out)
    cfg.commands.base_velocity.debug_vis = False
    cfg.sim.visualizer_cfgs = []
    configure_seed(42, False)
    raw = gym.make("G1-MultiSkill-v0", cfg=cfg)
    env = raw.unwrapped
    assert isinstance(env, G1MultiSkillEnv)
    if args.baseline:
        env._reset_idx = MethodType(ManagerBasedRLEnv._reset_idx, env)
    wrapped = RslRlVecEnvWrapper(raw, clip_actions=agent.get("clip_actions"))
    runner = OnPolicyRunner(wrapped, agent, log_dir=None, device=env.device)
    runner.load(str(checkpoint), load_cfg={"actor": True, "critic": True, "optimizer": False, "iteration": True}, map_location=env.device)
    actor = runner.get_inference_policy(device=env.device)
    critic = runner.alg.critic
    assert type(actor.obs_normalizer).__name__ == type(critic.obs_normalizer).__name__ == "Identity"
    robot = env.scene["robot"]
    command = env.command_manager.get_term("base_velocity")
    resample0, update0 = command._resample_command, command._update_command

    def resample(ids):
        resample0(ids)
        command.vel_command_b[ids] = 0

    def update():
        update0()
        command.vel_command_b[:] = 0

    command._resample_command, command._update_command = resample, update

    def tensor(x):
        return wp.to_torch(x) if isinstance(x, wp.array) else x

    def maximum(x):
        return float(x.abs().max()) if x.numel() else 0.0

    def native():
        pose = tensor(robot.root_view.get_root_transforms()).clone()
        vel = tensor(robot.root_view.get_root_velocities()).clone()
        gravity = torch.zeros_like(vel[:, :3]); gravity[:, 2] = -1
        return pose, vel, quat_apply_inverse(pose[:, 3:], vel[:, 3:]), quat_apply_inverse(pose[:, 3:], gravity)

    def blocks(obs, group):
        start, result = 0, {}
        for name, dims in zip(env.observation_manager.active_terms[group], env.observation_manager.group_obs_term_dim[group]):
            size = int(np.prod(dims))
            result[name] = obs[group][:, start:start+size].reshape(env.num_envs, 5, -1)
            start += size
        return result

    records, reset_checks, consumption = [], [], {"actor": 0, "critic": 0, "post_reset_actor": 0, "post_reset_critic": 0}
    expected_input = {}
    post_reset_pending = False

    def input_hook(group):
        def hook(module, inputs):
            assert torch.equal(inputs[0], expected_input[group]), f"{group} MLP did not consume returned observations"
            consumption[group] += 1
            if post_reset_pending:
                consumption["post_reset_" + group] += 1
        return hook

    actor.mlp.register_forward_pre_hook(input_hook("actor"))
    critic.mlp.register_forward_pre_hook(input_hook("critic"))
    reset0 = env._reset_idx

    def reset_probe(ids):
        before_pose, before_vel, before_w, before_g = native()
        reset0(ids)
        after_pose, after_vel, after_w, after_g = native()
        mask = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        mask[ids.long()] = False
        check = {"count": len(ids), "non_reset_count": int(mask.sum()),
                 "non_reset_native_pose_change": maximum(after_pose[mask] - before_pose[mask]),
                 "non_reset_native_velocity_change": maximum(after_vel[mask] - before_vel[mask]),
                 "non_reset_cached_angular_error": maximum(robot.data.root_ang_vel_b.torch[mask] - after_w[mask]),
                 "non_reset_cached_gravity_error": maximum(robot.data.projected_gravity_b.torch[mask] - after_g[mask])}
        assert check["non_reset_native_pose_change"] == check["non_reset_native_velocity_change"] == 0
        if not args.baseline:
            assert check["non_reset_cached_angular_error"] < 1e-5 and check["non_reset_cached_gravity_error"] < 1e-5
        reset_checks.append(check)

    env._reset_idx = reset_probe

    def audit(obs, ids, seed, step, require_zero_velocity=True):
        pose, vel, w, g = native()
        q = tensor(robot.root_view.get_dof_positions()).clone()
        qd = tensor(robot.root_view.get_dof_velocities()).clone()
        expected = {"base_lin_vel": quat_apply_inverse(pose[:, 3:], vel[:, :3]), "base_ang_vel": .2 * w,
                    "projected_gravity": g, "velocity_commands": command.command,
                    "joint_pos_rel": q - robot.data.default_joint_pos.torch,
                    "joint_vel_rel": .05 * (qd - robot.data.default_joint_vel.torch),
                    "last_action": env.action_manager.action}
        if require_zero_velocity:
            assert maximum(vel[ids]) < 1e-6, "Default reset expected native velocity zero"
        errors, violations = {}, []
        for group in ["policy", "critic"]:
            values = blocks(obs, group)
            for name, block in values.items():
                error = maximum(block[ids] - expected[name][ids, None])
                bounds = {"base_ang_vel": .04, "projected_gravity": .05, "joint_pos_rel": .01, "joint_vel_rel": .075}
                bound = bounds.get(name, 0) if group == "policy" else 0
                errors[group + "/" + name] = error
                if error > bound + 1e-5:
                    violations.append(group + "/" + name)
                # All five slots must hold the new episode's first observation.
                assert maximum(block[ids] - block[ids, :1]) == 0, f"{group}/{name} history leaked previous episode"
        row = {"seed": seed, "step": step, "reset_count": len(ids), "errors": errors, "violations": violations,
               "native_velocity_max": maximum(vel[ids])}
        records.append(row)
        if not args.baseline:
            assert not violations, row

    saved_post_reset = None
    counts = torch.zeros(env.num_envs, dtype=torch.int64, device=env.device)
    with torch.inference_mode():
        for seed in args.seeds:
            obs0, _ = env.reset(seed=seed)
            obs = TensorDict(obs0, batch_size=[env.num_envs])
            audit(obs, torch.arange(env.num_envs, device=env.device), seed, -1)
            post_reset_pending = True
            for step in range(round(args.seconds / env.step_dt)):
                expected_input["actor"], expected_input["critic"] = obs["policy"].clone(), obs["critic"].clone()
                action = actor(obs)
                value = critic(obs)
                assert torch.isfinite(action).all() and torch.isfinite(value).all()
                old_blocks = {g: {k: v.clone() for k, v in blocks(obs, g).items()} for g in ["policy", "critic"]}
                obs, _, done, _ = wrapped.step(action)
                ids = done.nonzero(as_tuple=False).flatten()
                # Non-reset history should advance exactly one sample, not be cleared.
                non_ids = (~done.bool()).nonzero(as_tuple=False).flatten()
                for group in ["policy", "critic"]:
                    for name, block in blocks(obs, group).items():
                        assert torch.equal(block[non_ids, :-1], old_blocks[group][name][non_ids, 1:]), f"non-reset history changed: {group}/{name}"
                if len(ids):
                    audit(obs, ids, seed, step)
                    counts[ids] += 1
                    saved_post_reset = obs.clone()
                post_reset_pending = bool(len(ids))
                actor.reset(done); critic.reset(done)
        assert (counts >= 4).all(), counts
        assert saved_post_reset is not None
        # The same PPO collection API used by train.py consumes this returned
        # TensorDict. No environment step, rollout storage update or optimizer.
        expected_input["actor"] = saved_post_reset["policy"].clone()
        expected_input["critic"] = saved_post_reset["critic"].clone()
        post_reset_pending = True
        runner.alg.act(saved_post_reset)

        setter_checks = []
        if not args.baseline:
            # Nonzero pose/velocity prove that the fix reads state, not zeros.
            # Exercise project invalidator after index/mask writes; it is not a
            # general override of the upstream setters outside reset.
            ids = torch.arange(0, env.num_envs, 2, device=env.device, dtype=torch.int32)
            for mode in ["index_pose", "index_velocity", "mask_pose", "mask_velocity"]:
                old_pose, old_vel, _, _ = native()
                # Populate derived buffers at the exact timestamp before write.
                _ = robot.data.projected_gravity_b.torch.clone(), robot.data.root_ang_vel_b.torch.clone()
                pose, vel = old_pose.clone(), old_vel.clone()
                if mode.endswith("pose"):
                    v = torch.full((len(ids),), .17 if mode.startswith("index") else -.23, device=env.device)
                    pose[ids.long(), 3:] = quat_from_euler_xyz(v, v * .5, v * 2)
                else:
                    vel[ids.long()] = torch.tensor([.2, -.1, .3, .7, -.4, .2], device=env.device)
                if mode.startswith("index"):
                    if mode.endswith("pose"):
                        robot.write_root_link_pose_to_sim_index(root_pose=pose[ids.long()], env_ids=ids)
                    else:
                        robot.write_root_com_velocity_to_sim_index(root_velocity=vel[ids.long()], env_ids=ids)
                else:
                    mask = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool); mask[ids.long()] = True
                    mask_wp = wp.from_torch(mask)
                    if mode.endswith("pose"):
                        robot.write_root_link_pose_to_sim_mask(root_pose=pose, env_mask=mask_wp)
                    else:
                        robot.write_root_com_velocity_to_sim_mask(root_velocity=vel, env_mask=mask_wp)
                native_before = native()
                env._invalidate_reset_root_cache()
                native_after = native()
                assert torch.equal(native_before[0], native_after[0]) and torch.equal(native_before[1], native_after[1])
                e_w = maximum(robot.data.root_ang_vel_b.torch - native_after[2])
                e_g = maximum(robot.data.projected_gravity_b.torch - native_after[3])
                assert e_w < 1e-5 and e_g < 1e-5
                setter_checks.append({"mode": mode, "angular_error": e_w, "gravity_error": e_g,
                                      "native_angular_max": maximum(native_after[2]), "native_gravity_xy_max": maximum(native_after[3][:, :2])})
            # Actual subset manual reset must also clear history only for ids.
            old_obs = env.obs_buf
            non_ids = torch.tensor([i for i in range(env.num_envs) if i % 2], device=env.device, dtype=torch.long)
            obs0, _ = env.reset(seed=47, env_ids=ids)
            obs = TensorDict(obs0, batch_size=[env.num_envs])
            audit(obs, ids.long(), 47, -2)
            if len(non_ids):
                for group in ["policy", "critic"]:
                    for name, block in blocks(obs, group).items():
                        assert torch.equal(block[non_ids, :-1], blocks(old_obs, group)[name][non_ids, 1:])
            if args.nonzero_reset_probe:
                apply0 = env.event_manager.apply

                def inject_reset(*positional, **kwargs):
                    apply0(*positional, **kwargs)
                    mode = kwargs.get("mode", positional[0] if positional else None)
                    if mode == "reset":
                        selected = kwargs["env_ids"]
                        pose, vel, _, _ = native()
                        angle = torch.full((len(selected),), .17, device=env.device)
                        pose[selected.long(), 3:] = quat_from_euler_xyz(angle, angle * .5, angle * 2)
                        vel[selected.long()] = torch.tensor([.2, -.1, .3, .7, -.4, .2], device=env.device)
                        robot.write_root_link_pose_to_sim_index(root_pose=pose[selected.long()], env_ids=selected)
                        robot.write_root_com_velocity_to_sim_index(root_velocity=vel[selected.long()], env_ids=selected)

                env.event_manager.apply = inject_reset
                try:
                    obs0, _ = env.reset(seed=48, env_ids=ids)
                    obs = TensorDict(obs0, batch_size=[env.num_envs])
                    audit(obs, ids.long(), 48, -3, require_zero_velocity=False)
                    assert maximum(native()[1][ids.long()]) > .6
                    expected_input["actor"], expected_input["critic"] = obs["policy"].clone(), obs["critic"].clone()
                    runner.alg.act(obs)
                finally:
                    env.event_manager.apply = apply0

    violations = sum(bool(r["violations"]) for r in records)
    assert violations > 0 if args.baseline else violations == 0
    assert consumption["post_reset_actor"] > 0 and consumption["post_reset_critic"] > 0
    result = {"passed": True, "baseline_expected_failure_reproduced": args.baseline,
              "env_class": type(env).__name__, "entry_point": gym.spec("G1-MultiSkill-v0").entry_point,
              "num_envs": env.num_envs, "seeds": args.seeds, "seconds_per_seed": args.seconds,
              "versions": {n: metadata.version(n) for n in ["isaacsim", "isaaclab", "torch", "rsl-rl-lib"]},
              "auto_resets_per_env": counts.cpu().tolist(), "violating_observation_events": violations,
              "records": records, "reset_checks": reset_checks, "network_input_checks": consumption,
              "ppo_collection_api_probe": True, "setter_checks": setter_checks, "training_iterations": 0,
              "nonzero_reset_observation_verified": args.nonzero_reset_probe and not args.baseline,
              "checkpoint_sha256": sha}
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == sha
    (out / "result.json").write_text(json.dumps(result, indent=2))
    wrapped.close()
    print(f"[PASS] num_envs={args.num_envs} baseline={args.baseline} auto_resets={int(counts.sum())} violations={violations}", flush=True)


exit_code = 0
try:
    main()
except BaseException:
    exit_code = 1
    traceback.print_exc()
finally:
    launcher.app.close(wait_for_replicator=False, exit_code=exit_code)
