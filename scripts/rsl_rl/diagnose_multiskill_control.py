"""Short paired G1 control diagnostics; never modifies training configs or weights."""

import argparse
import csv
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import sys
import traceback

import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
parser.add_argument("--gains", nargs="+", type=float, default=[1.0, 0.0])
parser.add_argument("--seconds", type=float, default=1.5)
parser.add_argument("--no_autoreset", action="store_true", help="Only for a separate post-terminal trajectory probe; termination criteria are still recorded.")
parser.add_argument("--refresh_reset_cache", action="store_true", help="Diagnostic-only cache invalidation after reset; does not modify physics state or source files.")
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

    from isaaclab.utils.io import dump_yaml
    from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse, yaw_quat
    from isaaclab.utils.seed import configure_seed
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    import g1_datn  # noqa: F401
    from evaluation_io import load_snapshot

    checkpoint = Path(args.checkpoint).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    trained_joints = json.loads((checkpoint.parent / "smoke_verification.json").read_text())["joint_names"]
    reports = []

    def array(value):
        if isinstance(value, wp.array):
            value = wp.to_torch(value)
        return value.detach().cpu().numpy().copy()

    for gain in args.gains:
        label = "gain_" + str(gain).replace(".", "p")
        folder = output / label
        folder.mkdir(parents=True, exist_ok=True)
        saved = load_snapshot(checkpoint.parent / "params/env.yaml")
        agent = load_snapshot(checkpoint.parent / "params/agent.yaml")
        cfg = load_cfg_from_registry("G1-MultiSkill-v0", "env_cfg_entry_point")
        cfg.seed, cfg.log_dir = saved["seed"], saved["log_dir"]
        cfg.scene.terrain.env_spacing = saved["scene"]["terrain"]["env_spacing"]
        for name in ["robot", "height_scanner", "sky_light"]:
            getattr(cfg.scene, name).spawn.spawn_path = saved["scene"][name]["spawn"]["spawn_path"]
        cfg.from_dict(saved)
        cfg.seed = 42
        cfg.scene.num_envs = args.num_envs
        cfg.log_dir = str(folder)
        cfg.commands.base_velocity.debug_vis = False
        cfg.sim.visualizer_cfgs = []
        dump_yaml(str(folder / "diagnostic_env.yaml"), cfg)
        configure_seed(42, False)
        raw = gym.make("G1-MultiSkill-v0", cfg=cfg)
        env = raw.unwrapped
        wrapped = RslRlVecEnvWrapper(raw, clip_actions=agent.get("clip_actions"))
        runner = OnPolicyRunner(wrapped, agent, log_dir=None, device=env.device)
        runner.load(str(checkpoint), load_cfg={"actor": True, "critic": True, "optimizer": False, "iteration": True}, map_location=env.device)
        policy = runner.get_inference_policy(device=env.device)
        robot = env.scene["robot"]
        term = env.action_manager.get_term("JointPositionAction")
        sensor = env.scene["contact_forces"]
        foot_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
        foot_ids, _ = robot.find_bodies(foot_names, preserve_order=True)
        sensor_ids, _ = sensor.find_bodies(foot_names, preserve_order=True)
        assert robot.joint_names == term._joint_names == trained_joints
        assert wrapped.num_actions == robot.num_joints == 29
        assert type(cfg.sim.physics).__name__ == "PhysxCfg"
        assert type(policy.obs_normalizer).__name__ == "Identity"
        default_q = robot.data.default_joint_pos.torch.clone()
        assert torch.allclose(term._offset, default_q)
        assert term._scale == .25 and cfg.actions.JointPositionAction.clip is None
        # Validate every actor output independently, without stepping physics.
        mapping_error = 0.0
        for j in range(29):
            probe = torch.zeros((env.num_envs, 29), device=env.device)
            probe[:, j] = 1
            term.process_actions(probe)
            mapping_error = max(mapping_error, float((term.processed_actions - (default_q + .25 * probe)).abs().max()))
        assert mapping_error < 1e-6
        term.process_actions(torch.zeros_like(default_q))
        static = {
            "joint_names": robot.joint_names, "body_names": robot.body_names,
            "sdk_joint_names": list(cfg.scene.robot.joint_sdk_names),
            "default_joint_pos": array(default_q).tolist(),
            "action_offset": array(term._offset).tolist(), "action_scale": term._scale,
            "stiffness": array(robot.data.joint_stiffness.torch).tolist(),
            "damping": array(robot.data.joint_damping.torch).tolist(),
            "effort_limits": array(robot.data.joint_effort_limits.torch).tolist(),
            "velocity_limits": array(robot.data.joint_vel_limits.torch).tolist(),
            "position_limits": array(robot.data.joint_pos_limits.torch).tolist(),
            "engine_stiffness": array(robot.root_view.get_dof_stiffnesses()).tolist(),
            "engine_damping": array(robot.root_view.get_dof_dampings()).tolist(),
            "engine_effort_limits": array(robot.root_view.get_dof_max_forces()).tolist(),
            "actuators": {k: {"class": type(a).__name__, "implicit": a.is_implicit_model,
                              "joint_indices": str(a.joint_indices)} for k, a in robot.actuators.items()},
            "mapping_probe_max_error": mapping_error,
            "obs_terms": env.observation_manager.active_terms,
            "obs_dimensions": env.observation_manager.group_obs_term_dim,
            "normalizer": type(policy.obs_normalizer).__name__,
            "actor_training_flag": policy.training, "actor_obs_groups": policy.obs_groups,
            "history_length": cfg.observations.policy.history_length,
            "noise_enabled": cfg.observations.policy.enable_corruption,
            "clip_actions": wrapped.clip_actions,
            "torque_note": "applied_torque is the implicit actuator PD estimate clipped to motor limits; projected_joint_torque is PhysX incoming-joint force projected on the joint axis, not a pure motor torque sensor",
        }
        assert np.allclose(static["stiffness"], static["engine_stiffness"])
        assert np.allclose(static["damping"], static["engine_damping"])
        assert np.allclose(static["effort_limits"], static["engine_effort_limits"])
        (folder / "control_audit.json").write_text(json.dumps(static, indent=2))
        with (folder / "joint_mapping.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["actor_output", "sim_joint", "sdk_index", "default_rad", "offset_rad", "kp", "kd", "torque_limit_Nm", "velocity_limit_radps"])
            for j, name in enumerate(robot.joint_names):
                writer.writerow([j, name, static["sdk_joint_names"].index(name), static["default_joint_pos"][0][j], static["action_offset"][0][j], static["stiffness"][0][j], static["damping"][0][j], static["effort_limits"][0][j], static["velocity_limits"][0][j]])

        command = env.command_manager.get_term("base_velocity")
        original_resample, original_update = command._resample_command, command._update_command

        def resample(ids):
            original_resample(ids)
            command.vel_command_b[ids] = 0

        def update():
            original_update()
            command.vel_command_b[:] = 0

        command._resample_command, command._update_command = resample, update
        original_compute = env.termination_manager.compute
        collecting = False
        trajectory = None
        current_action = None
        current_policy_action = None
        step_index = 0
        target_error = 0.0
        projected_force_available = True

        def capture(time_s, dones):
            nonlocal projected_force_available
            d = robot.data
            masses = d.body_mass.torch
            com = (d.body_com_pos_w.torch * masses.unsqueeze(-1)).sum(dim=1) / masses.sum(dim=1, keepdim=True)
            rpy = torch.stack(euler_xyz_from_quat(d.root_quat_w.torch), dim=1)
            record = {
                "time": np.array(time_s), "root_pose": array(d.root_link_pose_w.torch),
                "root_velocity": array(d.root_link_vel_w.torch), "roll_pitch_yaw": array(rpy),
                "tilt": array(torch.acos((-d.projected_gravity_b.torch[:, 2]).clamp(-1, 1))),
                "base_velocity_body": array(d.root_lin_vel_b.torch),
                "base_velocity_yaw": array(quat_apply_inverse(yaw_quat(d.root_quat_w.torch), d.root_lin_vel_w.torch)),
                "com_world": array(com), "foot_positions_world": array(d.body_pos_w.torch[:, foot_ids]),
                "foot_forces_world": array(sensor.data.net_forces_w.torch[:, sensor_ids]),
                "joint_pos": array(d.joint_pos.torch), "joint_vel": array(d.joint_vel.torch),
                "joint_target": array(d.joint_pos_target.torch),
                "torque_estimate": array(d.applied_torque.torch), "torque_unclipped": array(d.computed_torque.torch),
                "action": array(current_action), "policy_action": array(current_policy_action),
                "done": array(dones), "bad_orientation": array(env.termination_manager.get_term("bad_orientation")),
            }
            try:
                record["projected_joint_torque"] = array(robot.root_view.get_dof_projected_joint_forces())
            except Exception:
                projected_force_available = False
                record["projected_joint_torque"] = np.full((env.num_envs, 29), np.nan)
            for key, value in record.items():
                trajectory.setdefault(key, []).append(value)

        def compute():
            nonlocal target_error
            dones = original_compute()
            if collecting:
                expected = default_q + .25 * current_action
                target_error = max(target_error, float((robot.data.joint_pos_target.torch - expected).abs().max()))
                engine_targets = wp.to_torch(robot.root_view.get_dof_position_targets())
                target_error = max(target_error, float((engine_targets - expected).abs().max()))
                capture((step_index + 1) * env.step_dt, dones)
            return torch.zeros_like(dones) if args.no_autoreset and collecting else dones

        env.termination_manager.compute = compute
        original_reset = env._reset_idx
        reset_audits = []
        returned_obs_audits = []
        cache_fields = ["_root_link_vel_w", "_root_link_lin_vel_b", "_root_link_ang_vel_b",
                        "_root_com_lin_vel_b", "_root_com_ang_vel_b", "_root_com_pose_w",
                        "_projected_gravity_b", "_heading_w"]

        def inspect_reset(ids):
            original_reset(ids)
            d = robot.data
            engine_pose = wp.to_torch(robot.root_view.get_root_transforms()).clone()
            engine_velocity = wp.to_torch(robot.root_view.get_root_velocities()).clone()
            gravity = torch.zeros((env.num_envs, 3), device=env.device)
            gravity[:, 2] = -1
            expected_gravity = quat_apply_inverse(engine_pose[:, 3:], gravity)
            expected_angular = quat_apply_inverse(engine_pose[:, 3:], engine_velocity[:, 3:])
            cached_gravity = d.projected_gravity_b.torch.clone()
            cached_angular = d.root_ang_vel_b.torch.clone()
            audit = None
            if len(reset_audits) < 12:
                audit = {"during_rollout": collecting, "ids": array(ids).tolist(),
                         "engine_velocity": array(engine_velocity[ids]).tolist(),
                         "cached_gravity": array(cached_gravity[ids]).tolist(),
                         "expected_gravity": array(expected_gravity[ids]).tolist(),
                         "cached_angular": array(cached_angular[ids]).tolist(),
                         "expected_angular": array(expected_angular[ids]).tolist(),
                         "gravity_cache_error": float((cached_gravity[ids] - expected_gravity[ids]).abs().max()),
                         "angular_cache_error": float((cached_angular[ids] - expected_angular[ids]).abs().max())}
                reset_audits.append(audit)
            if args.refresh_reset_cache:
                for name in cache_fields:
                    getattr(d, name).timestamp = -1.0
                if audit is not None:
                    audit["refreshed_gravity_error"] = float((d.projected_gravity_b.torch[ids] - expected_gravity[ids]).abs().max())
                    audit["refreshed_angular_error"] = float((d.root_ang_vel_b.torch[ids] - expected_angular[ids]).abs().max())
                assert torch.equal(engine_velocity, wp.to_torch(robot.root_view.get_root_velocities()))

        env._reset_idx = inspect_reset
        for seed in args.seeds:
            collecting = False
            with torch.inference_mode():
                obs_dict, _ = env.reset(seed=seed)
                obs = TensorDict(obs_dict, batch_size=[env.num_envs])
                policy.reset()
                env.action_manager.process_action(torch.zeros_like(default_q))
                env.action_manager.apply_action()
                env.scene.write_data_to_sim()
                initial = {"root_pose": array(robot.data.root_link_pose_w.torch),
                           "root_velocity": array(robot.data.root_link_vel_w.torch),
                           "joint_pos": array(robot.data.joint_pos.torch),
                           "joint_vel": array(robot.data.joint_vel.torch),
                           "body_mass": array(robot.data.body_mass.torch),
                           "materials": array(robot.root_view.get_material_properties()),
                           "obs_policy": array(obs["policy"]), "obs_critic": array(obs["critic"])}
                initial["engine_root_velocity"] = array(robot.root_view.get_root_velocities())
                initial["cached_gravity"] = array(robot.data.projected_gravity_b.torch)
                # Check history concatenation and ordering against the actual buffers.
                start = 0
                for name, dim in zip(env.observation_manager.active_terms["policy"], env.observation_manager.group_obs_term_dim["policy"]):
                    buffer = env.observation_manager._group_obs_term_history_buffer["policy"][name].buffer
                    block = buffer.reshape(env.num_envs, -1)
                    assert torch.equal(block, obs["policy"][:, start:start + block.shape[1]]), name
                    start += block.shape[1]
                assert start == 480 and obs["critic"].shape[1] == 495
                rng = torch.cuda.get_rng_state().clone()
                first_action = policy(obs)
                assert torch.equal(first_action, policy(obs, stochastic_output=False))
                assert torch.equal(first_action, policy(obs))
                assert torch.equal(rng, torch.cuda.get_rng_state())
                np.savez_compressed(folder / f"initial_seed_{seed}.npz", **initial)
                trajectory = {}
                current_action = torch.zeros_like(first_action)
                current_policy_action = first_action
                capture(0.0, torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
                collecting = True
                for step_index in range(round(args.seconds / env.step_dt)):
                    current_policy_action = policy(obs)
                    current_action = current_policy_action * gain
                    assert torch.isfinite(current_action).all()
                    obs, reward, done, extras = wrapped.step(current_action)
                    if done.any() and not args.no_autoreset and len(returned_obs_audits) < 12:
                        ids = done.nonzero(as_tuple=False).flatten()
                        native_pose = wp.to_torch(robot.root_view.get_root_transforms())
                        native_vel = wp.to_torch(robot.root_view.get_root_velocities())
                        gravity = torch.zeros((env.num_envs, 3), device=env.device)
                        gravity[:, 2] = -1
                        expected_g = quat_apply_inverse(native_pose[:, 3:], gravity)
                        expected_w = quat_apply_inverse(native_pose[:, 3:], native_vel[:, 3:])
                        # Critic has no observation noise: term-major, five history samples.
                        critic_w = obs['critic'][ids, 15:30].reshape(-1, 5, 3)
                        critic_g = obs['critic'][ids, 30:45].reshape(-1, 5, 3)
                        returned_obs_audits.append({
                            'seed': seed, 'time_s': (step_index + 1) * env.step_dt,
                            'ids': array(ids).tolist(),
                            'native_velocity_max_abs': float(native_vel[ids].abs().max()),
                            'critic_angular_history_error_radps': float((critic_w / .2 - expected_w[ids, None]).abs().max()),
                            'critic_gravity_history_error': float((critic_g - expected_g[ids, None]).abs().max()),
                            'policy_angular_history_vs_native_scaled_error': float((obs['policy'][ids, :15].reshape(-1, 5, 3) - .2 * expected_w[ids, None]).abs().max()),
                            'policy_gravity_history_vs_native_error': float((obs['policy'][ids, 15:30].reshape(-1, 5, 3) - expected_g[ids, None]).abs().max()),
                        })
                    assert torch.all(command.command == 0)
                    policy.reset(done)
                collecting = False
            data = {key: np.stack(value) for key, value in trajectory.items()}
            assert target_error < 1e-6, target_error
            np.savez_compressed(folder / f"trajectory_seed_{seed}.npz", **data)
            first_failure = np.array([data["time"][np.flatnonzero(data["done"][:, i])[0]] if data["done"][:, i].any() else np.nan for i in range(env.num_envs)])
            reports.append({"gain": gain, "seed": seed, "first_failure_s": first_failure.tolist(),
                            "first_failure_mean_s": float(np.nanmean(first_failure)) if np.isfinite(first_failure).any() else None,
                            "censored_first_episodes": int(np.isnan(first_failure).sum()), "seconds": args.seconds,
                            "no_autoreset": args.no_autoreset, "deterministic_inference_verified": True,
                            "target_formula_max_error": target_error, "projected_force_available": projected_force_available})
            (output / "results.json").write_text(json.dumps(reports, indent=2))
            print("[DIAGNOSTIC] " + json.dumps(reports[-1]), flush=True)
        (folder / "reset_cache_audit.json").write_text(json.dumps(reset_audits, indent=2))
        (folder / "returned_observation_audit.json").write_text(json.dumps(returned_obs_audits, indent=2))
        wrapped.close()
    # Paired state/parameter/observation checks are required before causal comparison.
    baseline = output / ("gain_" + str(args.gains[0]).replace(".", "p"))
    pairing = []
    for gain in args.gains[1:]:
        folder = output / ("gain_" + str(gain).replace(".", "p"))
        for seed in args.seeds:
            a = np.load(baseline / f"initial_seed_{seed}.npz")
            b = np.load(folder / f"initial_seed_{seed}.npz")
            errors = {k: float(np.max(np.abs(a[k] - b[k]))) for k in a.files}
            assert all(e < 1e-5 for e in errors.values()), errors
            pairing.append({"gain": gain, "seed": seed, "max_errors": errors})
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == source_hash
    manifest = {"checkpoint": str(checkpoint), "checkpoint_sha256": source_hash,
                "versions": {n: metadata.version(n) for n in ["isaacsim", "isaaclab", "torch", "rsl-rl-lib"]},
                "gains": args.gains, "seeds": args.seeds, "num_envs": args.num_envs,
                "seconds": args.seconds, "no_autoreset": args.no_autoreset, "pairing": pairing,
                "diagnostic_refresh_reset_cache": args.refresh_reset_cache,
                "training_config_changed": False, "new_training_iterations": 0}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("[PASS] Paired control diagnostic complete", flush=True)


exit_code = 0
try:
    main()
except BaseException:
    exit_code = 1
    traceback.print_exc()
finally:
    launcher.app.close(wait_for_replicator=False, exit_code=exit_code)
