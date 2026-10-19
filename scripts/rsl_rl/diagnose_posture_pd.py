"""G1 posture/settling ablations based on the existing paired control probes.

Inference only. Each case creates its own scene from the saved configuration;
production configs, framework and checkpoint weights are never modified.
"""

import argparse
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
parser.add_argument("--cases", nargs="+", default=["zero"])
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
parser.add_argument("--seconds", type=float, default=5.)
parser.add_argument("--stop_after_first_failures", action="store_true", help="Stop once every robot has terminated its first episode; later survival is already zero.")
parser.add_argument("--reference", help="Reference zero-arm directory for matching nuisance state in a standalone plane test.")
parser.add_argument("--inspect_geometry", action="store_true", help="Export native link/COM poses and a conservative visual-envelope height at reset.")
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
    from isaaclab.utils.math import euler_xyz_from_quat
    from isaaclab.utils.seed import configure_seed
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    import g1_datn
    from evaluation_io import load_snapshot
    from posture_geometry import collision_vertices, support_metrics, transform_vertices

    allowed = {"zero", "direct_default", "no_push", "nominal_mass", "fixed_friction", "fixed_pose", "nominal_mass_friction", "controlled", "plane", "contact_start", "ankle_kp2", "ankle_kp4", "ankle_kp8", "ankle_kd2", "ankle_bias_p10", "policy1", "policy05", "policy025"}
    if set(args.cases) - allowed:
        raise ValueError(set(args.cases) - allowed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    checkpoint = Path(args.checkpoint)
    sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    reports = []

    def tensor(x):
        return wp.to_torch(x) if isinstance(x, wp.array) else x

    def array(x):
        return tensor(x).detach().cpu().numpy().copy()

    for case in args.cases:
        folder = out / case
        folder.mkdir()
        saved = load_snapshot(checkpoint.parent / "params/env.yaml")
        agent = load_snapshot(checkpoint.parent / "params/agent.yaml")
        cfg = load_cfg_from_registry("G1-MultiSkill-v0", "env_cfg_entry_point")
        cfg.seed, cfg.log_dir = saved["seed"], saved["log_dir"]
        cfg.scene.terrain.env_spacing = saved["scene"]["terrain"]["env_spacing"]
        for name in ["robot", "height_scanner", "sky_light"]:
            getattr(cfg.scene, name).spawn.spawn_path = saved["scene"][name]["spawn"]["spawn_path"]
        cfg.from_dict(saved)
        cfg.seed, cfg.scene.num_envs, cfg.log_dir = 42, args.num_envs, str(folder)
        cfg.commands.base_velocity.debug_vis = False
        cfg.sim.visualizer_cfgs = []
        # Same passive instrumentation in every arm. Net contact reporting and
        # the robot's physics settings stay unchanged.
        cfg.scene.contact_forces.filter_prim_paths_expr = ["/World/ground/terrain/mesh"]
        cfg.scene.contact_forces.track_contact_points = True
        cfg.scene.contact_forces.max_contact_data_count_per_prim = 64
        if case in {"fixed_pose", "controlled"}:
            cfg.events.reset_base.params["pose_range"] = {"x": (0., 0.), "y": (0., 0.), "yaw": (0., 0.)}
        if case == "plane":
            cfg.scene.terrain.terrain_type = "plane"
            cfg.scene.terrain.terrain_generator = None
            cfg.curriculum.terrain_levels = None  # generator-only updater cannot run on a plane
            cfg.scene.contact_forces.filter_prim_paths_expr = ["/World/ground/terrain/CollisionPlane"]
        dump_yaml(str(folder / "diagnostic_env.yaml"), cfg)
        configure_seed(42, False)
        raw = gym.make("G1-MultiSkill-v0", cfg=cfg)
        env = raw.unwrapped
        assert type(env).__name__ == "G1MultiSkillEnv"
        wrapped = RslRlVecEnvWrapper(raw, clip_actions=agent.get("clip_actions"))
        runner = OnPolicyRunner(wrapped, agent, log_dir=None, device=env.device)
        runner.load(str(checkpoint), load_cfg={"actor": True, "critic": True, "optimizer": False, "iteration": True}, map_location=env.device)
        policy = runner.get_inference_policy(device=env.device)
        if case == "no_push":
            # Preserve startup/reset RNG draws and timer resampling exactly.
            # Removing the term before scene construction changes unrelated
            # random masses; only replace its physical effect after startup.
            env.event_manager.get_term_cfg("push_robot").func = lambda *a, **k: None
        robot, sensor = env.scene["robot"], env.scene["contact_forces"]
        term = env.action_manager.get_term("JointPositionAction")
        assert robot.num_joints == wrapped.num_actions == 29
        assert robot.joint_names == term._joint_names
        assert term._scale == .25 and torch.equal(term._offset, robot.data.default_joint_pos.torch)
        feet = ["left_ankle_roll_link", "right_ankle_roll_link"]
        foot_ids, _ = robot.find_bodies(feet, preserve_order=True)
        sensor_ids, _ = sensor.find_bodies(feet, preserve_order=True)
        assert sensor.contact_view.filter_count > 0
        default_q = robot.data.default_joint_pos.torch.clone()
        nominal_mass = None
        nominal_inertia = None
        for event in env.event_manager._mode_class_term_cfgs["startup"]:
            if hasattr(event.func, "default_mass") and event.func.default_mass is not None:
                nominal_mass, nominal_inertia = event.func.default_mass.clone(), event.func.default_inertia.clone()
        if case in {"nominal_mass", "nominal_mass_friction", "controlled"}:
            assert nominal_mass is not None
            robot.set_masses_index(masses=nominal_mass)
            robot.set_inertias_index(inertias=nominal_inertia)
        if case in {"fixed_friction", "nominal_mass_friction", "controlled"}:
            materials = tensor(robot.root_view.get_material_properties()).clone()
            materials[..., 0], materials[..., 1], materials[..., 2] = .8, .6, 0.
            ids_cpu = wp.array(np.arange(env.num_envs, dtype=np.int32), device="cpu")
            robot.root_view.set_material_properties(wp.from_torch(materials), indices=ids_cpu)
        if case == "plane":
            # Terrain construction may consume a different RNG sequence. Match
            # nuisance physical parameters explicitly to the reference arm.
            reference_dir = Path(args.reference) if args.reference else out / "zero"
            reference = json.loads((reference_dir / "static.json").read_text())
            robot.set_masses_index(masses=torch.tensor(reference["mass"], device=env.device))
            robot.set_inertias_index(inertias=torch.tensor(reference["inertias"], device=env.device))
            ids_cpu = wp.array(np.arange(env.num_envs, dtype=np.int32), device="cpu")
            robot.root_view.set_material_properties(wp.array(np.asarray(reference["materials"], dtype=np.float32), device="cpu"), indices=ids_cpu)
        if case.startswith("ankle_k"):
            is_stiffness = "kp" in case
            factor = int(case.split("kp")[1]) if is_stiffness else 2
            joint_ids, _ = robot.find_joints(["left_ankle_pitch_joint", "right_ankle_pitch_joint"], preserve_order=True)
            values = robot.data.joint_stiffness.torch.clone() if is_stiffness else robot.data.joint_damping.torch.clone()
            values[:, joint_ids] *= factor
            if is_stiffness:
                robot.write_joint_stiffness_to_sim_index(stiffness=values)
            else:
                robot.write_joint_damping_to_sim_index(damping=values)
            # Keep the diagnostic estimate consistent with the actual engine gain.
            for actuator in robot.actuators.values():
                indices = list(range(29)) if isinstance(actuator.joint_indices, slice) else actuator.joint_indices.tolist()
                for j in joint_ids:
                    if j in indices:
                        value = actuator.stiffness if is_stiffness else actuator.damping
                        value[:, indices.index(j)] *= factor

        command = env.command_manager.get_term("base_velocity")
        resample0, update0 = command._resample_command, command._update_command

        def resample(ids):
            resample0(ids)
            command.vel_command_b[ids] = 0

        def update():
            update0()
            command.vel_command_b[:] = 0

        command._resample_command, command._update_command = resample, update
        if case == "direct_default":
            # Absolute posture servo, independent of action processing. With
            # zero action the expected PhysX target is identical.
            term.apply_actions = lambda: robot.set_joint_position_target(default_q)

        # Geometry is extracted in the foot link's own frame. Runtime transforms
        # come from PhysX, not potentially stale authored USD poses.
        local_vertices, shapes = [], []
        for body_id in foot_ids:
            v, s = collision_vertices(env.sim.stage, robot.root_view.link_paths[0][body_id])
            local_vertices.append(v); shapes.append(s)
        static = {"case": case, "joint_names": robot.joint_names, "body_names": robot.body_names,
                  "default_joint_pos": array(default_q).tolist(), "action_scale": term._scale,
                  "action_offset": array(term._offset).tolist(),
                  "kp": array(robot.data.joint_stiffness.torch).tolist(), "kd": array(robot.data.joint_damping.torch).tolist(),
                  "engine_kp": array(robot.root_view.get_dof_stiffnesses()).tolist(), "engine_kd": array(robot.root_view.get_dof_dampings()).tolist(),
                  "effort_limits": array(robot.data.joint_effort_limits.torch).tolist(),
                  "joint_limits": array(robot.data.joint_pos_limits.torch).tolist(),
                  "velocity_limits": array(robot.data.joint_vel_limits.torch).tolist(),
                  "mass": array(robot.data.body_mass.torch).tolist(),
                  "nominal_mass": None if nominal_mass is None else array(nominal_mass).tolist(),
                  "materials": array(robot.root_view.get_material_properties()).tolist(),
                  "inertias": array(robot.root_view.get_inertias()).tolist(),
                  "foot_shapes": shapes, "foot_local_vertices": [v.tolist() for v in local_vertices],
                  "terrain_paths": env.scene.terrain.terrain_prim_paths,
                  "physics_dt": env.physics_dt, "control_dt": env.step_dt,
                  "root_link": robot.body_names[0], "actor_training": policy.training,
                  "actuation_forces_note": "PhysX DOF actuation input buffer; do not assume implicit drive torque.",
                  "projected_torque_note": "PhysX joint-axis projection of incoming joint force, not isolated motor torque."}
        assert np.allclose(static["kp"], static["engine_kp"]) and np.allclose(static["kd"], static["engine_kd"])
        (folder / "static.json").write_text(json.dumps(static, indent=2))
        if nominal_mass is not None:
            print("[MASS]", case, array(robot.data.body_mass.torch).sum(1).tolist(), flush=True)
        print("[FOOT GEOMETRY]", shapes, flush=True)

        collect = False
        trajectory = None
        step_index, target_error = 0, 0.
        action = torch.zeros_like(default_q)
        compute0 = env.termination_manager.compute

        def capture(t, dones):
            nonlocal target_error
            d = robot.data
            expected = default_q + .25 * action
            target_error = max(target_error, float((tensor(robot.root_view.get_dof_position_targets()) - expected).abs().max()))
            masses = d.body_mass.torch
            com = (d.body_com_pos_w.torch * masses[..., None]).sum(1) / masses.sum(1, keepdim=True)
            comvel = (d.body_com_lin_vel_w.torch * masses[..., None]).sum(1) / masses.sum(1, keepdim=True)
            raw_contacts = [array(x) for x in sensor.contact_view.get_contact_data(dt=env.physics_dt)]
            forces, points, normals, separations, counts, starts = raw_contacts
            counts = counts.reshape(env.num_envs, sensor.num_bodies, -1)
            starts = starts.reshape(env.num_envs, sensor.num_bodies, -1)
            contact_points = np.full((env.num_envs, 2, 32, 3), np.nan)
            normal_force = np.zeros((env.num_envs, 2, 32))
            contact_separation = np.full((env.num_envs, 2, 32), np.nan)
            contact_count = np.zeros((env.num_envs, 2), dtype=int)
            hulls = np.full((env.num_envs, 64, 2), np.nan)
            margin, area = np.full(env.num_envs, np.nan), np.zeros(env.num_envs)
            foot_min_z = np.zeros((env.num_envs, 2))
            foot_pose = array(d.body_link_pose_w.torch[:, foot_ids])
            com_np = array(com)
            for e in range(env.num_envs):
                active = []
                for f, sid in enumerate(sensor_ids):
                    chunks = []
                    for k in range(counts.shape[-1]):
                        count, start = int(counts[e, sid, k]), int(starts[e, sid, k])
                        chunks.extend(range(start, start + count))
                    assert len(chunks) <= 32, "Increase diagnostic per-foot storage; never truncate support contacts"
                    n = len(chunks); contact_count[e, f] = n
                    contact_points[e, f, :n] = points[chunks]
                    normal_force[e, f, :n] = forces.reshape(-1)[chunks]
                    contact_separation[e, f, :n] = separations.reshape(-1)[chunks]
                    loaded = [i for i in chunks if forces.reshape(-1)[i] > 1e-4 and abs(normals[i, 2]) > .5]
                    if loaded:
                        active.extend(points[loaded, :2].tolist())
                    vertices = transform_vertices(local_vertices[f], foot_pose[e, f, :3], foot_pose[e, f, 3:])
                    foot_min_z[e, f] = vertices[:, 2].min()
                h, margin[e], area[e] = support_metrics(np.asarray(active).reshape(-1, 2), com_np[e, :2])
                hulls[e, :len(h)] = h
            record = {"time": np.asarray(t), "root_pose": array(d.root_link_pose_w.torch),
                      "root_velocity": array(d.root_com_vel_w.torch),
                      "rpy": array(torch.stack(euler_xyz_from_quat(d.root_quat_w.torch), 1)),
                      "com": com_np, "com_velocity": array(comvel), "foot_pose": foot_pose,
                      "foot_velocity": array(d.body_com_lin_vel_w.torch[:, foot_ids]),
                      "foot_forces": array(sensor.data.net_forces_w.torch[:, sensor_ids]),
                      "contact_points": contact_points, "normal_forces": normal_force,
                      "contact_separations": contact_separation, "contact_count": contact_count,
                      "contact_hull": hulls, "support_margin": margin, "support_area": area,
                      "foot_collision_min_z": foot_min_z,
                      "joint_pos": array(d.joint_pos.torch), "joint_vel": array(d.joint_vel.torch),
                      "joint_target": array(d.joint_pos_target.torch), "action": array(action),
                      "pd_torque": array(d.applied_torque.torch), "pd_unclipped": array(d.computed_torque.torch),
                      "instantaneous_pd": array(d.joint_stiffness.torch*(d.joint_pos_target.torch-d.joint_pos.torch)-d.joint_damping.torch*d.joint_vel.torch),
                      "actuation_input": array(robot.root_view.get_dof_actuation_forces()),
                      "projected_joint_force": array(robot.root_view.get_dof_projected_joint_forces()),
                      "done": array(dones), "bad_orientation": array(env.termination_manager.get_term("bad_orientation")),
                      "base_height": array(env.termination_manager.get_term("base_height")),
                      "timeout": array(env.termination_manager.get_term("time_out"))}
            for k, v in record.items():
                trajectory.setdefault(k, []).append(v)

        def compute():
            dones = compute0()
            if collect:
                capture((step_index+1)*env.step_dt, dones)
            return dones

        env.termination_manager.compute = compute
        for seed in args.seeds:
            collect = False
            with torch.inference_mode():
                obs0, _ = env.reset(seed=seed)
                obs = TensorDict(obs0, batch_size=[env.num_envs])
                policy.reset()
                action = torch.zeros_like(default_q)
                env.action_manager.process_action(action)
                env.action_manager.apply_action()
                env.scene.write_data_to_sim()
                if case == "plane":
                    # Generator curriculum consumes RNG even when no level
                    # changes. Restore the same initial pose relative to each
                    # origin so terrain representation is the only factor.
                    reference_initial = np.load(reference_dir / f"initial_{seed}.npz")
                    pose = reference_initial["root_pose"].copy()
                    pose[:, :3] += array(env.scene.env_origins) - reference_initial["origins"]
                    robot.write_root_link_pose_to_sim_index(root_pose=torch.tensor(pose, device=env.device))
                    env._invalidate_reset_root_cache()
                    env.scene.write_data_to_sim(); env.sim.forward()
                    env.observation_manager.reset()
                    obs = TensorDict(env.observation_manager.compute(update_history=True), batch_size=[env.num_envs])
                if case == "contact_start":
                    foot_pose = array(robot.data.body_link_pose_w.torch[:, foot_ids])
                    root_pose = robot.data.root_link_pose_w.torch.clone()
                    for e in range(env.num_envs):
                        lowest = min(transform_vertices(local_vertices[f], foot_pose[e, f, :3], foot_pose[e, f, 3:])[:, 2].min() for f in range(2))
                        root_pose[e, 2] += .001 - lowest
                    robot.write_root_link_pose_to_sim_index(root_pose=root_pose)
                    env._invalidate_reset_root_cache()
                    env.scene.write_data_to_sim(); env.sim.forward()
                    # Rebuild history from the changed reset state using the
                    # normal observation manager, only in this diagnostic arm.
                    env.observation_manager.reset()
                    obs = TensorDict(env.observation_manager.compute(update_history=True), batch_size=[env.num_envs])
                if args.inspect_geometry:
                    from pxr import Usd, UsdGeom
                    body_poses = array(robot.data.body_link_pose_w.torch)
                    bounds = []
                    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"])
                    for b, path in enumerate(robot.root_view.link_paths[0]):
                        # LocalBound includes this prim's authored transform;
                        # UntransformedBound is in the rigid link frame itself.
                        bound = bbox.ComputeUntransformedBound(env.sim.stage.GetPrimAtPath(path)).ComputeAlignedRange()
                        lo, hi = np.asarray(bound.GetMin()), np.asarray(bound.GetMax())
                        if not np.isfinite(lo).all() or not np.isfinite(hi).all():
                            continue
                        corners = np.array([[x,y,z] for x in [lo[0],hi[0]] for y in [lo[1],hi[1]] for z in [lo[2],hi[2]]])
                        bounds.append(transform_vertices(corners, body_poses[0,b,:3], body_poses[0,b,3:]))
                    points = np.concatenate(bounds)
                    (folder / f"geometry_{seed}.json").write_text(json.dumps({
                        "body_link_names": robot.body_names,
                        "env0_visual_envelope_min": points.min(0).tolist(),
                        "env0_visual_envelope_max": points.max(0).tolist(),
                        "env0_visual_envelope_height": float(points[:,2].max()-points[:,2].min()),
                        "note": "Union of native-transformed link-local USD bounding boxes; conservative visual envelope, not a support polygon."},indent=2))
                    np.savez_compressed(folder / f"body_poses_{seed}.npz", body_link_pose=body_poses,
                                        body_com_pose=array(robot.data.body_com_pose_w.torch),
                                        body_com_local=array(robot.data.body_com_pose_b.torch),
                                        mass=array(robot.data.body_mass.torch))
                initial = {"root_pose": array(robot.data.root_link_pose_w.torch),
                           "root_velocity": array(robot.root_view.get_root_velocities()),
                           "joint_pos": array(robot.data.joint_pos.torch), "joint_vel": array(robot.data.joint_vel.torch),
                           "obs_policy": array(obs["policy"]), "obs_critic": array(obs["critic"]),
                           "mass": array(robot.data.body_mass.torch),
                           "materials": array(robot.root_view.get_material_properties()),
                           "inertias": array(robot.root_view.get_inertias()),
                           "engine_kp": array(robot.root_view.get_dof_stiffnesses()), "engine_kd": array(robot.root_view.get_dof_dampings()),
                           "origins": array(env.scene.env_origins)}
                np.savez_compressed(folder / f"initial_{seed}.npz", **initial)
                trajectory = {}
                capture(0., torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
                collect = True
                first_terminated = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
                for step_index in range(round(args.seconds/env.step_dt)):
                    gain = {"policy1": 1., "policy05": .5, "policy025": .25}.get(case, 0.)
                    action = policy(obs)*gain
                    if case == "ankle_bias_p10":
                        ids, _ = robot.find_joints(["left_ankle_pitch_joint", "right_ankle_pitch_joint"], preserve_order=True)
                        action[:, ids] = .4  # bilateral target bias +0.10 rad; scale stays 0.25
                    obs, _, done, _ = wrapped.step(action)
                    assert torch.all(command.command == 0)
                    policy.reset(done)
                    first_terminated |= done.bool()
                    if args.stop_after_first_failures and first_terminated.all():
                        break
                collect = False
            data = {k: np.stack(v) for k, v in trajectory.items()}
            np.savez_compressed(folder / f"trajectory_{seed}.npz", **data)
            failures = [float(data["time"][np.flatnonzero(data["done"][:, e])[0]]) if data["done"][:, e].any() else None for e in range(env.num_envs)]
            reports.append({"case": case, "seed": seed, "first_failure_s": failures,
                            "observed_seconds": float(data["time"][-1]),
                            "survival_2s": sum(x is None or x > 2 for x in failures)/env.num_envs if args.seconds >= 2 else None,
                            "survival_5s": sum(x is None or x > 5 for x in failures)/env.num_envs if args.seconds >= 5 else None,
                            "target_formula_max_error": target_error})
            (out / "results.json").write_text(json.dumps(reports, indent=2))
            print("[RESULT]", json.dumps(reports[-1]), flush=True)
        assert target_error < 1e-6
        wrapped.close()
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == sha
    (out / "manifest.json").write_text(json.dumps({"checkpoint": str(checkpoint), "sha256": sha,
        "num_envs": args.num_envs, "seeds": args.seeds, "seconds": args.seconds, "cases": args.cases,
        "versions": {n: metadata.version(n) for n in ["isaacsim", "isaaclab", "torch", "rsl-rl-lib"]},
        "production_changes": False, "new_training_iterations": 0, "reference": args.reference}, indent=2))
    print("[PASS] Posture/PD diagnostic complete", flush=True)


exit_code = 0
try:
    main()
except BaseException:
    exit_code = 1
    traceback.print_exc()
finally:
    launcher.app.close(wait_for_replicator=False, exit_code=exit_code)
