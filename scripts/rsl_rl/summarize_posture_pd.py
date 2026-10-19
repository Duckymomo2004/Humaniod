"""Offline analysis of G1 posture, actual contact polygons and PD ablations."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

from posture_geometry import hull2d, transform_vertices

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", default="run/posture-pd-20261019")
args = parser.parse_args()
root = Path(args.root)
out = root / "analysis"
out.mkdir(exist_ok=True)
locations = {c: root / "initial_conditions" / c for c in ["zero", "direct_default", "nominal_mass", "fixed_friction", "fixed_pose", "nominal_mass_friction", "controlled", "contact_start"]}
locations["no_push"] = root / "no_push_paired/no_push"
locations["plane"] = root / "plane_paired/plane"
locations.update({c: root / "pd_policy" / c for c in ["ankle_kp2", "ankle_kd2", "policy1", "policy05", "policy025"]})
locations.update({c: root / "adaptive_controls" / c for c in ["ankle_kp4", "ankle_bias_p10"]})
locations["ankle_kp8"] = root / "kp8/ankle_kp8"
order = ["zero", "direct_default", "no_push", "nominal_mass", "fixed_friction", "fixed_pose", "nominal_mass_friction", "controlled", "plane", "contact_start", "policy1", "policy05", "policy025", "ankle_kd2", "ankle_kp2", "ankle_kp4", "ankle_kp8", "ankle_bias_p10"]
parents = {c: "zero" for c in order if c != "zero"}
parents.update(nominal_mass_friction="nominal_mass", controlled="nominal_mass_friction")
allowed = {"nominal_mass": {"mass", "inertias"}, "fixed_friction": {"materials"},
           "fixed_pose": {"root_pose", "obs_policy", "obs_critic"},
           "nominal_mass_friction": {"materials"}, "controlled": {"root_pose", "obs_policy", "obs_critic"},
           "contact_start": {"root_pose", "obs_policy", "obs_critic"},
           "plane": {"obs_policy", "obs_critic"},
           "ankle_kp2": {"engine_kp"}, "ankle_kp4": {"engine_kp"}, "ankle_kp8": {"engine_kp"},
           "ankle_kd2": {"engine_kd"}}
pairing, results, joint_rows = [], [], []
cached = {}
for case in order:
    folder = locations[case]
    static = json.loads((folder / "static.json").read_text())
    names = static["joint_names"]
    metrics, qerrors, torques, estimated_saturation = [], [], [], []
    for seed in [42, 43, 44]:
        initial = np.load(folder / f"initial_{seed}.npz")
        if case != "zero":
            base = np.load(locations[parents[case]] / f"initial_{seed}.npz")
            errors = {}
            for key in initial.files:
                if key == "origins":
                    continue
                a, b = initial[key].copy(), base[key].copy()
                if key == "root_pose":
                    a[:, :3] -= initial["origins"]
                    b[:, :3] -= base["origins"]
                errors[key] = float(np.max(np.abs(a-b)))
            unexpected = {k: v for k, v in errors.items() if v > 1e-5 and k not in allowed.get(case, set())}
            assert not unexpected, (case, seed, unexpected)
            # The settling probe changes only root z, never xy/orientation/q/qd.
            if case == "contact_start":
                assert np.max(np.abs(initial["root_pose"][:, [0, 1, 3, 4, 5, 6]] - base["root_pose"][:, [0, 1, 3, 4, 5, 6]])) < 1e-5
            pairing.append({"case": case, "parent": parents[case], "seed": seed, "max_errors": errors, "unexpected": unexpected})
        z = np.load(folder / f"trajectory_{seed}.npz")
        if seed == 42:
            cached[case] = z
        for e in range(z["done"].shape[1]):
            failed = np.flatnonzero(z["done"][:, e])
            stop = int(failed[0]) if len(failed) else len(z["time"])-1
            duration = float(z["time"][stop])
            assert len(failed) or duration >= 5, "Censor only after the full requested horizon"
            why = [k for k in ["bad_orientation", "base_height", "timeout"] if z[k][stop, e]] if len(failed) else ["censored_5s"]
            sl = slice(1, stop+1)
            margin = z["support_margin"][:stop+1, e]
            outside = np.flatnonzero(np.isfinite(margin) & (margin < 0))
            torque = z["pd_torque"][sl, e]
            error = z["joint_target"][sl, e] - z["joint_pos"][sl, e]
            limit = np.asarray(static["effort_limits"])[e]
            saturation = (abs(z["pd_unclipped"][sl, e]) > limit + 1e-4).sum(0)
            metrics.append({"seed": seed, "env_id": e, "first_failure_s": duration if len(failed) else None,
                            "observed_s": duration, "termination": why,
                            "survive_2s": not len(failed) or duration > 2., "survive_5s": not len(failed) or duration > 5.,
                            "first_com_outside_active_hull_s": float(z["time"][outside[0]]) if len(outside) else None,
                            "max_PD_estimate_Nm": float(abs(torque).max()), "estimate_clipped_samples": int(saturation.sum()),
                            "max_incoming_projection_Nm": float(abs(z["projected_joint_force"][sl, e]).max()),
                            "max_actuation_input_Nm": float(abs(z["actuation_input"][sl, e]).max()),
                            "max_contact_force_sample_N": float(np.max(z["foot_forces"][sl, e, :, 2].sum(1)))})
            qerrors.append(error); torques.append(torque); estimated_saturation.append(saturation)
    duration = np.array([m["observed_s"] for m in metrics])
    result = {"case": case, "episodes": len(metrics), "first_failure_mean_or_lower_bound_s": float(duration.mean()),
              "std_s": float(duration.std()), "min_s": float(duration.min()), "max_s": float(duration.max()),
              "censored_count": sum(m["first_failure_s"] is None for m in metrics),
              "survive_2s": float(np.mean([m["survive_2s"] for m in metrics])),
              "survive_5s": float(np.mean([m["survive_5s"] for m in metrics])), "metrics": metrics}
    results.append(result)
    qerr, tau = np.concatenate(qerrors), np.concatenate(torques)
    saturation = np.sum(estimated_saturation, 0)
    for j, name in enumerate(names):
        joint_rows.append({"case": case, "joint": name, "q_error_rms_rad": float(np.sqrt(np.mean(qerr[:, j]**2))),
                           "q_error_peak_rad": float(np.max(abs(qerr[:, j]))), "PD_estimate_peak_Nm": float(np.max(abs(tau[:, j]))),
                           "PD_estimate_rms_Nm": float(np.sqrt(np.mean(tau[:, j]**2))), "clipped_samples": int(saturation[j]),
                           "kp": static["kp"][0][j], "kd": static["kd"][0][j], "limit_Nm": static["effort_limits"][0][j]})

with (out / "standing_summary.csv").open("w", newline="") as f:
    keys = [k for k in results[0] if k != "metrics"]
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows({k: r[k] for k in keys} for r in results)
with (out / "joint_metrics.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(joint_rows[0])); w.writeheader(); w.writerows(joint_rows)
setup = json.loads((locations["zero"] / "static.json").read_text())
with (out / "joint_setup.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["actor_index", "joint", "default_rad", "offset_rad", "scale", "kp", "kd", "effort_limit_Nm", "velocity_limit_radps", "lower_rad", "upper_rad"])
    for j, name in enumerate(setup["joint_names"]):
        w.writerow([j, name, setup["default_joint_pos"][0][j], setup["action_offset"][0][j], setup["action_scale"],
                    setup["kp"][0][j], setup["kd"][0][j], setup["effort_limits"][0][j], setup["velocity_limits"][0][j], *setup["joint_limits"][0][j]])
with (out / "body_masses.csv").open("w", newline="") as f:
    w = csv.writer(f); w.writerow(["body", "nominal_kg", *[f"env_{e}_kg" for e in range(8)]])
    for b, name in enumerate(setup["body_names"]):
        w.writerow([name, setup["nominal_mass"][0][b], *np.asarray(setup["mass"])[:, b].tolist()])

fig, ax = plt.subplots(figsize=(10, 9), constrained_layout=True)
ax.barh(np.arange(len(results)), [r["first_failure_mean_or_lower_bound_s"] for r in results],
        xerr=[r["std_s"] for r in results], color=["#cc6e23" if r["case"].startswith("policy") else "#267891" for r in results], capsize=2)
ax.set(yticks=np.arange(len(results)), yticklabels=order, xlabel="first failure / censoring horizon (s)",
       title="G1 posture and control — 8 envs × 3 reset seeds; spread is std, not CI")
ax.invert_yaxis(); ax.axvline(2, linestyle="--", color="grey"); ax.axvline(5, linestyle=":", color="grey")
ax.grid(axis="x", alpha=.2)
for ext in ["png", "pdf"]: fig.savefig(out/f"comparison.{ext}", dpi=160)
plt.close(fig)

z = cached["zero"]
stop = int(np.flatnonzero(z["done"][:, 0])[0]); sl = slice(0, stop+1); t = z["time"][sl]
yaw = z["rpy"][0, 0, 2]; c, s = np.cos(yaw), np.sin(yaw)
rotation = np.array([[c, -s], [s, c]])
origin = z["foot_pose"][0, 0, :, :2].mean(0)
com_xy = (z["com"][sl, 0, :2]-origin) @ rotation
cop = []
for i in range(stop+1):
    pts, f = z["contact_points"][i, 0].reshape(-1, 3), z["normal_forces"][i, 0].reshape(-1)
    valid = np.isfinite(pts).all(1) & (f > 1e-4)
    cop.append(((pts[valid, :2]*f[valid, None]).sum(0)/f[valid].sum()-origin) @ rotation if valid.any() else [np.nan, np.nan])
cop = np.asarray(cop)
fig, axs = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
axs[0, 0].plot(t, np.rad2deg(z["rpy"][sl, 0, 0]), label="roll")
axs[0, 0].plot(t, np.rad2deg(z["rpy"][sl, 0, 1]), label="pitch")
axs[0, 1].plot(t, com_xy[:, 0], label="COM x")
axs[0, 1].plot(t, cop[:, 0], label="normal-force CoP x")
axs[1, 0].plot(t, z["support_margin"][sl, 0], label="min signed edge distance")
axs[1, 0].axhline(0, color="black", linewidth=1)
axs[1, 1].plot(t, z["foot_forces"][sl, 0, 0, 2], label="left foot")
axs[1, 1].plot(t, z["foot_forces"][sl, 0, 1, 2], label="right foot")
axs[2, 0].plot(t, z["joint_target"][sl, 0, 13]-z["joint_pos"][sl, 0, 13], label="L ankle pitch")
axs[2, 0].plot(t, z["joint_target"][sl, 0, 14]-z["joint_pos"][sl, 0, 14], label="R ankle pitch")
axs[2, 1].plot(t, z["pd_torque"][sl, 0, 13], label="L PD estimate")
axs[2, 1].plot(t, z["projected_joint_force"][sl, 0, 13], label="L incoming projection", linestyle=":")
axs[2, 1].plot(t, z["actuation_input"][sl, 0, 13], label="actuation input buffer", linestyle="--")
for ax, title, units in zip(axs.flat, ["Base orientation", "COM vs normal reaction center", "Active contact hull margin", "Foot vertical forces (20 ms samples)", "Ankle tracking error", "Different torque signals"], ["deg", "m (initial yaw frame)", "m", "N", "rad", "Nm"]):
    ax.set(title=title, xlabel="simulation time (s)", ylabel=units); ax.legend(fontsize=8); ax.grid(alpha=.2)
fig.suptitle("Zero action, seed 42, env 0/8; first episode only")
for ext in ["png", "pdf"]: fig.savefig(out/f"zero_dynamics.{ext}", dpi=160)
plt.close(fig)

fig, axs = plt.subplots(1, 4, figsize=(15, 4), constrained_layout=True)
static = json.loads((locations["zero"] / "static.json").read_text())
for ax, ti in zip(axs, [0, 3, 25, 35]):
    # Grey geometry is only a projected potential sole outline, not contact.
    for f in range(2):
        local = np.asarray(static["foot_local_vertices"][f]); local = local[local[:, 2] <= local[:, 2].min()+1e-5]
        vertices = transform_vertices(local, z["foot_pose"][ti, 0, f, :3], z["foot_pose"][ti, 0, f, 3:])
        polygon = hull2d((vertices[:, :2]-origin) @ rotation)
        if len(polygon) >= 3: ax.add_patch(Polygon(polygon, facecolor="lightgrey", edgecolor="grey", alpha=.4))
    pts = z["contact_points"][ti, 0].reshape(-1, 3); force = z["normal_forces"][ti, 0].reshape(-1)
    valid = np.isfinite(pts).all(1) & (force > 1e-4)
    pp = (pts[valid, :2]-origin) @ rotation
    if len(pp): ax.scatter(pp[:, 0], pp[:, 1], s=15+force[valid]*.1, color="#247b5c")
    hull = z["contact_hull"][ti, 0]; hull = hull[np.isfinite(hull).all(1)]
    if len(hull) >= 3: ax.add_patch(Polygon((hull-origin) @ rotation, fill=False, edgecolor="#247b5c", linewidth=2))
    ax.scatter([com_xy[ti, 0]], [com_xy[ti, 1]], marker="x", color="red", s=70, label="COM")
    if np.isfinite(cop[ti]).all(): ax.scatter([cop[ti, 0]], [cop[ti, 1]], color="blue", marker="+", s=60, label="normal CoP")
    ax.set(title=f"t={z['time'][ti]:.2f}s\nactive margin={z['support_margin'][ti,0]:.3f} m", xlabel="forward x (m)", ylabel="left y (m)", xlim=(-.08,.18), ylim=(-.19,.19), aspect="equal")
    ax.grid(alpha=.2); ax.legend(fontsize=7, loc="upper left")
fig.suptitle("Actual loaded contact hull (green); grey = projected sole points, not evidence of contact")
for ext in ["png", "pdf"]: fig.savefig(out/f"support_polygons.{ext}", dpi=160)
plt.close(fig)

fig, axs = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
for case in ["zero", "contact_start", "ankle_kd2", "ankle_kp2", "ankle_kp4", "ankle_kp8", "ankle_bias_p10"]:
    zz = cached[case]; end = int(np.flatnonzero(zz["done"][:, 0])[0]); part = slice(0, end+1); tt = zz["time"][part]
    axs[0, 0].plot(tt, np.rad2deg(zz["rpy"][part, 0, 1]), label=case)
    axs[0, 1].plot(tt, zz["support_margin"][part, 0], label=case)
    axs[1, 0].plot(tt, zz["joint_target"][part, 0, 13]-zz["joint_pos"][part, 0, 13], label=case)
    axs[1, 1].plot(tt, zz["pd_torque"][part, 0, 13], label=case)
for ax, title, unit in zip(axs.flat, ["Pitch", "Active hull margin", "Left ankle pitch tracking error", "Left ankle PD estimate"], ["deg", "m", "rad", "Nm"]):
    ax.set(title=title, xlabel="simulation time (s)", ylabel=unit); ax.legend(fontsize=7); ax.grid(alpha=.2)
for ext in ["png", "pdf"]: fig.savefig(out/f"pd_sensitivity.{ext}", dpi=160)
plt.close(fig)

with (out / "zero_env0_timeseries.csv").open("w", newline="") as f:
    w = csv.writer(f); w.writerow(["t", "roll_deg", "pitch_deg", "COM_forward_m", "CoP_forward_m", "active_margin_m", "active_area_m2", "Fz_left_N", "Fz_right_N", "com_vx_world", "com_vy_world", "com_vz_world"])
    for i in range(stop+1): w.writerow([float(t[i]), *np.rad2deg(z["rpy"][i,0,:2]).tolist(), float(com_xy[i,0]), float(cop[i,0]), float(z["support_margin"][i,0]), float(z["support_area"][i,0]), *z["foot_forces"][i,0,:,2].tolist(), *z["com_velocity"][i,0].tolist()])
(out / "summary.json").write_text(json.dumps({"pairing": pairing, "results": results, "source_dirs": {k: str(v) for k,v in locations.items()}}, indent=2, allow_nan=False))
print(out.resolve())
