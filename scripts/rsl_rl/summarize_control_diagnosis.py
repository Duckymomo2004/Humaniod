"""Offline plots and CSVs for the paired control diagnosis (no simulator imports)."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", default="run/control-diagnosis-20261019")
args = parser.parse_args()
root = Path(args.root)
out = root / "analysis"
out.mkdir(exist_ok=True)
audit = json.loads((root / "first_half_second/gain_1p0/control_audit.json").read_text())
names = audit["joint_names"]
arms = {"model_199": "gain_1p0", "zero_actions": "gain_0p0"}
data = {k: np.load(root / "first_half_second" / v / "trajectory_seed_42.npz") for k, v in arms.items()}
summary = {"notes": ["All time traces: seed 42, env 0 of 8 paired robots.",
                     "Auto-reset suppressed only for this 0.5 s trace; shaded portion is after original terminal criterion.",
                     "PD estimate and incoming-joint projection are not measured pure motor torques."]}
colors = {"model_199": "#d35400", "zero_actions": "#176b9c"}
fig, axs = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
for label, z in data.items():
    t = z["time"]
    color = colors[label]
    failures = np.flatnonzero(z["done"][:, 0])
    stop = int(failures[0]) if failures.size else len(t) - 1
    rpy = np.rad2deg(z["roll_pitch_yaw"][:, 0])
    delta = z["com_world"][:, 0] - z["com_world"][0, 0]
    yaw = z["roll_pitch_yaw"][0, 0, 2]
    c, s = np.cos(yaw), np.sin(yaw)
    local_xy = delta[:, :2] @ np.array([[c, -s], [s, c]])
    axs[0, 0].plot(t, rpy[:, 0], color=color, linestyle="--", label=f"{label} roll")
    axs[0, 0].plot(t, rpy[:, 1], color=color, label=f"{label} pitch")
    axs[0, 1].plot(t, z["com_world"][:, 0, 2], color=color, label=label)
    axs[1, 0].plot(t, local_xy[:, 0], color=color, label=f"{label} X")
    axs[1, 0].plot(t, local_xy[:, 1], color=color, linestyle="--", label=f"{label} Y")
    axs[1, 1].plot(t, z["base_velocity_yaw"][:, 0, 0], color=color, label=f"{label} vx")
    axs[1, 1].plot(t, z["base_velocity_yaw"][:, 0, 1], color=color, linestyle="--", label=f"{label} vy")
    axs[2, 0].plot(t, z["foot_forces_world"][:, 0, 0, 2], color=color, label=f"{label} L")
    axs[2, 0].plot(t, z["foot_forces_world"][:, 0, 1, 2], color=color, linestyle="--", label=f"{label} R")
    axs[2, 1].plot(t, np.sqrt(np.mean(z["action"][:, 0]**2, axis=1)), color=color, label=label)
    limits = np.asarray(audit["effort_limits"])
    summary[label] = {
        "first_bad_orientation_env0_s": float(t[stop]) if failures.size else None,
        "rms_action_before_terminal_all_8": float(np.sqrt(np.mean(z["action"][1:stop+1]**2))),
        "pd_estimate_saturation_count_before_env0_terminal": int(np.count_nonzero(np.abs(z["torque_unclipped"][1:stop+1]) > limits[None] + 1e-4)),
        "env0_first_action": z["action"][1, 0].tolist(),
        "env0_first_target": z["joint_target"][1, 0].tolist(),
        "samples_env0": [{"time_s": float(t[i]), "roll_pitch_deg": rpy[i, :2].tolist(),
                          "com_displacement_initial_yaw_m": [*local_xy[i].tolist(), float(delta[i, 2])],
                          "com_height_m": float(z["com_world"][i, 0, 2]),
                          "base_velocity_yaw_mps": z["base_velocity_yaw"][i, 0].tolist(),
                          "feet_Fz_N": z["foot_forces_world"][i, 0, :, 2].tolist(),
                          "post_terminal": bool(failures.size and i >= stop)}
                         for i in [0, 1, 2, 3, 5, 10, 13, 25]],
    }
    with (out / f"{label}_all_joints.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "env_id", "joint", "q_rad", "qd_radps", "target_rad", "action", "PD_estimate_Nm", "incoming_projected_Nm", "past_original_terminal"])
        terminal = np.maximum.accumulate(z["done"], axis=0)
        for ti, ts in enumerate(t):
            for e in range(z["joint_pos"].shape[1]):
                for j, name in enumerate(names):
                    writer.writerow([float(ts), e, name, *[float(z[k][ti, e, j]) for k in ["joint_pos", "joint_vel", "joint_target", "action", "torque_estimate", "projected_joint_torque"]], bool(terminal[ti, e])])

titles = [("Roll / pitch (deg)", "deg"), ("Whole robot COM height", "m"),
          ("COM displacement in initial yaw frame", "m"), ("Base velocity in current yaw frame", "m/s"),
          ("Foot net vertical contact force", "N"), ("Applied action RMS, env 0", "actor units")]
for ax, (title, unit) in zip(axs.flat, titles):
    ax.set(title=title, xlabel="simulation time (s)", ylabel=unit)
    ax.axvspan(.26, .5, color="#d35400", alpha=.07)
    ax.axvline(.26, color="#d35400", alpha=.5, linewidth=1)
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
fig.suptitle("G1 control diagnosis — shaded: model_199 past original terminal (env 0)")
for suffix in ["png", "pdf"]:
    fig.savefig(out / f"first_half_second.{suffix}", dpi=160)
plt.close(fig)

fig, axs = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
selected = ["left_knee_joint", "right_knee_joint", "left_ankle_pitch_joint", "right_ankle_pitch_joint"]
for ax, name in zip(axs.flat, selected):
    j = names.index(name)
    for label, z in data.items():
        ax.plot(z["time"], z["joint_pos"][:, 0, j], color=colors[label], label=f"{label} q")
        ax.plot(z["time"], z["joint_target"][:, 0, j], color=colors[label], linestyle="--", label=f"{label} target")
    ax.set(title=name, ylabel="rad")
for label, z in data.items():
    axs[2, 0].plot(z["time"], np.max(np.abs(z["torque_estimate"][:, 0]), axis=1), color=colors[label], label=f"{label} PD estimate")
    axs[2, 0].plot(z["time"], np.max(np.abs(z["projected_joint_torque"][:, 0]), axis=1), color=colors[label], linestyle=":", label=f"{label} incoming projected")
axs[2, 0].set(title="Maximum absolute torque across joints", ylabel="Nm")
z = data["model_199"]
im = axs[2, 1].imshow(z["action"][1:, 0].T, aspect="auto", origin="lower", extent=[.01, .51, -.5, 28.5], cmap="coolwarm", vmin=-4, vmax=4)
axs[2, 1].set(title="model_199 action, all 29 outputs", ylabel="actor output index")
fig.colorbar(im, ax=axs[2, 1], label="actor units")
for ax in axs.flat:
    ax.axvspan(.26, .5, color="grey", alpha=.07)
    ax.axvline(.26, color="grey", alpha=.5, linewidth=1)
    ax.set_xlabel("simulation time (s)")
    if ax is not axs[2, 1]:
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
fig.suptitle("Joint control — env 0; shaded region is post-terminal diagnostic only")
for suffix in ["png", "pdf"]:
    fig.savefig(out / f"joint_control.{suffix}", dpi=160)
plt.close(fig)

summary["fresh_seed_42_amplitude"] = json.loads((root / "fresh_seed_42/results.json").read_text())
summary["cache_refresh_ab"] = json.loads((root / "cache_refresh_ab/results.json").read_text())
for case in ["returned_obs_baseline", "returned_obs_refreshed"]:
    path = root / case / "gain_1p0/returned_observation_audit.json"
    if path.exists():
        summary[case] = json.loads(path.read_text())
(out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
print(out.resolve())
