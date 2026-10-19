"""Offline summary of project reset regression and matched standing tests."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", default="run/reset-standing-20261019")
args = parser.parse_args()
root = Path(args.root)
out = root / "analysis"
out.mkdir(exist_ok=True)
resets, standing = [], []
for n in [1, 8, 64]:
    for mode in ["baseline", "fixed"]:
        r = json.loads((root / f"reset_{mode}_{n}/result.json").read_text())
        assert r["passed"] and r["baseline_expected_failure_reproduced"] == (mode == "baseline")
        errors = {k: max(row["errors"][k] for row in r["records"]) for k in r["records"][0]["errors"]}
        resets.append({"num_envs": n, "mode": mode, "auto_resets": sum(r["auto_resets_per_env"]),
                       "violating_events": r["violating_observation_events"],
                       "critic_angular_error_radps": errors["critic/base_ang_vel"] / .2,
                       "critic_gravity_error": errors["critic/projected_gravity"],
                       "critic_linear_error_mps": errors["critic/base_lin_vel"],
                       "policy_angular_scaled_error": errors["policy/base_ang_vel"],
                       "policy_gravity_error": errors["policy/projected_gravity"],
                       "all_observation_errors": errors, "network_checks": r["network_input_checks"],
                       "max_setter_angular_error": max([v["angular_error"] for v in r["setter_checks"]], default=None),
                       "max_non_reset_pose_change": max(v["non_reset_native_pose_change"] for v in r["reset_checks"]),
                       "max_non_reset_velocity_change": max(v["non_reset_native_velocity_change"] for v in r["reset_checks"])})
    folder = root / f"standing_{n}"
    manifest = json.loads((folder / "manifest.json").read_text())
    assert not manifest["diagnostic_refresh_reset_cache"]  # project fix alone
    assert len(manifest["pairing"]) == 9
    for comparison in manifest["pairing"]:
        assert max(comparison["max_errors"].values()) < 1e-5
    rows = json.loads((folder / "results.json").read_text())
    for gain in [1., .5, .25, 0.]:
        arm = folder / ("gain_" + str(gain).replace(".", "p"))
        audit = json.loads((arm / "control_audit.json").read_text())
        q0 = np.asarray(audit["default_joint_pos"])
        limits = np.asarray(audit["effort_limits"])
        metrics = []
        for seed in manifest["seeds"]:
            z = np.load(arm / f"trajectory_seed_{seed}.npz")
            for e in range(n):
                failures = np.flatnonzero(z["done"][:, e])
                assert len(failures), "Record right-censoring explicitly if this becomes a successful policy"
                stop = int(failures[0])
                assert z["bad_orientation"][stop, e]
                pre = slice(1, stop + 1)
                com_delta = z["com_world"][:stop+1, e] - z["com_world"][0, e]
                metrics.append({"seed": seed, "env_id": e, "first_failure_s": float(z["time"][stop]),
                                "roll_terminal_deg": float(np.rad2deg(z["roll_pitch_yaw"][stop, e, 0])),
                                "pitch_terminal_deg": float(np.rad2deg(z["roll_pitch_yaw"][stop, e, 1])),
                                "max_com_xy_displacement_m": float(np.linalg.norm(com_delta[:, :2], axis=1).max()),
                                "com_height_initial_m": float(z["com_world"][0, e, 2]),
                                "com_height_terminal_m": float(z["com_world"][stop, e, 2]),
                                "action_rms": float(np.sqrt(np.mean(z["action"][pre, e]**2))),
                                "q_deviation_rms_rad": float(np.sqrt(np.mean((z["joint_pos"][pre, e] - q0[e])**2))),
                                "peak_abs_PD_estimate_Nm": float(np.abs(z["torque_estimate"][pre, e]).max()),
                                "PD_estimate_saturated_samples": int(np.count_nonzero(np.abs(z["torque_unclipped"][pre, e]) > limits[e] + 1e-4))})
        duration = np.array([v["first_failure_s"] for v in metrics])
        standing.append({"num_envs": n, "gain": gain, "first_episode_count": len(metrics),
                         "mean_s": float(duration.mean()), "std_s": float(duration.std()),
                         "min_s": float(duration.min()), "max_s": float(duration.max()),
                         "survival_2s_fraction": float(np.mean(duration >= 2.)),
                         "metrics": metrics})
with (out / "standing_summary.csv").open("w", newline="") as f:
    keys = [k for k in standing[0] if k != "metrics"]
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
    w.writerows({k: v[k] for k in keys} for v in standing)
with (out / "first_episodes.csv").open("w", newline="") as f:
    keys = ["num_envs", "gain", *standing[0]["metrics"][0]]
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
    for row in standing:
        w.writerows({"num_envs": row["num_envs"], "gain": row["gain"], **m} for m in row["metrics"])
nonzero = json.loads((root / "reset_nonzero_8/result.json").read_text())
assert nonzero["passed"] and nonzero["nonzero_reset_observation_verified"]
nonzero_row = next(row for row in nonzero["records"] if row["step"] == -3)
assert nonzero_row["native_velocity_max"] > .6 and not nonzero_row["violations"]
(out / "summary.json").write_text(json.dumps({"reset_tests": resets, "nonzero_reset": nonzero_row,
                                             "standing": standing}, indent=2, allow_nan=False))

fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
for i, n in enumerate([1, 8, 64]):
    rows = [r for r in standing if r["num_envs"] == n]
    ax.bar(np.arange(4) + (i-1)*.25, [r["mean_s"] for r in rows], width=.24,
           yerr=[r["std_s"] for r in rows], capsize=3, label=f"{n} envs")
ax.set(xticks=np.arange(4), xticklabels=["model_199", "gain 0.5", "gain 0.25", "zero actions"],
       ylabel="first failure (simulation seconds)", title="Project reset fix active — 3 reset seeds, paired initial states")
ax.text(.99, .98, "Error bars: sample spread (std), not confidence intervals", ha="right", va="top", transform=ax.transAxes, fontsize=8)
ax.legend(); ax.grid(axis="y", alpha=.2)
for ext in ["png", "pdf"]:
    fig.savefig(out / f"standing_comparison.{ext}", dpi=160)
plt.close(fig)

fig, axs = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
colors = {1.: "#c64d00", .5: "#bd8b00", .25: "#308c51", 0.: "#176b9c"}
for gain in [1., .5, .25, 0.]:
    arm = root / "standing_8" / ("gain_" + str(gain).replace(".", "p"))
    z = np.load(arm / "trajectory_seed_42.npz")
    stop = int(np.flatnonzero(z["done"][:, 0])[0])
    end = min(stop, 25)
    sl = slice(0, end+1)
    t, color = z["time"][sl], colors[gain]
    label = "zero" if gain == 0 else f"gain {gain}"
    axs[0, 0].plot(t, np.rad2deg(z["roll_pitch_yaw"][sl, 0, 1]), color=color, label=label)
    axs[0, 1].plot(t, z["com_world"][sl, 0, 2], color=color, label=label)
    axs[1, 0].plot(t, z["foot_forces_world"][sl, 0, 0, 2], color=color, label=label+" L")
    axs[1, 0].plot(t, z["foot_forces_world"][sl, 0, 1, 2], color=color, linestyle="--", label=label+" R")
    axs[1, 1].plot(t, np.max(np.abs(z["torque_estimate"][sl, 0]), axis=1), color=color, label=label)
    axs[2, 0].plot(t, z["joint_pos"][sl, 0, 13], color=color, label=label+" L")
    axs[2, 0].plot(t, z["joint_pos"][sl, 0, 14], color=color, linestyle="--", label=label+" R")
    axs[2, 1].plot(t, np.rad2deg(z["roll_pitch_yaw"][sl, 0, 0]), color=color, label=label)
for ax, title, unit in zip(axs.flat, ["Pitch", "Whole robot COM height", "Foot vertical contact forces", "Max absolute PD torque estimate", "Ankle pitch q (L solid, R dashed)", "Roll"], ["deg", "m", "N", "Nm", "rad", "deg"]):
    ax.set(title=title, xlabel="simulation time (s)", ylabel=unit)
    ax.grid(alpha=.2); ax.legend(fontsize=7)
fig.suptitle("First episode, seed 42, env 0/8 — traces stop at original terminal; no post-terminal data")
for ext in ["png", "pdf"]:
    fig.savefig(out / f"standing_traces.{ext}", dpi=160)
plt.close(fig)
print(out.resolve())
