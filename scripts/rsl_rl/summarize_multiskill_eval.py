"""Create standalone plots and CSV tables from measured evaluation data."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input", required=True)
args = parser.parse_args()
root = Path(args.input)
rows = json.loads((root / "results.json").read_text())
assert len(rows) == 13, "Wait for both checkpoints and the zero-action reference"
for row in rows:
    data = np.load(root / row["checkpoint"] / f"{row['scenario']}.npz")
    lengths = data["terminal_episode_steps"]
    # Correctly count only tails with at least one observed step as censored episodes.
    last_ends = np.max(np.where(lengths > 0, np.arange(len(lengths))[:, None], -1), axis=0)
    row["censored_episodes"] = int(np.sum(len(lengths) - last_ends - 1 > 0))
    assert row["completed_episodes"] == int((lengths > 0).sum())
    assert np.isfinite(data["features"]).all()
with (root / "summary.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
(root / "summary.json").write_text(json.dumps(rows, indent=2))

by_key = {(r["checkpoint"], r["scenario"]): r for r in rows}
scenarios = ["standing", "forward_slow", "turn_left", "turn_right", "forward_0p5"]
labels = ["Stand\n0, 0, 0", "Forward\n0.1 m/s", "Turn left\n0.1 rad/s", "Turn right\n-0.1 rad/s", "Forward\n0.5 m/s (OOD)"]
colors = {"model_100": "#64748b", "model_199": "#0284c7", "zero_actions": "#16a34a"}
fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
for ax, key, ylabel in [(axes[0, 0], "episode_mean_s", "Mean completed episode (s)"),
                         (axes[0, 1], "velocity_xy_rmse_mps", "XY tracking RMSE (m/s)"),
                         (axes[1, 0], "yaw_rate_rmse_radps", "Yaw tracking RMSE (rad/s)")]:
    x = np.arange(len(scenarios))
    for offset, model in [(-.18, "model_100"), (.18, "model_199")]:
        ax.bar(x + offset, [by_key[model, s][key] for s in scenarios], .35, label=model, color=colors[model])
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    ax.legend()
ax = axes[1, 1]
for model in ["model_100", "model_199", "zero_actions"]:
    data = np.load(root / model / "standing.npz")
    failure = data["first_failure_steps"].astype(float) * float(data["dt"])
    failure[data["first_failure_steps"] < 0] = np.inf
    t = np.linspace(0, 3, 301)
    ax.plot(t, [(failure > x).mean() * 100 for x in t], label=model, color=colors[model], lw=2)
ax.set(xlabel="Time from first reset (s)", ylabel="First episodes still alive (%)", title="Standing: survival, 64 robots")
ax.set_ylim(-2, 102)
ax.legend()
ax.grid(alpha=.2)
fig.suptitle("G1 29 DoF — deterministic policy evaluation\n64 environments, 20 simulated seconds per condition; training noise/randomization retained", fontsize=13)
fig.savefig(root / "evaluation-comparison.png", dpi=170)
fig.savefig(root / "evaluation-comparison.pdf")
plt.close(fig)

fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
for model in ["model_100", "model_199", "zero_actions"]:
    data = np.load(root / model / "standing.npz")
    first_failure = int(data["first_failure_steps"][0])
    end = first_failure if first_failure > 0 else len(data["features"])
    features = data["features"][:end, 0]
    times = (np.arange(end) + 1) * float(data["dt"])
    for ax, values in zip(axes, [np.rad2deg(features[:, 4]), features[:, 5], features[:, 0]]):
        ax.plot(times, values, label=model, color=colors[model], lw=2)
axes[0].axhline(np.rad2deg(.8), color="#dc2626", ls="--", label="bad_orientation threshold")
axes[0].set_ylabel("Tilt (degrees)")
axes[1].axhline(.2, color="#dc2626", ls="--", label="base_height threshold")
axes[1].set_ylabel("Root height (m)")
axes[2].axhline(0, color="#475569", ls="--", label="Command vx = 0")
axes[2].set(xlabel="Simulated time from reset (s)", ylabel="Body vx (m/s)", xlim=(0, 3))
for ax in axes:
    ax.grid(alpha=.2)
    ax.legend(loc="best")
fig.suptitle("Standing: first episode of environment 0\nTerminal state included; lines end at auto-reset", fontsize=13)
fig.savefig(root / "first-episode-traces.png", dpi=170)
fig.savefig(root / "first-episode-traces.pdf")
plt.close(fig)
print(f"[PASS] Tables and figures saved in {root}")
