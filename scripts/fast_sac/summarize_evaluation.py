"""Measure command tracking and body orientation from a recorded evaluation."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with np.load(args.trajectory, allow_pickle=False) as recording:
        metadata = json.loads(str(recording["_metadata_json"]))
        required = ("root_pos", "root_quat_xyzw", "root_lin_vel", "root_ang_vel", "commanded_velocity")
        arrays = {name: recording[name].copy() for name in required}
        finite = {
            name: bool(np.isfinite(recording[name]).all())
            for name in recording.files
            if np.issubdtype(recording[name].dtype, np.number)
        }
    if not all(finite.values()):
        raise ValueError(f"Non-finite evaluation data: {[name for name, ok in finite.items() if not ok]}")
    steps = len(arrays["root_pos"])
    if steps == 0 or any(len(value) != steps for value in arrays.values()):
        raise ValueError("Evaluation channels must contain the same nonzero number of samples")
    rotation = Rotation.from_quat(arrays["root_quat_xyzw"])
    linear_body = rotation.inv().apply(arrays["root_lin_vel"])
    angular_body = rotation.inv().apply(arrays["root_ang_vel"])
    commands = arrays["commanded_velocity"][:, :3]
    measured = np.column_stack((linear_body[:, :2], angular_body[:, 2]))
    errors = measured - commands
    tilt = np.rad2deg(np.arccos(np.clip(rotation.as_matrix()[:, 2, 2], -1, 1)))
    report = {
        "trajectory": str(args.trajectory.resolve()),
        "steps": steps,
        "simulated_duration_s": steps * float(metadata["dt"]),
        "finite_channels": finite,
        "command_tracking_rmse": dict(zip(("vx_m_s", "vy_m_s", "yaw_rad_s"),
                                          np.sqrt(np.mean(errors ** 2, axis=0)).tolist(), strict=True)),
        "command_ranges": {name: [float(commands[:, i].min()), float(commands[:, i].max())]
                           for i, name in enumerate(("vx_m_s", "vy_m_s", "yaw_rad_s"))},
        "body_tilt_degrees": {"mean": float(tilt.mean()), "p95": float(np.percentile(tilt, 95)),
                              "max": float(tilt.max())},
        "fraction_of_samples_with_tilt_over_45_degrees": float(np.mean(tilt > 45)),
        "root_height_range_m": [float(arrays["root_pos"][:, 2].min()),
                                float(arrays["root_pos"][:, 2].max())],
        "scope": "One recorded environment, including initial transient and any resets; not a real-robot test.",
    }
    output = args.output or args.trajectory.with_name("evaluation_summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
