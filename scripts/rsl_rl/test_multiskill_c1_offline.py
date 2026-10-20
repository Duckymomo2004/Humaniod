"""CPU-only C1 tests. Fixtures are isolated and marked SYNTHETIC_TEST_DATA."""
from __future__ import annotations

import ast
import copy
import io
import json
import math
import os
import shutil
import subprocess
import sys
import unittest
import uuid
from pathlib import Path

from evaluate_multiskill_c1 import (
    SCHEMA, SYNTHETIC, EpisodeAccumulator, classify_termination,
    histogram_quantile, km_quantile, normalize_support_margin,
    read_json, sha256_file, summarize_episodes, support_state, survival,
    validate_checkpoints, validate_episode, write_json_new,
)
from summarize_multiskill_c1 import (
    expected_matrix, read_condition, summarize_matrix, write_summary,
)

ROOT = Path(__file__).resolve().parents[2]
AREA = ROOT / "run/stage9-pretraining/synthetic_tests" / uuid.uuid4().hex
NAMES = [f"SYNTHETIC_TEST_DATA_joint_{j}" for j in range(29)]


def setUpModule():
    AREA.mkdir(parents=True, exist_ok=False)
    (AREA / "SYNTHETIC_TEST_DATA").write_text(
        "CPU synthetic fixtures only. Not PhysX measurements or research results.\n")


def sample(vx=0.0):
    return {
        "root_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
        "roll_deg": 0.0, "pitch_deg": 0.0, "tilt_deg": 0.0,
        "root_height_m": .8, "lin_vel_b": [vx, 0.0, 0.0],
        "ang_vel_b": [0.0]*3, "command": [0.0]*3, "com_w": [0.0, 0.0, .7],
        "action": [0.0]*29, "q": [0.0]*29, "qd": [0.0]*29, "target": [0.0]*29,
        "computed_torque": [0.0]*29, "applied_torque_estimate": [0.0]*29,
        "effort_limits": [25.0]*29, "foot_fz_N": [150.0, 150.0],
        "support_state": "MISSING_TELEMETRY",
    }


def make_episode(duration=20.0, event="evaluation_horizon", env=0, index=0, vx=0.0):
    accumulator = EpisodeAccumulator(env, index, .02, NAMES, [0.0, 0.0, .7])
    s = sample(vx)
    for _ in range(round(duration/.02)):
        accumulator.add(s)
    return accumulator.finish({"event_type": event,
        "physical_reasons": ["bad_orientation"] if event == "physical_failure" else [],
        "timeout_reasons": ["time_out"] if event == "time_limit" else [],
        "time_limit_flag": event == "time_limit"})


def clone_matrix(source, name):
    target = AREA / name
    shutil.copytree(source, target)
    return target


class LogicTests(unittest.TestCase):
    def test_fall_before_two(self):
        r = make_episode(1, "physical_failure")
        self.assertEqual(survival([r], 2)["fraction"], 0)
        self.assertEqual(km_quantile([r], .5)["value_s"], 1)

    def test_fall_between_two_and_five(self):
        r = make_episode(3, "physical_failure")
        self.assertEqual(survival([r], 2)["fraction"], 1)
        self.assertEqual(survival([r], 5)["fraction"], 0)

    def test_fall_after_ten(self):
        r = make_episode(12, "physical_failure")
        self.assertEqual(survival([r], 10)["fraction"], 1)
        self.assertEqual(survival([r], 20)["fraction"], 0)

    def test_timeout_at_twenty(self):
        r = make_episode(20, "time_limit")
        self.assertEqual(survival([r], 20)["fraction"], 1)
        self.assertIsNone(r["physical_failure_time_s"])
        self.assertIsNone(km_quantile([r], .5)["value_s"])

    def test_alive_at_horizon(self):
        r = make_episode(20, "evaluation_horizon")
        self.assertEqual(survival([r], 20)["fraction"], 1)
        self.assertFalse(r["time_limit_flag"])

    def test_early_censor_retained_unknown(self):
        r = make_episode(3, "evaluation_horizon")
        result = survival([r], 5)
        self.assertIsNone(result["fraction"])
        self.assertEqual(result["unknown_due_to_early_censor"], 1)
        self.assertEqual((result["lower_bound"], result["upper_bound"]), (0, 1))

    def test_physical_exact_twenty_is_failure(self):
        r = make_episode(20, "physical_failure")
        self.assertEqual(survival([r], 20)["fraction"], 0)

    def test_physical_exact_two_is_failure(self):
        self.assertEqual(survival([make_episode(2, "physical_failure")], 2)["fraction"], 0)

    def test_step_float_tolerance(self):
        r = make_episode(2, "physical_failure")
        r["observed_duration_s"] += 1e-10
        r["physical_failure_time_s"] += 1e-10
        self.assertEqual(survival([r], 2)["fraction"], 0)

    def test_missing_termination_flag(self):
        with self.assertRaises(ValueError):
            classify_termination({}, ["time_out"], done=True, terminated=True, truncated=False)

    def test_inconsistent_done(self):
        with self.assertRaises(ValueError):
            classify_termination({"bad_orientation": False, "time_out": False},
                                 ["time_out"], done=True, terminated=False, truncated=False)

    def test_missing_boolean_flag(self):
        with self.assertRaises(ValueError):
            classify_termination({"bad_orientation": None, "time_out": False},
                                 ["time_out"], done=False, terminated=False, truncated=False)

    def test_other_physical_termination(self):
        result = classify_termination({"custom_collision": True, "time_out": False},
                                      ["time_out"], done=True, terminated=True, truncated=False)
        self.assertEqual(result["physical_reasons"], ["custom_collision"])

    def test_custom_timeout_cfg_not_name_guess(self):
        result = classify_termination({"episode_limit": True, "bad_orientation": False},
                                      ["episode_limit"], done=True, terminated=False, truncated=True)
        self.assertEqual(result["event_type"], "time_limit")

    def test_simultaneous_physical_and_timeout(self):
        result = classify_termination({"bad_orientation": True, "time_out": True},
                                      ["time_out"], done=True, terminated=True, truncated=True)
        self.assertEqual(result["event_type"], "physical_failure")
        self.assertTrue(result["time_limit_flag"])

    def test_nonfinite_sample(self):
        a = EpisodeAccumulator(0, 0, .02, NAMES, [0.0, 0.0, .7])
        s = sample()
        s["roll_deg"] = float("nan")
        with self.assertRaises(ValueError):
            a.add(s)

    def test_nonfinite_metric(self):
        r = make_episode(.02, "physical_failure")
        r["metrics"]["xy_error_sq_sum"] = float("inf")
        with self.assertRaises(ValueError):
            validate_episode(r)

    def test_missing_required_metric(self):
        r = make_episode(.02, "physical_failure")
        del r["metrics"]["root_height_sum"]
        with self.assertRaises(ValueError):
            summarize_episodes([r], 1, 20)

    def test_censor_cannot_fabricate_failure_time(self):
        r = make_episode(.02, "evaluation_horizon")
        r["physical_failure_time_s"] = .02
        with self.assertRaises(ValueError):
            validate_episode(r)

    def test_short_tracking_duration_warning(self):
        r = make_episode(.02, "physical_failure", vx=2.0)
        row = summarize_episodes([r], 1, 20)
        self.assertAlmostEqual(row["xy_first_rmse_mps"], 2)
        self.assertAlmostEqual(row["first_exposure_robot_s"], .02)
        self.assertIsNone(row["action_delta_first_rms_per_control_step"])
        self.assertTrue(any("SHORT_TRACKING" in x for x in row["data_validity_warnings"]))

    def test_time_weighted_tracking(self):
        a = make_episode(1, "physical_failure", env=0, vx=2)
        b = make_episode(3, "physical_failure", env=1, vx=0)
        row = summarize_episodes([a, b], 2, 20)
        self.assertAlmostEqual(row["xy_first_rmse_mps"], 1)
        self.assertEqual(row["first_exposure_robot_s"], 4)

    def test_no_action_delta_across_reset(self):
        r = make_episode(.02, "physical_failure")
        self.assertEqual(r["metrics"]["action_delta_count"], 0)

    def test_angles_upper_bin(self):
        self.assertEqual(histogram_quantile([{"1": 2, "101": 1}], .95), 10.1)

    def test_km_event_censor_tie(self):
        a = make_episode(1, "physical_failure", env=0)
        b = make_episode(1, "evaluation_horizon", env=1)
        self.assertEqual(km_quantile([a, b], .5)["value_s"], 1)

    def test_contact_fewer_than_three(self):
        r = support_state([[0, 0], [1, 0]], [1, 1], [.2, 0], fresh=True)
        self.assertEqual(r["state"], "INSUFFICIENT_POINTS")
        self.assertIsNotNone(r["cop_xy"])

    def test_degenerate_contact_hull(self):
        r = support_state([[0, 0], [1, 0], [2, 0]], [1, 1, 1], [.2, 0], fresh=True)
        self.assertEqual(r["state"], "DEGENERATE_HULL")

    def test_support_margin_nan_not_contact_loss(self):
        self.assertIsNone(normalize_support_margin(float("nan"), "DEGENERATE_HULL"))
        with self.assertRaises(ValueError):
            normalize_support_margin(float("nan"), "VALID_SUPPORT_POLYGON")

    def test_no_contact(self):
        r = support_state([], [], [0, 0], fresh=True)
        self.assertEqual(r["state"], "NO_CONTACT")
        self.assertIsNone(r["cop_xy"])

    def test_missing_contact(self):
        self.assertEqual(support_state()["state"], "MISSING_TELEMETRY")

    def test_stale_contact(self):
        self.assertEqual(support_state([[0, 0]], [1], [0, 0])["state"], "INVALID_MEASUREMENT")

    def test_contact_nan_invalid(self):
        r = support_state([[float("nan"), 0]], [1], [0, 0], fresh=True)
        self.assertEqual(r["state"], "INVALID_MEASUREMENT")

    def test_valid_hull_and_cop(self):
        r = support_state([[0, 0], [1, 0], [1, 1], [0, 1]], [1]*4, [.5, .5], fresh=True)
        self.assertEqual(r["state"], "VALID_SUPPORT_POLYGON")
        self.assertEqual(r["cop_xy"], [.5, .5])
        self.assertAlmostEqual(r["margin_m"], .5)

    def test_outside_valid_hull(self):
        r = support_state([[0, 0], [1, 0], [1, 1], [0, 1]], [1]*4, [2, .5], fresh=True)
        self.assertEqual(r["state"], "OUTSIDE_VALID_HULL")
        self.assertLess(r["margin_m"], 0)

    def test_checkpoint_missing_snapshots(self):
        historic = ROOT/"logs/rsl_rl/g1_multiskill_baseline/2026-10-19_08-46-28/model_199.pt"
        with self.assertRaises(ValueError):
            validate_checkpoints([historic])

    def test_checkpoint_genuine_stage5(self):
        cp = ROOT/"logs/rsl_rl/g1_multiskill_baseline/2026-10-20_17-14-10/model_1.pt"
        self.assertEqual(validate_checkpoints([cp]), [cp.resolve()])

    def test_checkpoint_duplicate_labels(self):
        base = ROOT/"logs/rsl_rl/g1_multiskill_baseline"
        with self.assertRaises(ValueError):
            validate_checkpoints([base/"2026-10-20_17-14-10/model_0.pt",
                                  base/"2026-10-20_17-12-03/model_0.pt"])

    def test_checkpoint_nonexistent(self):
        with self.assertRaises(FileNotFoundError):
            validate_checkpoints([AREA/"missing.pt"])

    def test_no_heavy_offline_import(self):
        code = ("import evaluate_multiskill_c1, summarize_multiskill_c1,sys;"
                "assert not any(n in sys.modules for n in "
                "['torch','isaaclab','isaacsim','omni','numpy','pxr'])")
        result = subprocess.run([sys.executable, "-B", "-c", code],
                                cwd=ROOT/"scripts/rsl_rl", capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_cli_help_no_runtime(self):
        for file in ("evaluate_multiskill_c1.py", "summarize_multiskill_c1.py"):
            result = subprocess.run([sys.executable, "-B", str(ROOT/"scripts/rsl_rl"/file), "--help"],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0)
            self.assertIn("--output", result.stdout)

    def test_runtime_has_no_learning_or_update_calls(self):
        tree = ast.parse((ROOT/"scripts/rsl_rl/evaluate_multiskill_c1.py").read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotIn(node.func.attr, {"learn", "backward", "optimizer_step"})
                if node.func.attr == "step":
                    self.assertEqual(ast.unparse(node.func.value), "wrapped")


    def test_observed_p10_not_failure_quantile(self):
        a = make_episode(2, "evaluation_horizon", env=0)
        b = make_episode(20, "time_limit", env=1)
        row = summarize_episodes([a, b], 2, 20)
        self.assertEqual(row["first_observed_duration_p10_s"], 2)
        self.assertIsNone(row["p10_physical_failure"]["value_s"])

    def test_distribution_includes_short_environment(self):
        a = make_episode(.02, "physical_failure", env=0, vx=4)
        b = make_episode(20, "time_limit", env=1, vx=0)
        row = summarize_episodes([a, b], 2, 20)
        self.assertEqual(row["xy_environment_rmse_distribution"]["p95"], 4)
        self.assertEqual(len(row["per_environment_first_metrics"]), 2)

    def test_missing_velocity_metric_rejected(self):
        r = make_episode(.02, "physical_failure")
        del r["metrics"]["root_linear_velocity_sq_sum"]
        with self.assertRaises(ValueError):
            validate_episode(r)

    def test_unknown_support_state_rejected(self):
        r = make_episode(.02, "physical_failure")
        r["metrics"]["support_states"] = {"NaN_MEANS_FALL": 1}
        with self.assertRaises(ValueError):
            validate_episode(r)

    def test_inconsistent_joint_order_rejected(self):
        a = make_episode(.02, "physical_failure", env=0)
        b = make_episode(.02, "physical_failure", env=1)
        b["metrics"]["joint_names"].reverse()
        with self.assertRaises(ValueError):
            summarize_episodes([a, b], 2, 20)

    def test_mixed_timeout_flag_record(self):
        r = make_episode(20, "physical_failure")
        r["time_limit_flag"] = True
        r["timeout_reasons"] = ["time_out"]
        row = summarize_episodes([r], 1, 20)
        self.assertEqual(row["physical_failures"], 1)
        self.assertEqual(row["timeouts"], 0)
        self.assertEqual(row["time_limit_flags"], 1)
        self.assertEqual(row["S20"]["fraction"], 0)

    def test_invalid_episode_index(self):
        r = make_episode(.02, "physical_failure")
        r["episode_index"] = 1
        with self.assertRaises(ValueError):
            validate_episode(r)


    def test_incoming_not_confused_with_pd_estimate(self):
        a = EpisodeAccumulator(0, 0, .02, NAMES, [0.0, 0.0, .7])
        s = sample()
        s["computed_torque"] = [20.0]*29
        s["applied_torque_estimate"] = [20.0]*29
        s["projected_incoming_force"] = [1000.0]*29
        a.add(s)
        r = a.finish({"event_type": "physical_failure",
                      "physical_reasons": ["bad_orientation"],
                      "timeout_reasons": [], "time_limit_flag": False})
        summary = summarize_episodes([r], 1, 20)
        self.assertEqual(summary["pd_applied_estimate_first_peak_Nm"], 20)
        self.assertEqual(summary["projected_incoming_first_peak_Nm"], 1000)
        self.assertEqual(summary["estimated_clipped_joint_control_samples"], 0)

    def test_evaluator_existing_output_rejected_before_import(self):
        from evaluate_multiskill_c1 import main
        with self.assertRaises(FileExistsError):
            main(["--checkpoints", "no_checkpoint.pt", "--output", str(AREA)])
        self.assertNotIn("isaaclab", sys.modules)


    def test_lfs_pointer_checkpoint_rejected(self):
        pointer = ROOT/"logs/rsl_rl/g1_datn/2026-09-17_03-02-55/model_0.pt"
        with self.assertRaisesRegex(ValueError, "LFS pointer"):
            validate_checkpoints([pointer])


class MatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matrix = AREA/"valid_matrix"
        cls.matrix.mkdir()
        first = make_episode(3, "physical_failure", env=0)
        later = make_episode(17, "evaluation_horizon", env=0, index=1)
        timeout = make_episode(20, "time_limit", env=1)
        for checkpoint, scenario, seed in sorted(expected_matrix()):
            folder = cls.matrix/f"{checkpoint}-{scenario}-{seed}"
            folder.mkdir()
            with (folder/"episodes.jsonl").open("x") as handle:
                for record in [first, later, timeout]:
                    handle.write(json.dumps(record, allow_nan=False)+"\n")
            write_json_new(folder/"condition.json", {
                "schema": SCHEMA, "dataset_kind": SYNTHETIC, "status": "COMPLETE",
                "checkpoint": checkpoint, "scenario": scenario, "seed": seed,
                "num_envs": 2, "horizon_s": 20.0, "dt": .02,
                "episodes_path": "episodes.jsonl",
                "episodes_sha256": sha256_file(folder/"episodes.jsonl"),
                "snapshot_sha256": {"env": "SYNTHETIC_TEST_DATA", "agent": "SYNTHETIC_TEST_DATA"},
                "checkpoint_sha256": "SYNTHETIC_TEST_DATA_NO_CHECKPOINT_BINARY",
                "initial_fingerprints": {"q": "SYNTHETIC_TEST_DATA"}})
        (cls.matrix/"SYNTHETIC_TEST_DATA").write_text("Synthetic CPU fixtures, not research evaluation.\n")

    def test_valid_thirty_condition_matrix(self):
        result = summarize_matrix(self.matrix, num_envs=2, allow_synthetic=True)
        self.assertEqual(result["condition_count"], 30)
        self.assertEqual(result["dataset_kind"], SYNTHETIC)
        self.assertTrue(all(r["S20"]["fraction"] == .5 for r in result["rows"]))
        self.assertTrue(all(r["timeouts"] == 1 for r in result["rows"]))
        self.assertTrue(all(r["censored_first_episodes"] == 1 for r in result["rows"]))

    def test_synthetic_rejected_by_default(self):
        with self.assertRaises(ValueError):
            summarize_matrix(self.matrix, num_envs=2)

    def test_incomplete_seed_scenario_checkpoint_matrix(self):
        folder = clone_matrix(self.matrix, "missing_condition")
        next(folder.rglob("condition.json")).unlink()  # Only a newly created synthetic copy.
        with self.assertRaisesRegex(ValueError, "missing="):
            summarize_matrix(folder, num_envs=2, allow_synthetic=True)

    def test_matrix_wrong_horizon(self):
        with self.assertRaises(ValueError):
            summarize_matrix(self.matrix, num_envs=2, horizon_s=10, allow_synthetic=True)

    def test_duplicate_condition(self):
        folder = clone_matrix(self.matrix, "duplicate_condition")
        source = next(folder.rglob("condition.json")).parent
        shutil.copytree(source, folder/"duplicate")
        with self.assertRaises(ValueError):
            summarize_matrix(folder, num_envs=2, allow_synthetic=True)

    def test_output_does_not_overwrite(self):
        result = summarize_matrix(self.matrix, num_envs=2, allow_synthetic=True)
        output = AREA/"summary"
        write_summary(result, output)
        before = sha256_file(output/"summary.json")
        with self.assertRaises(FileExistsError):
            write_summary(result, output)
        self.assertEqual(sha256_file(output/"summary.json"), before)

    def test_corrupt_record_hash(self):
        folder = clone_matrix(self.matrix, "corrupt_stream")
        stream = next(folder.rglob("episodes.jsonl"))
        with stream.open("a") as handle:
            handle.write("{}\n")
        with self.assertRaises(ValueError):
            read_condition(stream.parent/"condition.json", allow_synthetic=True)

    def test_missing_episode_exposure(self):
        folder = clone_matrix(self.matrix, "missing_exposure")
        condition_path = next(folder.rglob("condition.json"))
        condition = read_json(condition_path)
        stream = condition_path.parent/"episodes.jsonl"
        lines = stream.read_text().splitlines()
        stream.write_text("\n".join([lines[0], lines[2]])+"\n")
        condition["episodes_sha256"] = sha256_file(stream)
        condition_path.write_text(json.dumps(condition))
        with self.assertRaises(ValueError):
            read_condition(condition_path, allow_synthetic=True)

    def test_parent_path_traversal(self):
        folder = clone_matrix(self.matrix, "path_traversal")
        condition_path = next(folder.rglob("condition.json"))
        condition = read_json(condition_path)
        condition["episodes_path"] = "../escaped.jsonl"
        condition_path.write_text(json.dumps(condition))
        with self.assertRaises(ValueError):
            read_condition(condition_path, allow_synthetic=True)

    def test_wrong_schema(self):
        folder = clone_matrix(self.matrix, "wrong_schema")
        condition_path = next(folder.rglob("condition.json"))
        condition = read_json(condition_path)
        condition["schema"] = "bad"
        condition_path.write_text(json.dumps(condition))
        with self.assertRaises(ValueError):
            read_condition(condition_path, allow_synthetic=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
