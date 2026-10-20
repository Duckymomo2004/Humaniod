"""C1 inference evaluator and CPU-only statistics.

Heavy simulator imports are lazy inside run_runtime(). Importing this module or
using --help never starts Kit. Runtime integration requires a separate PhysX
smoke approval; Stage 9 tests validate only the pure statistics/schema.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

SCHEMA = "g1-c1-evaluation-v1"
REAL = "REAL_EVALUATION"
SYNTHETIC = "SYNTHETIC_TEST_DATA"
SUPPORT_STATES = {"MISSING_TELEMETRY", "INVALID_MEASUREMENT", "NO_CONTACT",
                  "INSUFFICIENT_POINTS", "DEGENERATE_HULL", "VALID_SUPPORT_POLYGON",
                  "OUTSIDE_VALID_HULL"}
SCENARIOS = {"standing": (0.0, 0.0, 0.0),
             "forward_slow": (0.1, 0.0, 0.0), "training_commands": None}
EPS = 1e-8


def finite_tree(value):
    """Reject unexpected nonfinite values. Undefined metrics must use None."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite observation/metric")
    if isinstance(value, dict):
        for child in value.values():
            finite_tree(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            finite_tree(child)


def write_json_new(path, data):
    finite_tree(data)
    with Path(path).open("x") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)


def read_json(path):
    def reject(value):
        raise ValueError(f"Invalid JSON number {value}")
    def unique(pairs):
        result = {}
        for k, v in pairs:
            if k in result:
                raise ValueError(f"Duplicate JSON key {k}")
            result[k] = v
        return result
    return json.loads(Path(path).read_text(), parse_constant=reject,
                      object_pairs_hook=unique)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_checkpoints(paths):
    """Resolve real same-run snapshots; reject LFS pointers and label collisions."""
    result, labels = [], set()
    for text in paths:
        path = Path(text).resolve(strict=True)
        if not path.is_file() or path.stem in labels:
            raise ValueError("Checkpoint is not a file or duplicate output label")
        with path.open("rb") as handle:
            if handle.read(128).startswith(b"version https://git-lfs"):
                raise ValueError("Checkpoint is a Git LFS pointer")
        if path.stat().st_size < 1024:
            raise ValueError("Checkpoint too small")
        for name in ("agent.yaml", "env.yaml"):
            snapshot = path.parent / "params" / name
            if not snapshot.is_file() or snapshot.stat().st_size == 0:
                raise ValueError(f"Missing genuine same-run snapshot: {snapshot}")
        labels.add(path.stem)
        result.append(path)
    if not result:
        raise ValueError("At least one checkpoint required")
    if len({p.parent for p in result}) != 1:
        raise ValueError("C1 candidates must belong to one genuine training run")
    return result


def classify_termination(flags, timeout_terms, *, done, terminated, truncated):
    if not flags or not set(timeout_terms) <= set(flags):
        raise ValueError("Missing termination flags/schema")
    if any(type(v) is not bool for v in [*flags.values(), done, terminated, truncated]):
        raise ValueError("Termination flags must be booleans")
    physical = [k for k, v in flags.items() if v and k not in timeout_terms]
    timeouts = [k for k, v in flags.items() if v and k in timeout_terms]
    if terminated != bool(physical) or truncated != bool(timeouts):
        raise ValueError("Manager aggregates disagree with individual term flags")
    if done != (terminated or truncated):
        raise ValueError("Done disagrees with terminated/time_outs")
    return {"event_type": "physical_failure" if physical else
            "time_limit" if timeouts else "ongoing",
            "physical_reasons": physical, "timeout_reasons": timeouts,
            "time_limit_flag": truncated}


def validate_episode(record):
    if record.get("schema") != SCHEMA:
        raise ValueError("Unknown episode schema")
    finite_tree(record)
    kind = record.get("event_type")
    if kind not in {"physical_failure", "time_limit", "evaluation_horizon"}:
        raise ValueError("Invalid episode end classification")
    duration = record.get("observed_duration_s")
    if not isinstance(duration, (float, int)) or isinstance(duration, bool) or duration <= 0:
        raise ValueError("Invalid observed duration")
    if type(record.get("samples")) is not int or record["samples"] < 1:
        raise ValueError("Missing samples")
    dt = record.get("dt")
    if not isinstance(dt, (float, int)) or dt <= 0 or abs(duration - record["samples"] * dt) > EPS:
        raise ValueError("Duration/sample/control-dt mismatch")
    if type(record.get("first_episode")) is not bool:
        raise ValueError("Missing first-episode marker")
    if type(record.get("env_id")) is not int or record["env_id"] < 0:
        raise ValueError("Invalid env_id")
    if type(record.get("episode_index")) is not int or record["episode_index"] < 0:
        raise ValueError("Invalid episode index")
    if record["first_episode"] != (record["episode_index"] == 0):
        raise ValueError("First marker/index mismatch")
    reasons = record.get("physical_reasons")
    if not isinstance(reasons, list):
        raise ValueError("Missing physical reason list")
    failure = record.get("physical_failure_time_s")
    if kind == "physical_failure":
        if not reasons or failure is None or abs(failure - duration) > EPS:
            raise ValueError("Physical failure must have event time and reason")
    elif failure is not None or reasons:
        raise ValueError("Censoring must not fabricate physical failure")
    timeout_reasons = record.get("timeout_reasons")
    if (not isinstance(timeout_reasons, list) or
            any(not isinstance(x, str) or not x for x in reasons + timeout_reasons) or
            len(set(reasons)) != len(reasons) or len(set(timeout_reasons)) != len(timeout_reasons) or
            set(reasons) & set(timeout_reasons)):
        raise ValueError("Invalid/overlapping termination reason lists")
    if bool(timeout_reasons) != record.get("time_limit_flag"):
        raise ValueError("Timeout reason/flag mismatch")
    if type(record.get("time_limit_flag")) is not bool:
        raise ValueError("Missing timeout flag")
    if kind == "time_limit" and not record["time_limit_flag"]:
        raise ValueError("Timeout classification without timeout flag")
    if kind == "evaluation_horizon" and record["time_limit_flag"]:
        raise ValueError("Horizon censor cannot contain a timeout")
    metrics = record.get("metrics")
    required = {"xy_error_sq_sum", "yaw_error_sq_sum", "tilt_hist",
                "roll_abs_hist", "pitch_abs_hist", "root_height_sum",
                "root_height_min", "root_height_max", "action_sq_sum",
                "action_count", "action_delta_sq_sum", "action_delta_count",
                "joint_error_sq_sum", "joint_error_peak", "joint_velocity_sq_sum",
                "joint_count", "pd_unclipped_peak", "pd_applied_estimate_peak",
                "estimated_clipped_samples", "joint_names", "com_drift_max_m",
                "contact_asymmetry_sum", "contact_valid_samples", "support_states",
                "root_linear_velocity_sq_sum", "root_angular_velocity_sq_sum",
                "foot_load_samples", "first_foot_load_s"}
    if not isinstance(metrics, dict) or not required <= set(metrics):
        raise ValueError("Missing required metric fields")
    for k in ("root_height_sum", "root_height_min", "root_height_max",
              "pd_unclipped_peak", "pd_applied_estimate_peak", "com_drift_max_m",
              "root_linear_velocity_sq_sum", "root_angular_velocity_sq_sum",
              "contact_asymmetry_sum", "contact_valid_samples"):
        if type(metrics[k]) not in (float, int):
            raise ValueError(f"Invalid numeric metric {k}")
    if metrics["root_height_min"] > metrics["root_height_max"]:
        raise ValueError("Root height range invalid")
    if not 0 <= metrics["contact_valid_samples"] <= record["samples"]:
        raise ValueError("Invalid contact duration")
    if not 0 <= metrics["contact_asymmetry_sum"] <= metrics["contact_valid_samples"] + EPS:
        raise ValueError("Invalid force asymmetry")
    for k in ("xy_error_sq_sum", "yaw_error_sq_sum", "action_sq_sum",
              "action_count", "action_delta_sq_sum", "action_delta_count",
              "joint_velocity_sq_sum", "joint_count", "estimated_clipped_samples"):
        if type(metrics[k]) not in (float, int) or metrics[k] < 0:
            raise ValueError(f"Invalid/nonnegative metric {k}")
    for key in ("tilt_hist", "roll_abs_hist", "pitch_abs_hist"):
        hist = metrics[key]
        if (not isinstance(hist, dict) or sum(hist.values()) != record["samples"]
                or any(not str(k).isdigit() or not 0 <= int(k) <= 1800 or
                       type(v) is not int or v <= 0 for k, v in hist.items())):
            raise ValueError("Invalid angle histogram")
    names = metrics["joint_names"]
    if len(names) != 29 or len(set(names)) != 29:
        raise ValueError("Expected 29 unique joints")
    for key in ("joint_error_sq_sum", "joint_error_peak"):
        if len(metrics[key]) != 29 or any(type(x) not in (int, float) or x < 0 for x in metrics[key]):
            raise ValueError(f"Invalid joint metric {key}")
    if metrics["joint_count"] != record["samples"] * 29:
        raise ValueError("Joint sample exposure mismatch")
    if metrics["action_count"] != record["samples"] * 29:
        raise ValueError("Action sample exposure mismatch")
    if (not set(metrics["support_states"]) <= SUPPORT_STATES or
            any(type(v) is not int or v < 0 for v in metrics["support_states"].values()) or
            sum(metrics["support_states"].values()) != record["samples"]):
        raise ValueError("Missing support validity accounting")


def survival(records, threshold):
    if threshold <= 0:
        raise ValueError("Positive survival threshold required")
    success = failure = unknown = 0
    for row in records:
        validate_episode(row)
        duration = row["observed_duration_s"]
        if row["event_type"] == "physical_failure":
            if row["physical_failure_time_s"] <= threshold + EPS:
                failure += 1
            else:
                success += 1
        elif duration + EPS >= threshold:
            success += 1
        else:
            unknown += 1
    n = len(records)
    if not n:
        raise ValueError("Empty survival cohort")
    return {"fraction": success / n if not unknown else None, "n": n,
            "known_successes": success, "physical_failures": failure,
            "unknown_due_to_early_censor": unknown,
            "lower_bound": success / n, "upper_bound": (success + unknown) / n}


def km_quantile(records, probability):
    if not records or not 0 < probability < 1:
        raise ValueError("Invalid survival quantile")
    grouped = {}
    for row in records:
        validate_episode(row)
        time = round(row["observed_duration_s"], 9)
        counts = grouped.setdefault(time, [0, 0])
        counts[0 if row["event_type"] == "physical_failure" else 1] += 1
    risk, s = len(records), 1.0
    for time, (events, censored) in sorted(grouped.items()):
        s *= 1 - events / risk  # Events before censoring at a shared time.
        if 1 - s + EPS >= probability:
            return {"value_s": time, "status": "REACHED",
                    "followup_max_s": max(grouped)}
        risk -= events + censored
    return {"value_s": None, "status": "NOT_REACHED",
            "followup_max_s": max(grouped),
            "assumption": "Non-informative right censoring; not an exact failure time"}


def histogram_quantile(histograms, probability=.95):
    counts = {}
    for hist in histograms:
        for k, v in hist.items():
            counts[int(k)] = counts.get(int(k), 0) + v
    if not counts:
        return None
    rank, total = math.ceil(probability * sum(counts.values())), 0
    for bucket, count in sorted(counts.items()):
        total += count
        if total >= rank:
            return bucket / 10  # Conservative upper edge, resolution 0.1 degree.


def support_state(points=None, forces=None, com_xy=None, *, fresh=False):
    """Pure quality classifier; runtime intentionally does not enable raw contacts."""
    if points is None or forces is None or com_xy is None:
        return {"state": "MISSING_TELEMETRY", "margin_m": None, "cop_xy": None}
    if not fresh:
        return {"state": "INVALID_MEASUREMENT", "margin_m": None, "cop_xy": None}
    if len(points) != len(forces) or len(com_xy) != 2:
        raise ValueError("Invalid support array shape")
    if any(len(p) != 2 for p in points):
        raise ValueError("Expected planar contact coordinates")
    try:
        finite_tree([points, forces, com_xy])
    except ValueError:
        return {"state": "INVALID_MEASUREMENT", "margin_m": None, "cop_xy": None}
    if any(f < 0 for f in forces):
        return {"state": "INVALID_MEASUREMENT", "margin_m": None, "cop_xy": None}
    loaded = [(tuple(map(float, p)), float(f)) for p, f in zip(points, forces) if f > 1e-4]
    if not loaded:
        return {"state": "NO_CONTACT", "margin_m": None, "cop_xy": None}
    weight = sum(f for _, f in loaded)
    cop = [sum(p[a] * f for p, f in loaded) / weight for a in range(2)]
    unique = sorted(set(p for p, _ in loaded))
    if len(unique) < 3:
        return {"state": "INSUFFICIENT_POINTS", "margin_m": None, "cop_xy": cop}
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower, upper = [], []
    for p in unique:
        while len(lower) > 1 and cross(lower[-2], lower[-1], p) <= 1e-12:
            lower.pop()
        lower.append(p)
    for p in reversed(unique):
        while len(upper) > 1 and cross(upper[-2], upper[-1], p) <= 1e-12:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    area = abs(sum(p[0]*q[1]-p[1]*q[0] for p, q in zip(hull, hull[1:]+hull[:1]))) / 2
    if len(hull) < 3 or area <= 1e-8:
        return {"state": "DEGENERATE_HULL", "margin_m": None, "cop_xy": cop}
    margin = min(((q[0]-p[0])*(com_xy[1]-p[1]) -
                  (q[1]-p[1])*(com_xy[0]-p[0])) / math.dist(p, q)
                 for p, q in zip(hull, hull[1:]+hull[:1]))
    return {"state": "OUTSIDE_VALID_HULL" if margin < -1e-5 else "VALID_SUPPORT_POLYGON",
            "margin_m": margin, "area_m2": area, "cop_xy": cop}


def normalize_support_margin(margin, state):
    if state not in {"VALID_SUPPORT_POLYGON", "OUTSIDE_VALID_HULL"}:
        return None
    if margin is None or not math.isfinite(margin):
        raise ValueError("A valid hull cannot have undefined support margin")
    return float(margin)


class EpisodeAccumulator:
    """Online sufficient statistics; sparse 0.1-degree angle histograms."""
    def __init__(self, env_id, episode_index, dt, joint_names, initial_com):
        if dt <= 0 or len(joint_names) != 29 or len(set(joint_names)) != 29:
            raise ValueError("Invalid dt/joint schema")
        finite_tree(initial_com)
        self.env_id, self.index, self.dt = env_id, episode_index, dt
        self.initial_com = list(initial_com)
        self.samples, self.previous_action = 0, None
        self.m = {"xy_error_sq_sum": 0.0, "yaw_error_sq_sum": 0.0,
                  "tilt_hist": {}, "roll_abs_hist": {}, "pitch_abs_hist": {},
                  "root_height_sum": 0.0, "root_height_min": None, "root_height_max": None,
                  "root_linear_velocity_sq_sum": 0.0, "root_angular_velocity_sq_sum": 0.0,
                  "action_sq_sum": 0.0, "action_count": 0, "action_delta_sq_sum": 0.0,
                  "action_delta_count": 0, "joint_error_sq_sum": [0.0]*29,
                  "joint_error_peak": [0.0]*29, "joint_velocity_sq_sum": 0.0,
                  "joint_count": 0, "joint_names": list(joint_names),
                  "pd_unclipped_peak": 0.0, "pd_applied_estimate_peak": 0.0,
                  "projected_incoming_peak_Nm": None,
                  "estimated_clipped_samples": 0, "com_drift_max_m": 0.0,
                  "contact_asymmetry_sum": 0.0, "contact_valid_samples": 0,
                  "foot_load_samples": [0, 0], "first_foot_load_s": [None, None],
                  "support_states": {}}

    def add(self, sample):
        finite_tree(sample)
        for k in ("action", "q", "qd", "target", "computed_torque",
                  "applied_torque_estimate", "effort_limits"):
            if len(sample[k]) != 29:
                raise ValueError(f"Expected 29 values: {k}")
        for k in ("lin_vel_b", "ang_vel_b", "command", "com_w"):
            if len(sample[k]) != 3:
                raise ValueError(f"Expected 3 values: {k}")
        if any(x <= 0 for x in sample["effort_limits"]):
            raise ValueError("Positive effort limits required")
        angles = [abs(sample["roll_deg"]), abs(sample["pitch_deg"]), sample["tilt_deg"]]
        if any(not 0 <= x <= 180 + EPS for x in angles):
            raise ValueError("Invalid orientation angles")
        self.samples += 1
        for key, value in zip(("roll_abs_hist", "pitch_abs_hist", "tilt_hist"), angles):
            bucket = str(min(1800, math.ceil(value*10-1e-10)))
            self.m[key][bucket] = self.m[key].get(bucket, 0) + 1
        self.m["xy_error_sq_sum"] += sum((sample["lin_vel_b"][j]-sample["command"][j])**2 for j in (0, 1))
        self.m["yaw_error_sq_sum"] += (sample["ang_vel_b"][2]-sample["command"][2])**2
        self.m["root_linear_velocity_sq_sum"] += sum(x*x for x in sample["lin_vel_b"])
        self.m["root_angular_velocity_sq_sum"] += sum(x*x for x in sample["ang_vel_b"])
        height = sample["root_height_m"]
        self.m["root_height_sum"] += height
        self.m["root_height_min"] = height if self.m["root_height_min"] is None else min(height, self.m["root_height_min"])
        self.m["root_height_max"] = height if self.m["root_height_max"] is None else max(height, self.m["root_height_max"])
        action = sample["action"]
        self.m["action_sq_sum"] += sum(x*x for x in action)
        self.m["action_count"] += 29
        if self.previous_action is not None:
            self.m["action_delta_sq_sum"] += sum((a-b)**2 for a, b in zip(action, self.previous_action))
            self.m["action_delta_count"] += 29
        self.previous_action = list(action)
        for j, (target, q) in enumerate(zip(sample["target"], sample["q"])):
            error = target-q
            self.m["joint_error_sq_sum"][j] += error*error
            self.m["joint_error_peak"][j] = max(self.m["joint_error_peak"][j], abs(error))
        self.m["joint_velocity_sq_sum"] += sum(x*x for x in sample["qd"])
        self.m["joint_count"] += 29
        self.m["pd_unclipped_peak"] = max(self.m["pd_unclipped_peak"], max(map(abs, sample["computed_torque"])))
        self.m["pd_applied_estimate_peak"] = max(self.m["pd_applied_estimate_peak"], max(map(abs, sample["applied_torque_estimate"])))
        incoming = sample.get("projected_incoming_force")
        if incoming is not None:
            if len(incoming) != 29:
                raise ValueError("Expected 29 projected reaction values")
            self.m["projected_incoming_peak_Nm"] = max(
                self.m["projected_incoming_peak_Nm"] or 0.0, max(map(abs, incoming)))
        self.m["estimated_clipped_samples"] += sum(abs(t) > lim+1e-4 for t, lim in zip(sample["computed_torque"], sample["effort_limits"]))
        self.m["com_drift_max_m"] = max(self.m["com_drift_max_m"], math.dist(sample["com_w"], self.initial_com))
        fz = sample.get("foot_fz_N")
        if fz is not None:
            if len(fz) != 2:
                raise ValueError("Expected left/right foot forces")
            load = [max(0, x) for x in fz]
            if sum(load) > 1e-4:
                self.m["contact_asymmetry_sum"] += abs(load[0]-load[1])/sum(load)
                self.m["contact_valid_samples"] += 1
            for foot in (0, 1):
                if load[foot] > 1e-4:
                    self.m["foot_load_samples"][foot] += 1
                    if self.m["first_foot_load_s"][foot] is None:
                        self.m["first_foot_load_s"][foot] = self.samples*self.dt
        state = sample.get("support_state", "MISSING_TELEMETRY")
        if state not in SUPPORT_STATES:
            raise ValueError("Unknown support validity state")
        self.m["support_states"][state] = self.m["support_states"].get(state, 0) + 1

    def finish(self, classification):
        if not self.samples:
            raise ValueError("Cannot emit a zero-duration episode")
        kind = classification["event_type"]
        if kind not in {"physical_failure", "time_limit", "evaluation_horizon"}:
            raise ValueError("Episode is not complete/censored")
        duration = self.samples*self.dt
        record = {"schema": SCHEMA, "env_id": self.env_id, "episode_index": self.index,
                  "first_episode": self.index == 0, "samples": self.samples, "dt": self.dt,
                  "observed_duration_s": duration, "event_type": kind,
                  "physical_failure_time_s": duration if kind == "physical_failure" else None,
                  "physical_reasons": classification["physical_reasons"],
                  "timeout_reasons": classification.get("timeout_reasons", []),
                  "time_limit_flag": classification["time_limit_flag"], "metrics": self.m}
        validate_episode(record)
        return record


def empirical_quantile(values, probability):
    if not values or not 0 < probability <= 1:
        raise ValueError("Invalid empirical quantile")
    return sorted(values)[max(0, math.ceil(len(values)*probability)-1)]


def summarize_episodes(records, num_envs, horizon_s):
    first = [r for r in records if r.get("first_episode")]
    if len(first) != num_envs or {r["env_id"] for r in first} != set(range(num_envs)):
        raise ValueError("Incomplete/duplicate first-episode cohort")
    for record in records:
        validate_episode(record)
        if record["observed_duration_s"] > horizon_s + EPS:
            raise ValueError("Episode exceeds evaluation horizon")
    if any(r["metrics"]["joint_names"] != first[0]["metrics"]["joint_names"] for r in first):
        raise ValueError("Joint ordering differs within the condition")
    metric = lambda r: r["metrics"]
    n_samples = sum(r["samples"] for r in first)
    action_n = sum(metric(r)["action_count"] for r in first)
    delta_n = sum(metric(r)["action_delta_count"] for r in first)
    joint_n = sum(metric(r)["joint_count"] for r in first)
    warning = ["RUNTIME_VERIFICATION_REQUIRED", "SUPPORT_MARGIN_NOT_PRIMARY",
               "ANGLE_QUANTILES_UPPER_BIN_0.1_DEG"]
    if any(r["observed_duration_s"] < 2 for r in first):
        warning.append("SHORT_TRACKING_EXPOSURE: interpret RMSE alongside survival")
    warnings_support = {}
    for r in first:
        for state, count in metric(r)["support_states"].items():
            warnings_support[state] = warnings_support.get(state, 0) + count
    if warnings_support.get("MISSING_TELEMETRY"):
        warning.append("RAW_CONTACT_HULL_MISSING")
    result = {"schema": SCHEMA, "num_envs": num_envs, "horizon_s": horizon_s,
              "physical_failures": sum(r["event_type"] == "physical_failure" for r in first),
              "timeouts": sum(r["event_type"] == "time_limit" for r in first),
              "time_limit_flags": sum(r["time_limit_flag"] for r in first),
              "evaluation_horizon_censors": sum(r["event_type"] == "evaluation_horizon" for r in first),
              "censored_first_episodes": sum(r["event_type"] != "physical_failure" for r in first),
              "bad_orientation_first_fraction": sum("bad_orientation" in r["physical_reasons"] for r in first)/num_envs,
              "first_observed_duration_mean_s": statistics.mean(r["observed_duration_s"] for r in first),
              "first_observed_duration_median_s": statistics.median(r["observed_duration_s"] for r in first),
              "first_observed_duration_p10_s": empirical_quantile([r["observed_duration_s"] for r in first], .1),
              "observed_durations_are_not_failure_times": True,
              "median_physical_failure": km_quantile(first, .5),
              "p10_physical_failure": km_quantile(first, .1),
              "first_exposure_robot_s": n_samples*first[0]["dt"],
              "tracking_weighting": "time/sample weighted across first episodes; each environment reported separately",
              "xy_first_rmse_mps": math.sqrt(sum(metric(r)["xy_error_sq_sum"] for r in first)/n_samples),
              "yaw_first_rmse_radps": math.sqrt(sum(metric(r)["yaw_error_sq_sum"] for r in first)/n_samples),
              "tilt_first_p95_deg_upper": histogram_quantile(metric(r)["tilt_hist"] for r in first),
              "roll_abs_first_p95_deg_upper": histogram_quantile(metric(r)["roll_abs_hist"] for r in first),
              "pitch_abs_first_p95_deg_upper": histogram_quantile(metric(r)["pitch_abs_hist"] for r in first),
              "root_height_first_mean_m": sum(metric(r)["root_height_sum"] for r in first)/n_samples,
              "root_height_first_min_m": min(metric(r)["root_height_min"] for r in first),
              "com_drift_first_max_m": max(metric(r)["com_drift_max_m"] for r in first),
              "action_first_rms": math.sqrt(sum(metric(r)["action_sq_sum"] for r in first)/action_n),
              "action_delta_first_rms_per_control_step": math.sqrt(sum(metric(r)["action_delta_sq_sum"] for r in first)/delta_n) if delta_n else None,
              "joint_tracking_first_rms_rad": math.sqrt(sum(sum(metric(r)["joint_error_sq_sum"]) for r in first)/joint_n),
              "joint_velocity_first_rms_radps": math.sqrt(sum(metric(r)["joint_velocity_sq_sum"] for r in first)/joint_n),
              "pd_unclipped_first_peak_Nm": max(metric(r)["pd_unclipped_peak"] for r in first),
              "pd_applied_estimate_first_peak_Nm": max(metric(r)["pd_applied_estimate_peak"] for r in first),
              "estimated_clipped_joint_control_samples": sum(metric(r)["estimated_clipped_samples"] for r in first),
              "support_state_counts": warnings_support, "data_validity_warnings": warning,
              "all_episode_records": len(records),
              "later_episode_records": len(records)-len(first),
              "all_physical_failures": sum(r["event_type"] == "physical_failure" for r in records),
              "all_timeouts": sum(r["event_type"] == "time_limit" for r in records)}
    result["per_environment_first_metrics"] = [
        {"env_id": r["env_id"], "observed_duration_s": r["observed_duration_s"],
         "event_type": r["event_type"],
         "xy_rmse_mps": math.sqrt(metric(r)["xy_error_sq_sum"]/r["samples"]),
         "yaw_rmse_radps": math.sqrt(metric(r)["yaw_error_sq_sum"]/r["samples"]),
         "tilt_p95_deg_upper": histogram_quantile([metric(r)["tilt_hist"]]),
         "roll_abs_p95_deg_upper": histogram_quantile([metric(r)["roll_abs_hist"]]),
         "pitch_abs_p95_deg_upper": histogram_quantile([metric(r)["pitch_abs_hist"]])}
        for r in sorted(first, key=lambda x: x["env_id"])]
    result["tracking_distribution_weighting"] = "Quantiles over environment RMSEs: equal environment weight, durations retained"
    for value, unit in (("xy_rmse_mps", "xy"), ("yaw_rmse_radps", "yaw")):
        values = [r[value] for r in result["per_environment_first_metrics"]]
        result[f"{unit}_environment_rmse_distribution"] = {
            "p10": empirical_quantile(values, .1), "median": statistics.median(values),
            "p95": empirical_quantile(values, .95)}
    result["joint_tracking_first_by_joint"] = [
        {"joint": metric(first[0])["joint_names"][j],
         "rms_error_rad": math.sqrt(sum(metric(r)["joint_error_sq_sum"][j] for r in first)/n_samples),
         "peak_error_rad": max(metric(r)["joint_error_peak"][j] for r in first)}
        for j in range(29)]
    incoming = [metric(r).get("projected_incoming_peak_Nm") for r in first]
    result["projected_incoming_first_peak_Nm"] = (
        max(v for v in incoming if v is not None) if any(v is not None for v in incoming) else None)
    if any(v is None for v in incoming):
        result["data_validity_warnings"].append("INCOMING_PROJECTION_MISSING_OR_PARTIAL")
    result["torque_signal_note"] = "Computed/applied are implicit-PD estimates; incoming projection is reaction, not motor torque"
    result["root_linear_velocity_first_rms_mps"] = math.sqrt(
        sum(metric(r)["root_linear_velocity_sq_sum"] for r in first)/n_samples)
    result["root_angular_velocity_first_rms_radps"] = math.sqrt(
        sum(metric(r)["root_angular_velocity_sq_sum"] for r in first)/n_samples)
    valid_load = sum(metric(r)["contact_valid_samples"] for r in first)
    result["foot_load_asymmetry_first_mean"] = (
        sum(metric(r)["contact_asymmetry_sum"] for r in first)/valid_load if valid_load else None)
    result["contact_force_load_sample_count"] = valid_load
    for threshold in (2, 5, 10, 20):
        result[f"S{threshold}"] = survival(first, threshold)
    finite_tree(result)
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenarios", nargs="+", choices=list(SCENARIOS), default=list(SCENARIOS))
    parser.add_argument("--zero_reference", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--visualizer", choices=["none"], default="none")
    parser.add_argument("--chunk_steps", type=int, default=16)
    parser.add_argument("--max_chunk_mib", type=float, default=64)
    return parser


# Runtime adapter appended below; all heavy imports remain inside this function.
def run_runtime(args):
    """Adapter based on evaluate_multiskill.py; never called by offline tests."""
    import copy
    import importlib.metadata as metadata
    import sys
    import numpy as np
    import torch
    from isaaclab.app import AppLauncher

    paths = validate_checkpoints(args.checkpoints)
    if not (1 <= args.num_envs <= 256 and 0 < args.seconds <= 60
            and 1 <= args.chunk_steps <= 256 and args.max_chunk_mib > 0
            and 0 <= args.seed < 2**31):
        raise ValueError("Bounds: 1..256 envs/chunk_steps, 0..60 seconds")
    if not args.device.startswith("cuda"):
        raise ValueError("C1 runtime requires PhysX CUDA")
    if len(set(args.scenarios)) != len(args.scenarios):
        raise ValueError("Duplicate scenarios")
    chunk_estimate = args.chunk_steps * args.num_envs * 320 * 4
    if chunk_estimate * 3 > args.max_chunk_mib * 1024**2:
        raise ValueError("Conservative 3x CPU chunk budget exceeded")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    pinned = {str(p): sha256_file(p) for p in paths}
    snapshot_hashes = {str(p.parent / "params" / f"{name}.yaml"):
                       sha256_file(p.parent / "params" / f"{name}.yaml")
                       for p in paths for name in ("agent", "env")}
    write_json_new(output / "invocation.json", {
        "schema": SCHEMA, "dataset_kind": REAL, "status": "STARTED",
        "arguments": vars(args), "checkpoint_hashes": pinned,
        "snapshot_hashes": snapshot_hashes,
        "runtime_qualification": "RUNTIME_VERIFICATION_REQUIRED",
        "resource_bounds": {"chunk_estimate_bytes": chunk_estimate,
                            "max_chunk_mib": args.max_chunk_mib,
                            "quantile_histogram_bins_per_episode": 3*1801},
        "production_changes": False, "training_iterations": 0})
    launch_parser = argparse.ArgumentParser(add_help=False)
    AppLauncher.add_app_launcher_args(launch_parser)
    launch_args = launch_parser.parse_args(["--device", args.device, "--visualizer", "none"])
    original_argv = sys.argv[:]
    sys.argv = [sys.argv[0]]
    launcher = None
    exit_code = 1
    try:
        launcher = AppLauncher(launch_args)
        import gymnasium as gym
        import warp as wp
        from tensordict import TensorDict
        from rsl_rl.runners import OnPolicyRunner
        from isaaclab.utils.io import dump_yaml
        from isaaclab.utils.math import euler_xyz_from_quat
        from isaaclab.utils.seed import configure_seed
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from g1_datn.tasks.manager_based.g1_multiskill.g1_multiskill_env import G1MultiSkillEnv
        import g1_datn  # noqa: F401
        from evaluation_io import load_snapshot

        def tensor(x):
            return wp.to_torch(x) if isinstance(x, wp.array) else x
        def array(x):
            return tensor(x).detach().cpu().numpy().copy()
        conditions = [(p, False, scenario) for p in paths for scenario in args.scenarios]
        if args.zero_reference:
            conditions.append((paths[-1], True, "standing"))
        completed = []
        for checkpoint, zero, scenario in conditions:
            label = "zero_actions" if zero else checkpoint.stem
            folder = output / label / scenario
            folder.mkdir(parents=True, exist_ok=False)
            saved_env = load_snapshot(checkpoint.parent / "params/env.yaml")
            saved_agent = load_snapshot(checkpoint.parent / "params/agent.yaml")
            if (saved_agent.get("class_name") != "OnPolicyRunner" or
                    saved_agent.get("algorithm", {}).get("class_name") != "PPO"):
                raise ValueError("C1 requires the saved OnPolicyRunner/PPO configuration")
            cfg = load_cfg_from_registry("G1-MultiSkill-v0", "env_cfg_entry_point")
            cfg.seed, cfg.log_dir = saved_env["seed"], saved_env["log_dir"]
            cfg.scene.terrain.env_spacing = saved_env["scene"]["terrain"]["env_spacing"]
            for name in ("robot", "height_scanner", "sky_light"):
                getattr(cfg.scene, name).spawn.spawn_path = saved_env["scene"][name]["spawn"]["spawn_path"]
            cfg.from_dict(saved_env)
            cfg.scene.num_envs, cfg.seed, cfg.log_dir = args.num_envs, args.seed, str(folder)
            cfg.commands.base_velocity.debug_vis = False
            cfg.sim.visualizer_cfgs = []
            if type(cfg.sim.physics).__name__ != "PhysxCfg" or cfg.sim.device != args.device:
                raise ValueError("Snapshot backend/device conflicts with C1 protocol")
            dump_yaml(str(folder / "eval_env.yaml"), cfg)
            configure_seed(args.seed, False)
            raw = gym.make("G1-MultiSkill-v0", cfg=cfg)
            wrapped = None
            try:
                env = raw.unwrapped
                if not isinstance(env, G1MultiSkillEnv):
                    raise ValueError("Reset-fixed task class inactive")
                wrapped = RslRlVecEnvWrapper(raw, clip_actions=saved_agent.get("clip_actions"))
                robot, sensor = env.scene["robot"], env.scene["contact_forces"]
                if robot.num_joints != 29 or wrapped.num_actions != 29:
                    raise ValueError("Expected 29 controlled DoF")
                steps = round(args.seconds/env.step_dt)
                if steps < 1 or abs(steps*env.step_dt-args.seconds) > EPS:
                    raise ValueError("Horizon must be a multiple of actual control dt")
                policy = None
                if not zero:
                    runner = OnPolicyRunner(wrapped, copy.deepcopy(saved_agent), log_dir=None, device=env.device)
                    loaded = torch.load(checkpoint, map_location=env.device, weights_only=True)
                    runner.alg.load(loaded, {"actor": True, "critic": True,
                                             "optimizer": False, "iteration": False}, strict=True)
                    for group in ("actor", "critic"):
                        if any(not torch.equal(v, loaded[group+"_state_dict"][k])
                               for k, v in getattr(runner.alg, group).state_dict().items()):
                            raise ValueError("Checkpoint/network tensor mismatch")
                    policy = runner.get_inference_policy(device=env.device)
                    del loaded
                command = env.command_manager.get_term("base_velocity")
                target = SCENARIOS[scenario]
                target_tensor = None if target is None else torch.tensor(target, device=env.device)
                resample0, update0 = command._resample_command, command._update_command
                def resample(ids):
                    resample0(ids)
                    if target_tensor is not None:
                        command.vel_command_b[ids] = target_tensor
                def update():
                    update0()
                    if target_tensor is not None:
                        command.vel_command_b[:] = target_tensor
                command._resample_command, command._update_command = resample, update
                feet, _ = sensor.find_bodies(["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True)
                if len(feet) != 2:
                    raise ValueError("Missing paired foot sensor bodies")
                with torch.inference_mode():
                    obs0, _ = env.reset(seed=args.seed)  # No scenario index offset.
                    obs = TensorDict(obs0, batch_size=[env.num_envs])
                if tuple(obs["policy"].shape) != (env.num_envs, 480) or tuple(obs["critic"].shape) != (env.num_envs, 495):
                    raise ValueError("Observation group dimensions do not match G1 MLP")
                if policy is not None:
                    policy.reset()
                def com():
                    masses = robot.data.body_mass.torch
                    return (masses[..., None]*robot.data.body_com_pos_w.torch).sum(1)/masses.sum(1, keepdim=True)
                initial_com = array(com())
                root_pose = array(robot.data.root_link_pose_w.torch)
                root_pose[:, :3] -= array(env.scene.env_origins)
                initial = {
                    "root_pose_relative_origins": root_pose,
                    "root_velocity": array(robot.root_view.get_root_velocities()),
                    "q": array(robot.data.joint_pos.torch), "qd": array(robot.data.joint_vel.torch),
                    "obs_policy": array(obs["policy"]), "obs_critic": array(obs["critic"]),
                    "mass": array(robot.data.body_mass.torch),
                    "inertias": array(robot.root_view.get_inertias()),
                    "materials": array(robot.root_view.get_material_properties()),
                    "kp": array(robot.data.joint_stiffness.torch),
                    "kd": array(robot.data.joint_damping.torch),
                    "effort_limits": array(robot.data.joint_effort_limits.torch),
                    "command": array(command.command), "com_w": initial_com,
                    "env_origins": array(env.scene.env_origins)}
                if any(not np.isfinite(v).all() for v in initial.values()):
                    raise ValueError("Nonfinite initial state")
                with (folder / "initial_state.npz").open("xb") as handle:
                    np.savez_compressed(handle, **initial)
                fingerprints = {k: hashlib.sha256(v.tobytes()).hexdigest()
                                for k, v in initial.items() if k != "env_origins"}
                manager = env.termination_manager
                term_names = list(manager.active_terms)
                timeout_terms = [n for n in term_names if manager.get_term_cfg(n).time_out]
                if not timeout_terms or not {"bad_orientation", "base_height"} <= set(term_names):
                    raise ValueError("Missing required termination terms")
                indices = [0]*env.num_envs
                accumulators = [EpisodeAccumulator(i, 0, env.step_dt, robot.joint_names, initial_com[i].tolist())
                                for i in range(env.num_envs)]
                first_records, chunks, buffer = [], [], []
                totals = {"records": 0, "physical": 0, "timeouts": 0}
                last_actions = None
                last_dones = None
                chunk_index = 0
                compute0 = manager.compute
                def emit(record, handle):
                    handle.write(json.dumps(record, allow_nan=False)+"\n")
                    totals["records"] += 1
                    totals["physical"] += record["event_type"] == "physical_failure"
                    totals["timeouts"] += record["event_type"] == "time_limit"
                    if record["first_episode"]:
                        first_records.append(record)
                def flush():
                    nonlocal chunk_index
                    if not buffer:
                        return
                    data = {k: np.stack([row[k] for row in buffer]) for k in buffer[0]}
                    filename = f"chunk-{chunk_index:05d}.npz"
                    with (folder / filename).open("xb") as handle:
                        np.savez_compressed(handle, **data)
                    chunks.append({"path": filename, "steps": len(buffer),
                                   "sha256": sha256_file(folder / filename),
                                   "bytes": (folder / filename).stat().st_size})
                    buffer.clear()
                    chunk_index += 1
                with (folder / "episodes.jsonl").open("x") as episode_file:
                    def measured_compute():
                        nonlocal last_dones
                        dones = compute0()
                        d = robot.data
                        roll, pitch, _ = euler_xyz_from_quat(d.root_quat_w.torch)
                        # atan2 exports can be [0,2pi); normalize signed angles.
                        roll = torch.atan2(torch.sin(roll), torch.cos(roll))
                        pitch = torch.atan2(torch.sin(pitch), torch.cos(pitch))
                        frame = {
                            "simulation_time_s": np.full(env.num_envs, (step+1)*env.step_dt, dtype=np.float32),
                            "episode_control_steps": array(env.episode_length_buf),
                            "root_quat_xyzw": array(d.root_quat_w.torch),
                            "root_pos_w": array(d.root_pos_w.torch),
                            "lin_vel_b": array(d.root_lin_vel_b.torch),
                            "ang_vel_b": array(d.root_ang_vel_b.torch),
                            "command": array(command.command), "com_w": array(com()),
                            "roll_deg": array(torch.rad2deg(roll)),
                            "pitch_deg": array(torch.rad2deg(pitch)),
                            "tilt_deg": array(torch.rad2deg(torch.acos((-d.projected_gravity_b.torch[:, 2]).clamp(-1, 1)))),
                            "root_height_m": array(d.root_pos_w.torch[:, 2]-env.scene.env_origins[:, 2]),
                            "action": array(last_actions), "q": array(d.joint_pos.torch),
                            "qd": array(d.joint_vel.torch), "target": array(d.joint_pos_target.torch),
                            "computed_torque": array(d.computed_torque.torch),
                            "applied_torque_estimate": array(d.applied_torque.torch),
                            "projected_incoming_force": array(robot.root_view.get_dof_projected_joint_forces()),
                            "effort_limits": array(d.joint_effort_limits.torch),
                            "foot_fz_N": array(sensor.data.net_forces_w.torch[:, feet, 2]),
                            "done": array(dones), "terminated": array(manager.terminated),
                            "truncated": array(manager.time_outs),
                            "term_flags": np.stack([array(manager.get_term(n)) for n in term_names], axis=-1),
                        }
                        if sum(v.nbytes for v in frame.values()) > args.num_envs*320*4:
                            raise ValueError("Telemetry schema exceeded conservative chunk budget")
                        if any(not np.isfinite(v).all() for v in frame.values()):
                            raise ValueError("Nonfinite runtime telemetry")
                        if target is not None and not np.allclose(frame["command"], target, atol=1e-6):
                            raise ValueError("Fixed command not maintained")
                        last_dones = frame["done"].astype(bool)
                        for i in range(env.num_envs):
                            sample = {k: v[i].tolist() for k, v in frame.items()
                                      if k not in {"done", "terminated", "truncated", "term_flags"}}
                            sample["support_state"] = "MISSING_TELEMETRY"
                            accumulators[i].add(sample)
                            classification = classify_termination(
                                {n: bool(frame["term_flags"][i, k]) for k, n in enumerate(term_names)},
                                timeout_terms, done=bool(frame["done"][i]),
                                terminated=bool(frame["terminated"][i]), truncated=bool(frame["truncated"][i]))
                            if classification["event_type"] != "ongoing":
                                emit(accumulators[i].finish(classification), episode_file)
                                indices[i] += 1
                                accumulators[i] = None
                        buffer.append(frame)
                        if len(buffer) >= args.chunk_steps:
                            flush()
                        return dones
                    manager.compute = measured_compute
                    try:
                        with torch.inference_mode():
                            for step in range(steps):
                                last_actions = torch.zeros((env.num_envs, 29), device=env.device) if zero else policy(obs, stochastic_output=False)
                                if not torch.isfinite(last_actions).all():
                                    raise ValueError("Nonfinite policy action")
                                obs, rewards, done, _ = wrapped.step(last_actions)
                                if not torch.isfinite(rewards).all() or any(not torch.isfinite(v).all() for v in obs.values()):
                                    raise ValueError("Nonfinite returned observation/reward")
                                if not np.array_equal(array(done).astype(bool), last_dones):
                                    raise ValueError("Wrapper dones disagree with manager callback")
                                if policy is not None:
                                    policy.reset(done)
                                # Initialize only when a subsequent sample will exist.
                                if step+1 < steps and any(x is None for x in accumulators):
                                    reset_com = array(com())
                                    for i, accumulator in enumerate(accumulators):
                                        if accumulator is None:
                                            accumulators[i] = EpisodeAccumulator(i, indices[i], env.step_dt, robot.joint_names, reset_com[i].tolist())
                        for accumulator in accumulators:
                            if accumulator is not None and accumulator.samples:
                                emit(accumulator.finish({"event_type": "evaluation_horizon",
                                     "physical_reasons": [], "timeout_reasons": [], "time_limit_flag": False}), episode_file)
                    finally:
                        manager.compute = compute0
                    flush()
                summary = summarize_episodes(first_records, env.num_envs, args.seconds)
                summary.update(all_episode_records=totals["records"],
                               later_episode_records=totals["records"]-env.num_envs,
                               all_physical_failures=totals["physical"], all_timeouts=totals["timeouts"])
                write_json_new(folder / "summary.json", summary)
                condition = {
                    "schema": SCHEMA, "dataset_kind": REAL, "status": "COMPLETE",
                    "checkpoint": label, "checkpoint_path": str(checkpoint),
                    "checkpoint_sha256": pinned[str(checkpoint)],
                    "snapshot_sha256": {name: snapshot_hashes[str(checkpoint.parent/"params"/f"{name}.yaml")]
                                        for name in ("env", "agent")},
                    "scenario": scenario, "seed": args.seed, "num_envs": env.num_envs,
                    "horizon_s": args.seconds, "dt": env.step_dt,
                    "episodes_path": "episodes.jsonl", "episodes_sha256": sha256_file(folder/"episodes.jsonl"),
                    "initial_state_path": "initial_state.npz", "initial_sha256": sha256_file(folder/"initial_state.npz"),
                    "initial_fingerprints": fingerprints, "term_names": term_names,
                    "timeout_terms": timeout_terms, "env_class": type(env).__name__,
                    "joint_names": robot.joint_names, "chunks": chunks,
                    "runtime_qualification": "RUNTIME_VERIFICATION_REQUIRED",
                    "raw_contact_hull": "MISSING_TELEMETRY",
                    "projected_incoming_forces": "Collected separately as reaction projection, not motor torque",
                    "measurement": "post-physics/pre-auto-reset; computed/applied are implicit-PD estimates",
                    "angle_histogram_resolution_deg": .1,
                    "local_overrides": ["scene.num_envs", "seed", "log_dir", "debug_vis=false",
                                        "visualizer_cfgs=[]", "fixed commands for named scenarios"],
                    "preserved": ["reward", "curriculum", "PD", "posture", "action_scale",
                                  "observation_noise", "startup/reset/interval_events", "terminations"],
                    "training_iterations": 0,
                    "versions": {n: metadata.version(n) for n in ("isaacsim", "isaaclab", "torch", "rsl-rl-lib")}}
                write_json_new(folder / "condition.json", condition)
                completed.append(str((folder/"condition.json").relative_to(output)))
            finally:
                if wrapped is not None:
                    wrapped.close()
                else:
                    raw.close()
        for name, digest in {**pinned, **snapshot_hashes}.items():
            if sha256_file(name) != digest:
                raise ValueError(f"Input mutated during inference: {name}")
        write_json_new(output / "manifest.json", {"schema": SCHEMA, "dataset_kind": REAL,
                       "status": "COMPLETE", "conditions": completed, "training_iterations": 0,
                       "runtime_qualification": "RUNTIME_VERIFICATION_REQUIRED"})
        exit_code = 0
    finally:
        sys.argv = original_argv
        if launcher is not None:
            launcher.app.close(wait_for_replicator=False, exit_code=exit_code)


def main(argv=None):
    args = build_parser().parse_args(argv)
    if Path(args.output).exists():
        raise FileExistsError("Output exists; refusing overwrite before simulator imports")
    run_runtime(args)


if __name__ == "__main__":
    main()
