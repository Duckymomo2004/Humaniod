"""Strict C1 matrix summarizer; standard-library only, never starts a simulator."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from evaluate_multiskill_c1 import (
    SCHEMA, REAL, SYNTHETIC, EPS, finite_tree, read_json, sha256_file,
    summarize_episodes, validate_episode, write_json_new,
)

DEFAULT_CHECKPOINTS = ("model_0", "model_100", "model_199")
DEFAULT_SCENARIOS = ("standing", "forward_slow", "training_commands")
DEFAULT_SEEDS = (42, 43, 44)


def expected_matrix(checkpoints=DEFAULT_CHECKPOINTS, scenarios=DEFAULT_SCENARIOS,
                    seeds=DEFAULT_SEEDS):
    if (len(set(checkpoints)) != len(checkpoints) or
            len(set(scenarios)) != len(scenarios) or len(set(seeds)) != len(seeds)):
        raise ValueError("Duplicate matrix specification")
    if not checkpoints or not seeds or set(scenarios) != set(DEFAULT_SCENARIOS):
        raise ValueError("Three C1 scenarios and nonempty candidates/seeds required")
    return {(c, s, seed) for c in checkpoints for s in scenarios for seed in seeds} | {
        ("zero_actions", "standing", seed) for seed in seeds}


def child_path(folder, value):
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError("Manifest paths must be relative")
    result = (folder / relative).resolve()
    if not result.is_relative_to(folder.resolve()):
        raise ValueError("Manifest path traversal")
    return result


def read_condition(path, allow_synthetic=False):
    condition = read_json(path)
    finite_tree(condition)
    if condition.get("schema") != SCHEMA or condition.get("status") != "COMPLETE":
        raise ValueError("Incomplete/unknown condition")
    kind = condition.get("dataset_kind")
    if kind != REAL and not (allow_synthetic and kind == SYNTHETIC):
        raise ValueError("Synthetic fixtures require --allow-synthetic; no mixed research data")
    for field in ("checkpoint", "scenario", "seed", "num_envs", "horizon_s", "dt",
                  "episodes_path", "episodes_sha256", "snapshot_sha256",
                  "checkpoint_sha256", "initial_fingerprints"):
        if field not in condition:
            raise ValueError(f"Missing condition field {field}")
    if condition["num_envs"] <= 0 or condition["horizon_s"] <= 0 or condition["dt"] <= 0:
        raise ValueError("Invalid condition dimensions")
    folder = Path(path).parent
    episodes = child_path(folder, condition["episodes_path"])
    if sha256_file(episodes) != condition["episodes_sha256"]:
        raise ValueError("Episode stream checksum mismatch")
    first, seen, total, physical, timeouts, tails, exposure = [], set(), 0, 0, 0, 0, {}
    with episodes.open() as handle:
        while True:
            line = handle.readline(1024*1024 + 1)
            if not line:
                break
            if len(line) > 1024*1024 or not line.endswith("\n"):
                raise ValueError("Oversized/truncated episode record")
            # Same strict duplicate/nonfinite checks as JSON object files.
            def unique(pairs):
                d = {}
                for key, value in pairs:
                    if key in d:
                        raise ValueError("Duplicate episode key")
                    d[key] = value
                return d
            def invalid_number(x):
                raise ValueError(f"Nonfinite JSON token {x}")
            record = json.loads(line, object_pairs_hook=unique, parse_constant=invalid_number)
            validate_episode(record)
            env = record["env_id"]
            if env >= condition["num_envs"] or abs(record["dt"]-condition["dt"]) > EPS:
                raise ValueError("Episode env/dt conflicts with condition")
            expected_index = exposure.get(env, {}).get("next_index", 0)
            if record.get("episode_index") != expected_index or record["first_episode"] != (expected_index == 0):
                raise ValueError("Duplicate/out-of-order episode or incorrect first marker")
            state = exposure.setdefault(env, {"duration": 0.0, "next_index": 0, "censored": False})
            if state["censored"]:
                raise ValueError("A censored tail cannot be followed by another episode")
            state["duration"] += record["observed_duration_s"]
            state["next_index"] += 1
            state["censored"] = record["event_type"] == "evaluation_horizon"
            total += 1
            physical += record["event_type"] == "physical_failure"
            timeouts += record["event_type"] == "time_limit"
            tails += record["event_type"] == "evaluation_horizon"
            if record["first_episode"]:
                if env in seen:
                    raise ValueError("Duplicate first episode")
                seen.add(env)
                first.append(record)
    if seen != set(range(condition["num_envs"])):
        raise ValueError("Missing first episodes")
    if any(abs(v["duration"]-condition["horizon_s"]) > EPS for v in exposure.values()):
        raise ValueError("Missing/truncated per-environment evaluation exposure")
    if kind == REAL:
        import re
        digests = [condition["checkpoint_sha256"]]
        snapshots = condition["snapshot_sha256"]
        if not isinstance(snapshots, dict) or set(snapshots) != {"env", "agent"}:
            raise ValueError("Missing runtime snapshot identities")
        digests.extend(snapshots.values())
        fingerprints = condition["initial_fingerprints"]
        required_fingerprints = {"root_pose_relative_origins", "root_velocity", "q", "qd",
                                 "obs_policy", "obs_critic", "mass", "inertias", "materials",
                                 "kp", "kd", "effort_limits", "command", "com_w"}
        if not isinstance(fingerprints, dict) or not required_fingerprints <= set(fingerprints):
            raise ValueError("Missing runtime pairing fingerprint fields")
        digests.extend(fingerprints.values())
        if any(not isinstance(x, str) or not re.fullmatch("[0-9a-f]{64}", x) for x in digests):
            raise ValueError("Invalid runtime provenance digest")
        for filename_key, hash_key in (("initial_state_path", "initial_sha256"),):
            if filename_key not in condition or hash_key not in condition:
                raise ValueError("Missing runtime initial-state artifact")
            if sha256_file(child_path(folder, condition[filename_key])) != condition[hash_key]:
                raise ValueError("Initial state checksum mismatch")
        chunks = condition.get("chunks")
        if not isinstance(chunks, list) or not chunks:
            raise ValueError("Missing bounded telemetry chunks")
        steps = 0
        names = set()
        for chunk in chunks:
            file = child_path(folder, chunk["path"])
            if file in names or chunk["steps"] < 1:
                raise ValueError("Invalid/duplicate telemetry chunk")
            names.add(file)
            if file.stat().st_size != chunk["bytes"] or sha256_file(file) != chunk["sha256"]:
                raise ValueError("Telemetry chunk integrity failure")
            steps += chunk["steps"]
        if abs(steps*condition["dt"]-condition["horizon_s"]) > EPS:
            raise ValueError("Incomplete telemetry horizon")
    summary = summarize_episodes(first, condition["num_envs"], condition["horizon_s"])
    summary.update(all_episode_records=total, later_episode_records=total-len(first),
                   all_physical_failures=physical, all_timeouts=timeouts,
                   all_evaluation_horizon_censors=tails)
    summary.update(checkpoint=condition["checkpoint"], scenario=condition["scenario"],
                   seed=condition["seed"], dataset_kind=kind, condition_path=str(path))
    return condition, summary


def summarize_matrix(root, *, checkpoints=DEFAULT_CHECKPOINTS, scenarios=DEFAULT_SCENARIOS,
                     seeds=DEFAULT_SEEDS, num_envs=64, horizon_s=20.0, allow_synthetic=False):
    expected = expected_matrix(checkpoints, scenarios, seeds)
    metadata, summaries = {}, {}
    for path in sorted(Path(root).rglob("condition.json")):
        condition, summary = read_condition(path, allow_synthetic)
        key = (condition["checkpoint"], condition["scenario"], condition["seed"])
        if key in metadata:
            raise ValueError(f"Duplicate evaluation condition {key}")
        if condition["num_envs"] != num_envs or abs(condition["horizon_s"]-horizon_s) > EPS:
            raise ValueError("Horizon/env count mismatch across planned matrix")
        metadata[key], summaries[key] = condition, summary
    found = set(metadata)
    if found != expected:
        raise ValueError(f"Incomplete evaluation matrix; missing={sorted(expected-found)}, unexpected={sorted(found-expected)}")
    if len({m["dataset_kind"] for m in metadata.values()}) != 1:
        raise ValueError("Never mix synthetic and real evaluation conditions")
    snapshots = {json.dumps(m["snapshot_sha256"], sort_keys=True) for m in metadata.values()}
    if len(snapshots) != 1:
        raise ValueError("C1 matrix mixes different saved configuration snapshots")
    pairing = []
    for seed in seeds:
        for scenario in scenarios:
            base = metadata[(checkpoints[0], scenario, seed)]["initial_fingerprints"]
            labels = list(checkpoints[1:]) + (["zero_actions"] if scenario == "standing" else [])
            for label in labels:
                current = metadata[(label, scenario, seed)]["initial_fingerprints"]
                differences = sorted(k for k in set(base)|set(current) if base.get(k) != current.get(k))
                pairing.append({"seed": seed, "scenario": scenario, "checkpoint": label,
                                "reference": checkpoints[0], "different_fingerprint_fields": differences,
                                "status": "EXACT_FINGERPRINT_MATCH" if not differences else "PAIRING_UNVERIFIED"})
                if differences:
                    summaries[(label, scenario, seed)]["data_validity_warnings"].append("INITIAL_PAIRING_MISMATCH")
    rows = [summaries[key] for key in sorted(expected)]
    return {"schema": SCHEMA, "status": "COMPLETE_OFFLINE_RECOMPUTED_RUNTIME_QUALIFICATION_PENDING",
            "dataset_kind": rows[0]["dataset_kind"], "condition_count": len(rows),
            "training_replications": 1,
            "statistical_unit": "evaluation seed/scene cluster; not independent training replications",
            "primary_support_gate": "NOT_QUALIFIED",
            "runtime_qualification": "RUNTIME_VERIFICATION_REQUIRED",
            "rows": rows, "pairing": pairing}


def write_summary(result, output):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    write_json_new(out / "summary.json", result)
    # All nested statistics preserved as JSON cells; no censor/validity fields dropped.
    rows = result["rows"]
    with (out / "summary.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True, allow_nan=False)
                             if isinstance(v, (dict, list)) else v for k, v in row.items()})
    return out


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoints", nargs="+", default=list(DEFAULT_CHECKPOINTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--allow-synthetic", action="store_true",
                        help="Tests only: accept explicitly marked isolated fixtures")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if Path(args.output).exists():
        raise FileExistsError("Output exists; refusing overwrite")
    result = summarize_matrix(args.input, checkpoints=args.checkpoints, seeds=args.seeds,
                              num_envs=args.num_envs, horizon_s=args.seconds,
                              allow_synthetic=args.allow_synthetic)
    print(write_summary(result, args.output))


if __name__ == "__main__":
    main()
