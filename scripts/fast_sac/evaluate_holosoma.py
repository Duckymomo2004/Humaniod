"""Evaluate a saved G1 policy and record its trajectory using upstream Holosoma."""

import argparse
import os
from dataclasses import replace
from pathlib import Path

from holosoma.config_types.eval_callback import EvalCallbacksConfig, RecordingCallbackConfig, RecordingConfig
from holosoma.eval_agent import run_eval_with_tyro
from holosoma.utils.eval_utils import CheckpointConfig, init_eval_logging, load_saved_experiment_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--demo", action="store_true", help="Run the final policy continuously at real-time speed without recording.")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    os.environ["FASTSAC_LIVE_MODE"] = "Final policy demo" if args.demo else "Policy evaluation"
    if args.demo:
        os.environ["FASTSAC_LIVE_REALTIME"] = "1"
        os.environ["FASTSAC_STATE_FILE"] = "run/fastsac/demo_state.npz"
    if args.steps <= 0:
        parser.error("--steps must be positive")
    checkpoint = CheckpointConfig(checkpoint=str(args.checkpoint.resolve()))
    init_eval_logging()
    saved, wandb_path = load_saved_experiment_config(checkpoint)
    config = saved.get_eval_config()
    config = replace(
        config,
        training=replace(
            config.training,
            headless=args.headless,
            max_eval_steps=None if args.demo else args.steps,
            name=f"{saved.training.name}_evaluation",
            export_onnx=False,
        ),
        logger=replace(config.logger, base_dir="logs/fastsac_holosoma"),
    )
    callbacks = EvalCallbacksConfig(
        recording=RecordingCallbackConfig(config=RecordingConfig(enabled=not args.demo, output_path="trajectory.npz"))
    )
    run_eval_with_tyro(config, checkpoint, saved, wandb_path, eval_cbs_cfg=callbacks)


if __name__ == "__main__":
    main()
