"""Run the upstream G1 FastSAC recipe without its dynamic CLI parser."""

import argparse
from dataclasses import replace

from holosoma.config_values.loco.g1.experiment import g1_29dof_fast_sac
from holosoma.config_values import logger, simulator
from holosoma.train_agent import train

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=4096)
parser.add_argument("--iterations", type=int, default=50000)
parser.add_argument("--name", default="g1_fastsac_paper")
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--headless", action="store_true")
parser.add_argument("--learning-starts", type=int, default=None)
parser.add_argument("--simulator", choices=("isaacsim", "mjwarp"), default="isaacsim")
args = parser.parse_args()
backend = getattr(simulator, args.simulator)

config = replace(
    g1_29dof_fast_sac,
    simulator=replace(
        backend,
        config=replace(
            backend.config,
            debug_viz=False,
            viewer=replace(backend.config.viewer, enable_tracking=True),
        ),
    ),
    logger=replace(logger.disabled, base_dir="logs/fastsac_holosoma"),
    training=replace(
        g1_29dof_fast_sac.training,
        headless=args.headless,
        seed=1,
        num_envs=args.num_envs,
        name=args.name,
        checkpoint=args.checkpoint,
    ),
    algo=replace(
        g1_29dof_fast_sac.algo,
        _target_="training_visualization.LiveFastSACAgent",
        config=replace(
            g1_29dof_fast_sac.algo.config,
            num_learning_iterations=args.iterations,
            learning_starts=(
                g1_29dof_fast_sac.algo.config.learning_starts
                if args.learning_starts is None else args.learning_starts
            ),
        ),
    ),
)
train(config)
