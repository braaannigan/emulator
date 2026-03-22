from __future__ import annotations

import argparse
from pathlib import Path

from src.models.residual_thickness.config import load_residual_thickness_config
from src.models.residual_thickness.tuning import run_residual_thickness_hpo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Ray Tune hyperparameter optimization for the residual thickness emulator.")
    parser.add_argument(
        "--config",
        default="config/emulator/residual_thickness_depthwise_kernel5.yaml",
        help="Path to the incumbent emulator YAML configuration.",
    )
    parser.add_argument(
        "--experiment-name",
        required=True,
        help="Experiment id prefix for the Ray tuning run and trial artifacts.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help="Number of Ray Tune samples to evaluate.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_residual_thickness_config(Path(args.config))
    summary = run_residual_thickness_hpo(
        config,
        experiment_name=args.experiment_name,
        num_samples=args.num_samples,
    )
    print(summary["summary_path"])
    print(summary["best_artifacts"]["metrics_path"])
    print(summary["best_artifacts"]["rollout_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
