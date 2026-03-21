from __future__ import annotations

import argparse
from pathlib import Path

from src.models.residual_thickness.config import load_residual_thickness_config
from src.models.residual_thickness.pipeline import run_residual_thickness_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate a residual thickness emulator.")
    parser.add_argument(
        "--config",
        default="config/emulator/residual_thickness.yaml",
        help="Path to the emulator YAML configuration.",
    )
    parser.add_argument(
        "--source-experiment-id",
        help="Override the source double gyre experiment id.",
    )
    parser.add_argument(
        "--experiment-id",
        help="Optional emulator experiment id. Defaults to a timestamp.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Optional training epoch override.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_residual_thickness_config(Path(args.config))
    if args.source_experiment_id is not None:
        config = config.with_overrides(source_experiment_id=args.source_experiment_id)
    if args.experiment_id is not None:
        config = config.with_overrides(experiment_id=args.experiment_id)
    if args.epochs is not None:
        config = config.with_overrides(epochs=args.epochs)

    outputs = run_residual_thickness_experiment(config)
    print(outputs["metrics_path"])
    print(outputs["rollout_path"])
    print(outputs["animation_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
