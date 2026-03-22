from __future__ import annotations

import argparse
from pathlib import Path

from src.generator.config import load_double_gyre_config
from src.generator.double_gyre import run_double_gyre_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Aronnax double gyre model with shifting wind stress.")
    parser.add_argument(
        "--config",
        default="config/generator/double_gyre_shifting_wind.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--duration-days",
        type=float,
        help="Override the YAML run duration in days.",
    )
    parser.add_argument(
        "--output-interval-days",
        type=float,
        help="Override the YAML output interval in days.",
    )
    parser.add_argument(
        "--experiment-id",
        help="Optional experiment directory name. Defaults to a timestamp like 20260319T120000.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_double_gyre_config(Path(args.config))
    if args.duration_days is not None:
        config = config.with_overrides(duration_days=args.duration_days)
    if args.output_interval_days is not None:
        config = config.with_overrides(output_interval_days=args.output_interval_days)
    if args.experiment_id is not None:
        config = config.with_overrides(experiment_id=args.experiment_id)
    output_path = run_double_gyre_pipeline(config)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
