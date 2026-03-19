from __future__ import annotations

import argparse
from pathlib import Path

from src.animation.double_gyre_animation import create_animation
from src.viewer.double_gyre_viewer import list_experiments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an MP4 animation for a double gyre experiment.")
    parser.add_argument(
        "--experiment-id",
        help="Experiment directory name under data/raw/double_gyre. Defaults to the most recent experiment.",
    )
    parser.add_argument(
        "--netcdf-path",
        help="Optional explicit path to the experiment NetCDF file.",
    )
    parser.add_argument(
        "--output-path",
        help="Optional explicit path for the MP4. Defaults to the same directory as the NetCDF.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=4,
        help="Frames per second for the MP4 animation.",
    )
    return parser.parse_args()


def resolve_netcdf_path(args: argparse.Namespace) -> Path:
    if args.netcdf_path:
        return Path(args.netcdf_path)

    experiments = list_experiments()
    if not experiments:
        raise FileNotFoundError("No experiments found under data/raw/double_gyre.")

    if args.experiment_id:
        for experiment in experiments:
            if experiment.experiment_id == args.experiment_id:
                return experiment.netcdf_path
        raise FileNotFoundError(f"Experiment {args.experiment_id!r} was not found.")

    return experiments[0].netcdf_path


def main() -> int:
    args = parse_args()
    netcdf_path = resolve_netcdf_path(args)
    output_path = create_animation(netcdf_path=netcdf_path, output_path=args.output_path, fps=args.fps)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
