from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a 2x2 heatmap for truth/rollout surface thickness and temporal residuals "
            "from a rollout dataset."
        )
    )
    parser.add_argument(
        "--rollout-path",
        required=True,
        help=(
            "Path to rollout NetCDF. Accepts thickness variables "
            "(`truth_layer_thickness`/`rollout_layer_thickness`) or legacy (`truth`/`rollout`) "
            "with shape [time, y, x]."
        ),
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Path for output PNG.",
    )
    parser.add_argument(
        "--step-index",
        type=int,
        default=-1,
        help="Time index to visualize. Negative values use Python-style indexing (default: -1).",
    )
    return parser.parse_args()


def _resolve_step_index(step_index: int, frame_count: int) -> int:
    resolved = step_index if step_index >= 0 else frame_count + step_index
    if resolved <= 0 or resolved >= frame_count:
        raise ValueError(
            f"step_index resolves to {resolved}, but valid range for residual plotting is [1, {frame_count - 1}]."
        )
    return resolved


def render_heatmap(rollout_path: Path, output_path: Path, step_index: int = -1) -> Path:
    dataset = xr.open_dataset(rollout_path)
    try:
        if "truth_layer_thickness" in dataset and "rollout_layer_thickness" in dataset:
            truth_key = "truth_layer_thickness"
            rollout_key = "rollout_layer_thickness"
        elif "truth" in dataset and "rollout" in dataset:
            truth_key = "truth"
            rollout_key = "rollout"
        else:
            raise ValueError(
                "rollout dataset must contain either "
                "`truth_layer_thickness`/`rollout_layer_thickness` or `truth`/`rollout`."
            )
        truth_series = np.asarray(dataset[truth_key].values, dtype=np.float32)
        rollout_series = np.asarray(dataset[rollout_key].values, dtype=np.float32)
    finally:
        dataset.close()

    if truth_series.ndim != 3 or rollout_series.ndim != 3:
        raise ValueError("truth and rollout variables must have shape [time, y, x].")
    if truth_series.shape != rollout_series.shape:
        raise ValueError("truth and rollout variables must have matching shapes.")

    resolved_step = _resolve_step_index(step_index, truth_series.shape[0])

    truth = truth_series[resolved_step]
    truth_prev = truth_series[resolved_step - 1]
    rollout = rollout_series[resolved_step]
    rollout_prev = rollout_series[resolved_step - 1]

    truth_residual = truth - truth_prev
    rollout_residual = rollout - rollout_prev

    thickness_min = float(min(truth.min(), rollout.min()))
    thickness_max = float(max(truth.max(), rollout.max()))
    residual_abs = float(
        max(
            np.abs(truth_residual).max(),
            np.abs(rollout_residual).max(),
        )
    )

    figure, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)

    im_truth = axes[0, 0].imshow(
        truth,
        origin="lower",
        cmap="viridis",
        vmin=thickness_min,
        vmax=thickness_max,
        aspect="auto",
    )
    axes[0, 0].set_title("Truth Surface Thickness")

    im_truth_residual = axes[0, 1].imshow(
        truth_residual,
        origin="lower",
        cmap="RdBu_r",
        vmin=-residual_abs,
        vmax=residual_abs,
        aspect="auto",
    )
    axes[0, 1].set_title("Truth Residual")

    im_rollout = axes[1, 0].imshow(
        rollout,
        origin="lower",
        cmap="viridis",
        vmin=thickness_min,
        vmax=thickness_max,
        aspect="auto",
    )
    axes[1, 0].set_title("Rollout Surface Thickness")

    im_rollout_residual = axes[1, 1].imshow(
        rollout_residual,
        origin="lower",
        cmap="RdBu_r",
        vmin=-residual_abs,
        vmax=residual_abs,
        aspect="auto",
    )
    axes[1, 1].set_title("Rollout Residual")

    for row in axes:
        for axis in row:
            axis.set_xlabel("x index")
            axis.set_ylabel("y index")

    figure.colorbar(im_truth, ax=[axes[0, 0], axes[1, 0]], shrink=0.9, label="thickness")
    figure.colorbar(
        im_truth_residual,
        ax=[axes[0, 1], axes[1, 1]],
        shrink=0.9,
        label="residual (current - previous)",
    )
    figure.suptitle(f"Step {resolved_step}: Truth vs Rollout Surface Thickness and Residuals")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def main() -> int:
    args = parse_args()
    output = render_heatmap(Path(args.rollout_path), Path(args.output_path), step_index=args.step_index)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
