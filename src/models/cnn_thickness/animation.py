from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def create_rollout_comparison_animation(
    rollout_path: Path, output_path: Path, fps: int
) -> Path:
    dataset = xr.open_dataset(rollout_path)
    try:
        time_days = np.asarray(dataset["time_days"].values)
        x = np.asarray(dataset["x"].values)
        y = np.asarray(dataset["y"].values)
        fields = _resolve_rollout_fields(dataset)
        color_limits = {
            field_name: _compute_color_limits(np.asarray(dataset[f"truth_{field_name}"].values), field_name)
            for field_name in fields
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(output_path, fps=fps) as writer:
            for index, time_day in enumerate(time_days):
                frame = _render_frame(
                    dataset=dataset,
                    frame_index=index,
                    fields=fields,
                    x=x,
                    y=y,
                    time_day=float(time_day),
                    color_limits=color_limits,
                )
                writer.append_data(frame)
    finally:
        dataset.close()
    return output_path


def _resolve_rollout_fields(dataset: xr.Dataset) -> tuple[str, ...]:
    fields = ["layer_thickness"]
    if "truth_zonal_velocity" in dataset and "rollout_zonal_velocity" in dataset:
        fields.append("zonal_velocity")
    if "truth_meridional_velocity" in dataset and "rollout_meridional_velocity" in dataset:
        fields.append("meridional_velocity")
    return tuple(fields)


def _compute_color_limits(values: np.ndarray, field_name: str) -> tuple[float, float]:
    if field_name in {"zonal_velocity", "meridional_velocity"}:
        max_abs = float(np.nanmax(np.abs(values)))
        return -max_abs, max_abs
    return float(np.nanmin(values)), float(np.nanmax(values))


def _render_frame(
    dataset: xr.Dataset,
    frame_index: int,
    fields: tuple[str, ...],
    x: np.ndarray,
    y: np.ndarray,
    time_day: float,
    color_limits: dict[str, tuple[float, float]],
) -> np.ndarray:
    figure, axes = plt.subplots(2, len(fields), figsize=(6.4 * len(fields), 10.08), constrained_layout=True)
    if len(fields) == 1:
        axes = np.asarray(axes)[:, np.newaxis]
    row_titles = ["ARONNAX", "AIRONNAX"]
    for row_index, prefix in enumerate(["truth", "rollout"]):
        layer_thickness = np.asarray(dataset[f"{prefix}_layer_thickness"].values[frame_index])
        for column_index, field_name in enumerate(fields):
            axis = axes[row_index, column_index]
            values = np.asarray(dataset[f"{prefix}_{field_name}"].values[frame_index])
            vmin, vmax = color_limits[field_name]
            mesh = axis.pcolormesh(
                x, y, values, cmap="RdBu_r", shading="auto", vmin=vmin, vmax=vmax
            )
            if field_name in {"zonal_velocity", "meridional_velocity"}:
                axis.contour(
                    x,
                    y,
                    layer_thickness,
                    colors="black",
                    linewidths=0.7,
                )
            axis.set_title(f"{row_titles[row_index]} {field_name}")
            axis.set_xlabel("x")
            axis.set_ylabel("y")
            figure.colorbar(mesh, ax=axis)
    figure.suptitle(f"Rollout comparison at day {time_day:.3f}", fontsize=16)
    figure.text(
        0.5,
        0.98,
        f"Time = {time_day:.3f} days",
        ha="center",
        va="top",
        fontsize=14,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "black"},
    )
    figure.canvas.draw()
    frame = np.asarray(figure.canvas.buffer_rgba())[..., :3]
    plt.close(figure)
    return frame
