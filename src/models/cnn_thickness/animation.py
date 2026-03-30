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
        truth = np.asarray(dataset["truth_layer_thickness"].values)
        rollout = np.asarray(dataset["rollout_layer_thickness"].values)
        time_days = np.asarray(dataset["time_days"].values)
        x = np.asarray(dataset["x"].values)
        y = np.asarray(dataset["y"].values)
        # Keep the comparison scale anchored to the generator truth field so rollout
        # excursions do not wash out the visual contrast in the reference solution.
        vmin = float(truth.min())
        vmax = float(truth.max())

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(output_path, fps=fps) as writer:
            for index, time_day in enumerate(time_days):
                frame = _render_frame(
                    truth=truth[index],
                    rollout=rollout[index],
                    x=x,
                    y=y,
                    time_day=float(time_day),
                    vmin=vmin,
                    vmax=vmax,
                )
                writer.append_data(frame)
    finally:
        dataset.close()
    return output_path


def _render_frame(
    truth: np.ndarray,
    rollout: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    time_day: float,
    vmin: float,
    vmax: float,
) -> np.ndarray:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.04), constrained_layout=True)
    for axis, values, title in zip(axes, [truth, rollout], ["Aronnax", "AIronnax"]):
        mesh = axis.pcolormesh(
            x, y, values, cmap="RdBu_r", shading="auto", vmin=vmin, vmax=vmax
        )
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        figure.colorbar(mesh, ax=axis)
    figure.suptitle(f"Layer thickness comparison at day {time_day:.3f}", fontsize=16)
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
