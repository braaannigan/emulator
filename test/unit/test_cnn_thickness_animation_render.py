from pathlib import Path

import numpy as np
import xarray as xr

from src.models.cnn_thickness.animation import create_rollout_comparison_animation


def test_create_rollout_comparison_animation_writes_mp4(tmp_path: Path):
    rollout_path = tmp_path / "rollout.nc"
    dataset = xr.Dataset(
        data_vars={
            "truth_layer_thickness": (("time_days", "y", "x"), np.ones((1, 2, 2), dtype=np.float32)),
            "rollout_layer_thickness": (("time_days", "y", "x"), np.zeros((1, 2, 2), dtype=np.float32)),
        },
        coords={"time_days": [1.0], "y": [0.0, 1.0], "x": [0.0, 1.0]},
    )
    dataset.to_netcdf(rollout_path)
    dataset.close()

    output_path = create_rollout_comparison_animation(rollout_path, tmp_path / "comparison.mp4", fps=2)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_create_rollout_comparison_animation_uses_truth_color_limits(monkeypatch, tmp_path: Path):
    rollout_path = tmp_path / "rollout.nc"
    dataset = xr.Dataset(
        data_vars={
            "truth_layer_thickness": (("time_days", "y", "x"), np.array([[[2.0, 3.0], [4.0, 5.0]]], dtype=np.float32)),
            "rollout_layer_thickness": (
                ("time_days", "y", "x"),
                np.array([[[0.0, 10.0], [20.0, 30.0]]], dtype=np.float32),
            ),
        },
        coords={"time_days": [1.0], "y": [0.0, 1.0], "x": [0.0, 1.0]},
    )
    dataset.to_netcdf(rollout_path)
    dataset.close()

    captured: dict[str, float] = {}

    def fake_render_frame(*, truth, rollout, x, y, time_day, vmin, vmax):
        captured["vmin"] = vmin
        captured["vmax"] = vmax
        return np.zeros((16, 16, 3), dtype=np.uint8)

    monkeypatch.setattr("src.models.cnn_thickness.animation._render_frame", fake_render_frame)

    create_rollout_comparison_animation(rollout_path, tmp_path / "comparison.mp4", fps=2)

    assert captured == {"vmin": 2.0, "vmax": 5.0}
