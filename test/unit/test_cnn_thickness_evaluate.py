from pathlib import Path

import json
import numpy as np
import xarray as xr

from src.models.cnn_thickness.evaluate import mse_per_timestep, save_metrics, save_rollout_dataset


def test_mse_per_timestep_returns_one_value_per_frame():
    truth = np.array([[[1.0]], [[3.0]]], dtype=np.float32)
    rollout = np.array([[[2.0]], [[5.0]]], dtype=np.float32)

    values = mse_per_timestep(truth, rollout)

    assert values.tolist() == [1.0, 4.0]


def test_save_rollout_dataset_writes_truth_and_rollout(tmp_path: Path):
    output_path = tmp_path / "rollout.nc"
    save_rollout_dataset(
        output_path,
        truth=np.ones((2, 2, 2), dtype=np.float32),
        rollout=np.zeros((2, 2, 2), dtype=np.float32),
        time_days=np.array([1.0, 2.0], dtype=np.float32),
        y=np.array([0.0, 1.0], dtype=np.float32),
        x=np.array([0.0, 1.0], dtype=np.float32),
    )

    dataset = xr.open_dataset(output_path)
    try:
        assert "truth_layer_thickness" in dataset
        assert "rollout_layer_thickness" in dataset
        assert "truth_zonal_velocity" not in dataset
    finally:
        dataset.close()


def test_save_rollout_dataset_writes_optional_velocity_fields(tmp_path: Path):
    output_path = tmp_path / "rollout.nc"
    save_rollout_dataset(
        output_path,
        truth=np.ones((2, 2, 2), dtype=np.float32),
        rollout=np.zeros((2, 2, 2), dtype=np.float32),
        time_days=np.array([1.0, 2.0], dtype=np.float32),
        y=np.array([0.0, 1.0], dtype=np.float32),
        x=np.array([0.0, 1.0], dtype=np.float32),
        truth_zonal_velocity=np.full((2, 2, 2), 2.0, dtype=np.float32),
        rollout_zonal_velocity=np.full((2, 2, 2), -2.0, dtype=np.float32),
        truth_meridional_velocity=np.full((2, 2, 2), 3.0, dtype=np.float32),
        rollout_meridional_velocity=np.full((2, 2, 2), -3.0, dtype=np.float32),
    )

    dataset = xr.open_dataset(output_path)
    try:
        assert "truth_zonal_velocity" in dataset
        assert "rollout_zonal_velocity" in dataset
        assert "truth_meridional_velocity" in dataset
        assert "rollout_meridional_velocity" in dataset
    finally:
        dataset.close()


def test_save_metrics_writes_json(tmp_path: Path):
    metrics_path = tmp_path / "metrics.json"
    save_metrics(metrics_path, {"mse": 1.23, "epochs": 5})

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert payload == {"mse": 1.23, "epochs": 5}
