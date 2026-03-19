from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr


def mean_squared_error(truth: np.ndarray, rollout: np.ndarray) -> float:
    return float(np.mean((truth - rollout) ** 2))


def mse_per_timestep(truth: np.ndarray, rollout: np.ndarray) -> np.ndarray:
    return np.mean((truth - rollout) ** 2, axis=(1, 2))


def save_rollout_dataset(
    rollout_path: Path,
    truth: np.ndarray,
    rollout: np.ndarray,
    time_days: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
) -> Path:
    rollout_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = xr.Dataset(
        data_vars={
            "truth_layer_thickness": (("time_days", "y", "x"), truth),
            "rollout_layer_thickness": (("time_days", "y", "x"), rollout),
        },
        coords={"time_days": time_days, "y": y, "x": x},
    )
    dataset.to_netcdf(rollout_path)
    return rollout_path


def save_metrics(metrics_path: Path, metrics: dict[str, float | int | str]) -> Path:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics_path
