from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import xarray as xr


def mean_squared_error(truth: np.ndarray, rollout: np.ndarray) -> float:
    return float(np.mean((truth - rollout) ** 2))


def mse_per_timestep(truth: np.ndarray, rollout: np.ndarray) -> np.ndarray:
    axes = tuple(range(1, truth.ndim))
    return np.mean((truth - rollout) ** 2, axis=axes)


def save_rollout_dataset(
    rollout_path: Path,
    truth: np.ndarray,
    rollout: np.ndarray,
    time_days: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    truth_zonal_velocity: np.ndarray | None = None,
    rollout_zonal_velocity: np.ndarray | None = None,
    truth_meridional_velocity: np.ndarray | None = None,
    rollout_meridional_velocity: np.ndarray | None = None,
) -> Path:
    rollout_path.parent.mkdir(parents=True, exist_ok=True)
    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {
        "truth_layer_thickness": (("time_days", "y", "x"), truth),
        "rollout_layer_thickness": (("time_days", "y", "x"), rollout),
    }
    if truth_zonal_velocity is not None and rollout_zonal_velocity is not None:
        data_vars["truth_zonal_velocity"] = (("time_days", "y", "x"), truth_zonal_velocity)
        data_vars["rollout_zonal_velocity"] = (("time_days", "y", "x"), rollout_zonal_velocity)
    if truth_meridional_velocity is not None and rollout_meridional_velocity is not None:
        data_vars["truth_meridional_velocity"] = (("time_days", "y", "x"), truth_meridional_velocity)
        data_vars["rollout_meridional_velocity"] = (("time_days", "y", "x"), rollout_meridional_velocity)
    dataset = xr.Dataset(
        data_vars=data_vars,
        coords={"time_days": time_days, "y": y, "x": x},
    )
    dataset.to_netcdf(rollout_path)
    return rollout_path


def save_metrics(metrics_path: Path, metrics: dict[str, float | int | str]) -> Path:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics["updated_at"] = datetime.now(timezone.utc).isoformat()
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics_path
