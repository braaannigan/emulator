from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import xarray as xr

from .double_gyre_viewer import ExperimentRecord, select_time

EMULATOR_EXPERIMENT_ROOT = Path("data/raw/emulator/cnn_thickness")


@dataclass(frozen=True)
class EmulatorExperimentRecord:
    experiment_id: str
    rollout_path: Path
    metrics_path: Path | None


def list_emulator_experiments(root: Path = EMULATOR_EXPERIMENT_ROOT) -> list[EmulatorExperimentRecord]:
    experiments: list[EmulatorExperimentRecord] = []
    if not root.exists():
        return experiments

    for rollout_path in sorted(root.glob("*/rollout.nc"), reverse=True):
        metrics_path = rollout_path.parent / "metrics.json"
        experiments.append(
            EmulatorExperimentRecord(
                experiment_id=rollout_path.parent.name,
                rollout_path=rollout_path,
                metrics_path=metrics_path if metrics_path.exists() else None,
            )
        )
    return experiments


def open_rollout_dataset(rollout_path: Path) -> xr.Dataset:
    return xr.open_dataset(rollout_path)


def load_metrics(metrics_path: Path | None) -> dict[str, float | int | str]:
    if metrics_path is None:
        return {}
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def mse_timeseries(dataset: xr.Dataset) -> np.ndarray:
    truth = np.asarray(dataset["truth_layer_thickness"].values, dtype=float)
    rollout = np.asarray(dataset["rollout_layer_thickness"].values, dtype=float)
    return np.mean((truth - rollout) ** 2, axis=(1, 2))


def mse_timeseries_figure(dataset: xr.Dataset) -> go.Figure:
    mse_values = mse_timeseries(dataset)
    time_days = np.asarray(dataset["time_days"].values, dtype=float)
    figure = go.Figure(
        data=[
            go.Scatter(
                x=time_days,
                y=mse_values,
                mode="lines+markers",
                name="MSE",
                line={"color": "#b22222", "width": 3},
            )
        ]
    )
    figure.update_layout(
        title="Rollout MSE Over Time",
        xaxis_title="time_days",
        yaxis_title="mean_squared_error",
    )
    return figure


def comparison_heatmap_figure(
    dataset: xr.Dataset,
    field_name: str,
    title: str,
    zmin: float,
    zmax: float,
    show_colorbar: bool,
) -> go.Figure:
    data_array = dataset[field_name]
    y_dim = data_array.dims[-2]
    x_dim = data_array.dims[-1]
    figure = go.Figure(
        data=go.Heatmap(
            x=data_array[x_dim].values,
            y=data_array[y_dim].values,
            z=data_array.transpose(y_dim, x_dim).values,
            colorscale="RdBu_r",
            zmin=zmin,
            zmax=zmax,
            zauto=False,
            colorbar={"title": "layer_thickness"} if show_colorbar else None,
            showscale=show_colorbar,
        )
    )
    figure.update_layout(
        title=title,
        xaxis_title=x_dim,
        yaxis_title=y_dim,
    )
    return figure


def rollout_color_limits(dataset: xr.Dataset) -> tuple[float, float]:
    truth = np.asarray(dataset["truth_layer_thickness"].values, dtype=float)
    rollout = np.asarray(dataset["rollout_layer_thickness"].values, dtype=float)
    return float(min(truth.min(), rollout.min())), float(max(truth.max(), rollout.max()))
