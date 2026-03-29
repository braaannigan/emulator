from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import xarray as xr

RAW_EXPERIMENTS_ROOT = Path("data/raw")


@dataclass(frozen=True)
class ExperimentRecord:
    physical_experiment_name: str
    experiment_id: str
    netcdf_path: Path


def list_experiments(root: Path = RAW_EXPERIMENTS_ROOT) -> list[ExperimentRecord]:
    if _looks_like_generator_root(root):
        return _list_experiments_from_generator_root(root)

    experiments: list[ExperimentRecord] = []
    if not root.exists():
        return experiments

    for physical_root in sorted(path for path in root.iterdir() if path.is_dir()):
        generator_root = physical_root / "generator"
        experiments.extend(_list_experiments_from_generator_root(generator_root))
    return sorted(
        experiments,
        key=lambda experiment: (experiment.experiment_id, experiment.physical_experiment_name),
        reverse=True,
    )


def _list_experiments_from_generator_root(root: Path) -> list[ExperimentRecord]:
    experiments: list[ExperimentRecord] = []
    if not root.exists():
        return experiments

    for netcdf_path in sorted(root.glob("*/double_gyre.nc"), reverse=True):
        experiments.append(
            ExperimentRecord(
                physical_experiment_name=root.parent.name,
                experiment_id=netcdf_path.parent.name,
                netcdf_path=netcdf_path,
            )
        )
    return experiments


def open_experiment_dataset(netcdf_path: Path) -> xr.Dataset:
    return xr.open_dataset(netcdf_path)


def select_time(dataset: xr.Dataset, requested_day: float) -> tuple[float, xr.Dataset]:
    available_days = dataset["time_days"].values.astype(float)
    index = int(np.abs(available_days - requested_day).argmin())
    selected_day = float(available_days[index])
    selected_dataset = dataset.sel(time_days=selected_day)
    return selected_day, selected_dataset


def available_fields(dataset: xr.Dataset) -> list[str]:
    return list(dataset.data_vars)


def available_layers(dataset: xr.Dataset) -> list[int]:
    if "layers" not in dataset.dims:
        return [0]
    return [int(layer) for layer in dataset["layers"].values.tolist()]


def field_to_heatmap(
    dataset: xr.Dataset,
    field_name: str,
    layer_index: int = 0,
) -> go.Figure:
    data_array = dataset[field_name]
    if "layers" in data_array.dims:
        data_array = data_array.sel(layers=layer_index)

    y_dim, x_dim = _spatial_dims(data_array)
    heatmap_kwargs = {
        "x": data_array[x_dim].values,
        "y": data_array[y_dim].values,
        "z": data_array.transpose(y_dim, x_dim).values,
        "colorscale": "RdBu_r",
        "colorbar": {"title": field_name},
    }
    if field_name in {"zonal_velocity", "meridional_velocity"}:
        max_abs = float(np.nanmax(np.abs(data_array.values)))
        heatmap_kwargs["zmin"] = -max_abs
        heatmap_kwargs["zmax"] = max_abs

    figure = go.Figure(
        data=go.Heatmap(**heatmap_kwargs)
    )
    if field_name in {"zonal_velocity", "meridional_velocity"} and "layer_thickness" in dataset:
        contour_array = dataset["layer_thickness"]
        if "layers" in contour_array.dims:
            contour_array = contour_array.sel(layers=layer_index)
        contour_y_dim, contour_x_dim = _spatial_dims(contour_array)
        figure.add_trace(
            go.Contour(
                x=contour_array[contour_x_dim].values,
                y=contour_array[contour_y_dim].values,
                z=contour_array.transpose(contour_y_dim, contour_x_dim).values,
                contours={
                    "coloring": "none",
                    "showlabels": False,
                },
                line={"color": "black", "width": 1},
                showscale=False,
                hoverinfo="skip",
            )
        )
    figure.update_layout(
        title=field_name,
        xaxis_title=x_dim,
        yaxis_title=y_dim,
    )
    return figure


def wind_stress_figure(dataset: xr.Dataset) -> go.Figure:
    if "y" not in dataset.coords:
        raise ValueError("Dataset must contain a y coordinate to derive wind stress.")

    y = np.asarray(dataset["y"].values, dtype=float)
    wind_stress_max = float(dataset.attrs.get("wind_stress_max", 0.05))
    wind_stress = wind_stress_max * (1.0 - np.cos(2.0 * np.pi * y / np.max(y)))
    experiment_name = str(dataset.attrs.get("experiment", "double_gyre"))
    figure = go.Figure(
        data=[
            go.Scatter(
                x=y,
                y=wind_stress,
                mode="lines",
                line={"color": "#b22222", "width": 3},
                name="zonal_wind_stress",
            )
        ]
    )
    figure.update_layout(
        title=f"Zonal Wind Stress ({experiment_name})",
        xaxis_title="y",
        yaxis_title="wind_stress",
    )
    return figure


def _looks_like_generator_root(root: Path) -> bool:
    return any(root.glob("*/double_gyre.nc"))


def _spatial_dims(data_array: xr.DataArray) -> tuple[str, str]:
    spatial_dims = [dim for dim in data_array.dims if dim not in {"time_days", "layers"}]
    if len(spatial_dims) != 2:
        raise ValueError(f"Expected exactly two spatial dims for heatmap, found {spatial_dims}.")
    return spatial_dims[0], spatial_dims[1]
