from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


DEFAULT_FIELDS = ("layer_thickness", "zonal_velocity", "meridional_velocity")
SYNTHETIC_WIND_FIELD = "zonal_wind_stress"


def default_animation_path(netcdf_path: Path) -> Path:
    return netcdf_path.with_suffix(".mp4")


def animation_output_path(netcdf_path: Path, output_path: str | Path | None = None) -> Path:
    if output_path is None:
        return default_animation_path(netcdf_path)
    return Path(output_path)


def compute_color_limits(dataset: xr.Dataset, field_name: str) -> tuple[float, float]:
    if field_name == SYNTHETIC_WIND_FIELD:
        wind = wind_stress_series(dataset)
        return float(np.nanmin(wind)), float(np.nanmax(wind))
    data = np.asarray(dataset[field_name].values, dtype=float)
    if field_name in {"zonal_velocity", "meridional_velocity"}:
        max_abs = float(np.nanmax(np.abs(data)))
        return -max_abs, max_abs
    return float(np.nanmin(data)), float(np.nanmax(data))


def create_animation(
    netcdf_path: str | Path,
    output_path: str | Path | None = None,
    fps: int = 4,
    fields: tuple[str, ...] = DEFAULT_FIELDS,
) -> Path:
    netcdf_path = Path(netcdf_path)
    output_path = animation_output_path(netcdf_path, output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = xr.open_dataset(netcdf_path)
    try:
        fields = resolved_fields(dataset, fields)
        color_limits = {
            field_name: compute_color_limits(dataset, field_name)
            for field_name in fields
            if field_name in dataset or field_name == SYNTHETIC_WIND_FIELD
        }
        if not color_limits:
            raise ValueError(f"No requested fields {fields} found in {netcdf_path}.")

        with imageio.get_writer(output_path, fps=fps) as writer:
            for time_day in dataset["time_days"].values.tolist():
                frame = render_frame(dataset.sel(time_days=time_day), float(time_day), color_limits)
                writer.append_data(frame)
    finally:
        dataset.close()

    return output_path


def render_frame(
    dataset_at_time: xr.Dataset,
    time_day: float,
    color_limits: dict[str, tuple[float, float]],
) -> np.ndarray:
    fields = [field_name for field_name in color_limits]
    figure, axes = plt.subplots(1, len(fields), figsize=(6 * len(fields), 5), constrained_layout=True)
    if len(fields) == 1:
        axes = [axes]

    for axis, field_name in zip(axes, fields):
        if field_name == SYNTHETIC_WIND_FIELD:
            render_wind_stress_panel(axis, dataset_at_time, color_limits[field_name])
            continue
        data_array = dataset_at_time[field_name]
        if "layers" in data_array.dims:
            data_array = data_array.sel(layers=0)

        y_dim, x_dim = spatial_dims(data_array)
        vmin, vmax = color_limits[field_name]
        mesh = axis.pcolormesh(
            data_array[x_dim].values,
            data_array[y_dim].values,
            data_array.transpose(y_dim, x_dim).values,
            cmap="RdBu_r",
            shading="auto",
            vmin=vmin,
            vmax=vmax,
        )
        figure.colorbar(mesh, ax=axis)
        if field_name in {"zonal_velocity", "meridional_velocity"} and "layer_thickness" in dataset_at_time:
            contour_array = dataset_at_time["layer_thickness"]
            if "layers" in contour_array.dims:
                contour_array = contour_array.sel(layers=0)
            contour_y_dim, contour_x_dim = spatial_dims(contour_array)
            axis.contour(
                contour_array[contour_x_dim].values,
                contour_array[contour_y_dim].values,
                contour_array.transpose(contour_y_dim, contour_x_dim).values,
                colors="black",
                linewidths=0.7,
            )
        axis.set_title(field_name)
        axis.set_xlabel(x_dim)
        axis.set_ylabel(y_dim)

    figure.suptitle(f"Double gyre at {time_day:.3f} days")
    figure.canvas.draw()
    frame = np.asarray(figure.canvas.buffer_rgba())[..., :3]
    plt.close(figure)
    return frame


def resolved_fields(dataset: xr.Dataset, requested_fields: tuple[str, ...]) -> tuple[str, ...]:
    fields = tuple(field_name for field_name in requested_fields if field_name in dataset)
    if can_derive_wind_stress(dataset) and SYNTHETIC_WIND_FIELD not in fields:
        fields = (fields[0], SYNTHETIC_WIND_FIELD, *fields[1:]) if "layer_thickness" in fields else (*fields, SYNTHETIC_WIND_FIELD)
    return fields


def can_derive_wind_stress(dataset: xr.Dataset) -> bool:
    attrs = dataset.attrs
    return (
        "y" in dataset.coords
        and attrs.get("experiment") == "double_gyre_shifting_wind"
        and "wind_stress_max" in attrs
        and "wind_shift_amplitude_m" in attrs
        and "wind_shift_period_days" in attrs
    )


def wind_stress_series(dataset: xr.Dataset) -> np.ndarray:
    if not can_derive_wind_stress(dataset):
        raise ValueError("Dataset does not contain enough metadata to derive shifting wind stress.")

    time_days = np.asarray(dataset["time_days"].values, dtype=float)
    y = np.asarray(dataset["y"].values, dtype=float)
    x = np.asarray(dataset["x"].values, dtype=float)
    y_grid = np.repeat(y[:, np.newaxis], x.size, axis=1)
    y_max = float(np.max(y))
    wind_stress_max = float(dataset.attrs["wind_stress_max"])
    shift_amplitude_m = float(dataset.attrs["wind_shift_amplitude_m"])
    shift_period_days = float(dataset.attrs["wind_shift_period_days"])

    output = np.empty((time_days.size, y.size, x.size), dtype=float)
    for index, time_day in enumerate(time_days):
        phase = 2.0 * np.pi * (time_day / shift_period_days)
        shift_m = shift_amplitude_m * np.sin(phase)
        shifted_y = np.clip(y_grid - shift_m, 0.0, y_max)
        output[index] = wind_stress_max * (1.0 - np.cos(2.0 * np.pi * shifted_y / y_max))
    return output


def wind_stress_at_time(dataset_at_time: xr.Dataset) -> np.ndarray:
    series = wind_stress_series(dataset_at_time.expand_dims("time_days"))
    return series[0]


def render_wind_stress_panel(axis, dataset_at_time: xr.Dataset, color_limits: tuple[float, float]) -> None:
    wind = wind_stress_at_time(dataset_at_time)
    y = dataset_at_time["y"].values
    x = dataset_at_time["x"].values
    vmin, vmax = color_limits
    mesh = axis.pcolormesh(
        x,
        y,
        wind,
        cmap="viridis",
        shading="auto",
        vmin=vmin,
        vmax=vmax,
    )
    axis.figure.colorbar(mesh, ax=axis)
    axis.set_title(SYNTHETIC_WIND_FIELD)
    axis.set_xlabel("x")
    axis.set_ylabel("y")


def spatial_dims(data_array: xr.DataArray) -> tuple[str, str]:
    dims = [dim for dim in data_array.dims if dim not in {"time_days", "layers"}]
    if len(dims) != 2:
        raise ValueError(f"Expected exactly two spatial dims, found {dims}.")
    return dims[0], dims[1]
