from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


DEFAULT_FIELDS = ("layer_thickness", "zonal_velocity", "meridional_velocity")


def default_animation_path(netcdf_path: Path) -> Path:
    return netcdf_path.with_suffix(".mp4")


def animation_output_path(netcdf_path: Path, output_path: str | Path | None = None) -> Path:
    if output_path is None:
        return default_animation_path(netcdf_path)
    return Path(output_path)


def compute_color_limits(dataset: xr.Dataset, field_name: str) -> tuple[float, float]:
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
        color_limits = {
            field_name: compute_color_limits(dataset, field_name)
            for field_name in fields
            if field_name in dataset
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
    fields = [field_name for field_name in DEFAULT_FIELDS if field_name in dataset_at_time]
    figure, axes = plt.subplots(1, len(fields), figsize=(6 * len(fields), 5), constrained_layout=True)
    if len(fields) == 1:
        axes = [axes]

    for axis, field_name in zip(axes, fields):
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


def spatial_dims(data_array: xr.DataArray) -> tuple[str, str]:
    dims = [dim for dim in data_array.dims if dim not in {"time_days", "layers"}]
    if len(dims) != 2:
        raise ValueError(f"Expected exactly two spatial dims, found {dims}.")
    return dims[0], dims[1]
