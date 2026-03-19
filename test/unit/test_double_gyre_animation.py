from pathlib import Path

import numpy as np
import xarray as xr

from src.animation.double_gyre_animation import animation_output_path, compute_color_limits


def test_animation_output_path_defaults_to_same_directory():
    netcdf_path = Path("data/raw/double_gyre/20260319T180000/double_gyre.nc")
    output_path = animation_output_path(netcdf_path)

    assert output_path == Path("data/raw/double_gyre/20260319T180000/double_gyre.mp4")


def test_velocity_color_limits_are_symmetric():
    dataset = xr.Dataset(
        {"zonal_velocity": (("time_days", "y", "x"), np.array([[[-2.0, 1.0], [0.5, 3.0]]]))},
        coords={"time_days": [0.0], "y": [0.0, 1.0], "x": [0.0, 1.0]},
    )

    vmin, vmax = compute_color_limits(dataset, "zonal_velocity")

    assert vmin == -3.0
    assert vmax == 3.0
