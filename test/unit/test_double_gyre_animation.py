from pathlib import Path

import numpy as np
import xarray as xr

from src.animation.double_gyre_animation import (
    SYNTHETIC_WIND_FIELD,
    animation_output_path,
    compute_color_limits,
    resolved_fields,
    wind_stress_series,
)


def test_animation_output_path_defaults_to_same_directory():
    netcdf_path = Path("data/raw/double_gyre/generator/20260319T180000/double_gyre.nc")
    output_path = animation_output_path(netcdf_path)

    assert output_path == Path("data/raw/double_gyre/generator/20260319T180000/double_gyre.mp4")


def test_velocity_color_limits_are_symmetric():
    dataset = xr.Dataset(
        {"zonal_velocity": (("time_days", "y", "x"), np.array([[[-2.0, 1.0], [0.5, 3.0]]]))},
        coords={"time_days": [0.0], "y": [0.0, 1.0], "x": [0.0, 1.0]},
    )

    vmin, vmax = compute_color_limits(dataset, "zonal_velocity")

    assert vmin == -3.0
    assert vmax == 3.0


def test_resolved_fields_includes_synthetic_wind_for_shifting_experiment():
    dataset = xr.Dataset(
        {"layer_thickness": (("time_days", "y", "x"), np.ones((2, 2, 2), dtype=float))},
        coords={"time_days": [0.0, 20.0], "y": [0.0, 1.0], "x": [0.0, 1.0]},
        attrs={
            "experiment": "double_gyre_shifting_wind",
            "wind_stress_max": "0.05",
            "wind_shift_amplitude_m": "1.0",
            "wind_shift_period_days": "40.0",
        },
    )

    fields = resolved_fields(dataset, ("layer_thickness",))

    assert fields == ("layer_thickness", SYNTHETIC_WIND_FIELD)


def test_wind_stress_series_varies_in_time_for_shifting_experiment():
    dataset = xr.Dataset(
        coords={"time_days": [0.0, 10.0], "y": [0.0, 1.0, 2.0], "x": [0.0, 1.0]},
        attrs={
            "experiment": "double_gyre_shifting_wind",
            "wind_stress_max": "0.05",
            "wind_shift_amplitude_m": "0.5",
            "wind_shift_period_days": "40.0",
        },
    )

    wind = wind_stress_series(dataset)

    assert wind.shape == (2, 3, 2)
    assert not np.allclose(wind[0], wind[1])
