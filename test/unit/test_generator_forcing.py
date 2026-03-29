import math

import numpy as np

from src.generator.config import load_double_gyre_config
from src.generator.double_gyre import build_zonal_wind
from src.generator.forcing import build_shifting_double_gyre_wind, build_static_double_gyre_wind


def test_build_static_double_gyre_wind_matches_expected_profile():
    config = load_double_gyre_config("config/generator/double_gyre.yaml")
    forcing = build_static_double_gyre_wind(config)
    y = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

    values = forcing(None, y)

    assert np.allclose(values[:, 0], [0.0, 0.1, 0.0])


def test_build_shifting_double_gyre_wind_returns_time_varying_records():
    config = load_double_gyre_config("config/generator/double_gyre_shifting_wind.yaml")
    forcing = build_shifting_double_gyre_wind(config)
    y = np.tile(np.linspace(0.0, 2.0e6, 4).reshape(4, 1), (1, 2))

    values = forcing(None, y, 4)

    assert values.shape == (4, 4, 2)
    assert not np.allclose(values[0], values[1])


def test_build_zonal_wind_uses_shifting_options_for_shifting_experiment():
    config = load_double_gyre_config("config/generator/double_gyre_shifting_wind.yaml")

    forcing, options = build_zonal_wind(config)

    assert callable(forcing)
    assert options["wind_loop_fields"] is True
    assert options["wind_interpolate"] is True
    assert math.isclose(options["wind_period"], config.dt_seconds)


def test_build_zonal_wind_accepts_coarser_snapshot_interval():
    config = load_double_gyre_config("config/generator/double_gyre_shifting_wind.yaml").with_overrides(
        wind_shift_period_days=100.0,
        wind_record_interval_hours=12.0,
    )

    _, options = build_zonal_wind(config)

    assert options["wind_n_records"] == 200
    assert math.isclose(options["wind_period"], 12.0 * 3600.0)


def test_build_zonal_wind_treats_double_gyre_2x_fine_as_static_family():
    config = load_double_gyre_config("config/generator/double_gyre_2x_fine.yaml")

    forcing, options = build_zonal_wind(config)

    assert callable(forcing)
    assert options == {}
