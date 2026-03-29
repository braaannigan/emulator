from pathlib import Path

from src.generator.config import load_double_gyre_config


def test_double_gyre_config_loads_experiment_name_and_wind_fields():
    config = load_double_gyre_config("config/generator/double_gyre_shifting_wind.yaml")

    assert config.experiment_name == "double_gyre_shifting_wind"
    assert config.wind_shift_amplitude_m == 40000.0
    assert config.wind_shift_period_days == 40.0
    assert config.wind_record_interval_hours is None


def test_resolve_experiment_keeps_new_experiment_output_root():
    config = load_double_gyre_config("config/generator/double_gyre_shifting_wind.yaml")
    resolved = config.resolve_experiment("demo")

    assert resolved.netcdf_output_path == Path("data/raw/double_gyre_shifting_wind/generator/demo/double_gyre.nc")
    assert resolved.run_directory == Path("data/interim/double_gyre_shifting_wind_run/demo")


def test_double_gyre_2x_fine_config_uses_finer_grid_and_new_output_root():
    config = load_double_gyre_config("config/generator/double_gyre_2x_fine.yaml")
    resolved = config.resolve_experiment("demo")

    assert config.experiment_name == "double_gyre_2x_fine"
    assert config.nx == 200
    assert config.ny == 400
    assert config.dx_m == 5000.0
    assert config.dy_m == 5000.0
    assert resolved.netcdf_output_path == Path("data/raw/double_gyre_2x_fine/generator/demo/double_gyre.nc")
    assert resolved.run_directory == Path("data/interim/double_gyre_2x_fine_run/demo")
