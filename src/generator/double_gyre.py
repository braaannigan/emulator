from __future__ import annotations

import shutil
from pathlib import Path

from .aronnax_runtime import load_aronnax_modules
from .config import SECONDS_PER_DAY
from .config import DoubleGyreConfig, load_double_gyre_config
from .forcing import build_shifting_double_gyre_wind, build_static_double_gyre_wind
from .netcdf_writer import write_netcdf_output
from .runner import prepare_run_directory, write_aronnax_configuration


def _require_aronnax_driver():
    try:
        return load_aronnax_modules()
    except Exception as exc:
        raise RuntimeError("Aronnax must be available from the pinned source checkout.") from exc


def build_zonal_wind(config: DoubleGyreConfig):
    if config.experiment_name.startswith("double_gyre_shifting_wind"):
        forcing_cycle_seconds = config.wind_shift_period_days * SECONDS_PER_DAY
        snapshot_seconds = config.dt_seconds
        if config.wind_record_interval_hours is not None:
            snapshot_seconds = config.wind_record_interval_hours * 3600.0
        total_records_float = forcing_cycle_seconds / snapshot_seconds
        if round(total_records_float) != total_records_float:
            raise ValueError("wind_shift_period_days must align exactly with the wind record interval.")
        total_records = max(2, int(total_records_float))
        return build_shifting_double_gyre_wind(config), {
            "wind_n_records": total_records,
            # Aronnax expects the time between adjacent wind snapshots here, not the full cycle length.
            "wind_period": snapshot_seconds,
            "wind_loop_fields": True,
            "wind_interpolate": True,
        }
    if config.experiment_name.startswith("double_gyre"):
        return build_static_double_gyre_wind(config), {}
    raise ValueError(f"Unsupported experiment_name {config.experiment_name!r}")


def run_double_gyre_pipeline(config: DoubleGyreConfig | str | Path) -> Path:
    if not isinstance(config, DoubleGyreConfig):
        config = load_double_gyre_config(config)
    config = config.resolve_experiment()

    _, driver = _require_aronnax_driver()

    prepare_run_directory(config)
    conf_path = write_aronnax_configuration(config, config.run_directory / "aronnax.conf")
    zonal_wind, wind_options = build_zonal_wind(config)

    driver.simulate(
        work_dir=str(config.run_directory),
        config_path=conf_path.name,
        zonal_wind_file=[zonal_wind],
        exe=config.executable_name,
        **wind_options,
    )

    output_path = write_netcdf_output(config)
    shutil.rmtree(config.run_directory / "netcdf-output", ignore_errors=True)
    return output_path
