from __future__ import annotations

import shutil
from pathlib import Path

from .aronnax_runtime import load_aronnax_modules
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
    if config.experiment_name == "double_gyre":
        return build_static_double_gyre_wind(config), {}
    if config.experiment_name == "double_gyre_shifting_wind":
        total_records = max(2, config.n_time_steps)
        return build_shifting_double_gyre_wind(config), {
            "wind_n_records": total_records,
            "wind_period": config.wind_shift_period_days * 86_400,
            "wind_loop_fields": True,
            "wind_interpolate": True,
        }
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
