from __future__ import annotations

import shutil
from pathlib import Path

from .aronnax_runtime import load_aronnax_modules
from .config import DoubleGyreConfig, load_double_gyre_config
from .netcdf_writer import write_netcdf_output
from .runner import prepare_run_directory, write_aronnax_configuration


def _require_aronnax_driver():
    try:
        return load_aronnax_modules()
    except Exception as exc:
        raise RuntimeError("Aronnax must be available from the pinned source checkout.") from exc


def build_zonal_wind(config: DoubleGyreConfig):
    aro, _ = _require_aronnax_driver()
    grid = aro.Grid(config.nx, config.ny, config.layers, config.dx_m, config.dy_m)

    def wind(_, y_coordinates):
        import numpy as np

        return 0.05 * (1 - np.cos(2 * np.pi * y_coordinates / grid.y.max()))

    return wind


def run_double_gyre_pipeline(config: DoubleGyreConfig | str | Path) -> Path:
    if not isinstance(config, DoubleGyreConfig):
        config = load_double_gyre_config(config)
    config = config.resolve_experiment()

    _, driver = _require_aronnax_driver()

    prepare_run_directory(config)
    conf_path = write_aronnax_configuration(config, config.run_directory / "aronnax.conf")

    driver.simulate(
        work_dir=str(config.run_directory),
        config_path=conf_path.name,
        zonal_wind_file=[build_zonal_wind(config)],
        exe=config.executable_name,
    )

    output_path = write_netcdf_output(config)
    shutil.rmtree(config.run_directory / "netcdf-output", ignore_errors=True)
    return output_path
