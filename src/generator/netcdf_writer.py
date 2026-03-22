from __future__ import annotations

import glob
from pathlib import Path

import xarray as xr

from .aronnax_runtime import load_aronnax_modules
from .config import DoubleGyreConfig, SECONDS_PER_DAY


RAW_VARIABLES = {
    "snap.h": "layer_thickness",
    "snap.u": "zonal_velocity",
    "snap.v": "meridional_velocity",
}


def _require_aronnax():
    try:
        aro, _ = load_aronnax_modules()
    except Exception as exc:
        raise RuntimeError("Aronnax is required to read model output.") from exc
    return aro


def write_netcdf_output(config: DoubleGyreConfig) -> Path:
    aro = _require_aronnax()
    run_output_dir = config.run_directory / "output"
    grid = aro.Grid(config.nx, config.ny, config.layers, config.dx_m, config.dy_m)

    dataset = xr.Dataset(attrs=_dataset_attrs(config))

    for raw_name in config.output_variables:
        files = sorted(glob.glob(str(run_output_dir / f"{raw_name}.*")))
        if not files:
            continue

        data_array = aro.open_mfdataarray(files, grid)
        time_days = (data_array["iter"].values * config.dt_seconds) / SECONDS_PER_DAY
        data_array = data_array.assign_coords(time_days=("iter", time_days)).swap_dims({"iter": "time_days"})
        data_array = data_array.drop_vars("iter")
        variable_name = RAW_VARIABLES.get(raw_name, raw_name.replace(".", "_"))
        dataset[variable_name] = data_array.rename(variable_name)

    if not dataset.data_vars:
        raise FileNotFoundError(f"No supported Aronnax output files found in {run_output_dir}.")

    output_path = config.netcdf_output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(output_path)
    return output_path


def _dataset_attrs(config: DoubleGyreConfig) -> dict[str, str]:
    return {
        "model": "Aronnax",
        "experiment": config.experiment_name,
        "duration_days": str(config.duration_days),
        "output_interval_days": str(config.output_interval_days),
        "dt_seconds": str(config.dt_seconds),
    }
