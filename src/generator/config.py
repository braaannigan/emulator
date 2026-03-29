from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

SECONDS_PER_DAY = 86_400


@dataclass(frozen=True)
class DoubleGyreConfig:
    experiment_name: str
    nx: int
    ny: int
    layers: int
    dx_m: float
    dy_m: float
    dt_seconds: float
    duration_days: float
    output_interval_days: float
    viscosity: float
    thickness_diffusivity: float
    linear_drag: float
    slip: float
    hmin: float
    max_iterations: int
    pressure_tolerance: float
    free_surface_factor: float
    thickness_error: float
    debug_level: int
    mean_layer_thickness_m: float
    resting_depth_m: float
    reduced_gravity: float
    density: float
    f0: float
    beta: float
    run_directory: Path
    netcdf_output_path: Path
    wind_stress_max: float = 0.05
    wind_shift_amplitude_m: float = 0.0
    wind_shift_period_days: float | None = None
    wind_record_interval_hours: float | None = None
    executable_name: str = "aronnax_core"
    n_proc_x: int = 1
    n_proc_y: int = 1
    output_variables: tuple[str, ...] = ("snap.h", "snap.u", "snap.v")
    experiment_id: str | None = None

    @property
    def n_time_steps(self) -> int:
        total_seconds = self.duration_days * SECONDS_PER_DAY
        steps = total_seconds / self.dt_seconds
        if round(steps) != steps:
            raise ValueError("duration_days must align exactly with dt_seconds.")
        return int(steps)

    @property
    def dump_freq_seconds(self) -> float:
        dump_seconds = self.output_interval_days * SECONDS_PER_DAY
        steps = dump_seconds / self.dt_seconds
        if round(steps) != steps:
            raise ValueError("output_interval_days must align exactly with dt_seconds.")
        return dump_seconds

    @property
    def averaging_freq_seconds(self) -> float:
        return (self.n_time_steps + 1) * self.dt_seconds

    @property
    def expected_output_count(self) -> int:
        dump_steps = int(round(self.dump_freq_seconds / self.dt_seconds))
        return self.n_time_steps // dump_steps

    def with_overrides(self, **overrides: Any) -> "DoubleGyreConfig":
        normalized: dict[str, Any] = {}
        for key, value in overrides.items():
            if key in {"run_directory", "netcdf_output_path"} and value is not None:
                normalized[key] = Path(value)
            elif key == "output_variables" and value is not None:
                normalized[key] = tuple(value)
            else:
                normalized[key] = value
        return replace(self, **normalized)

    def resolve_experiment(self, experiment_id: str | None = None) -> "DoubleGyreConfig":
        resolved_id = experiment_id or self.experiment_id or timestamp_experiment_id()
        output_path = self.netcdf_output_path.parent / resolved_id / self.netcdf_output_path.name
        run_directory = self.run_directory / resolved_id
        return replace(
            self,
            experiment_id=resolved_id,
            netcdf_output_path=output_path,
            run_directory=run_directory,
        )


def load_double_gyre_config(path: str | Path) -> DoubleGyreConfig:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("double_gyre.yaml must contain a top-level mapping.")

    required_keys = {
        "nx",
        "ny",
        "layers",
        "dx_m",
        "dy_m",
        "dt_seconds",
        "duration_days",
        "output_interval_days",
        "viscosity",
        "thickness_diffusivity",
        "linear_drag",
        "slip",
        "hmin",
        "max_iterations",
        "pressure_tolerance",
        "free_surface_factor",
        "thickness_error",
        "debug_level",
        "mean_layer_thickness_m",
        "resting_depth_m",
        "reduced_gravity",
        "density",
        "f0",
        "beta",
        "experiment_name",
        "run_directory",
        "netcdf_output_path",
    }
    missing = sorted(required_keys - payload.keys())
    if missing:
        raise ValueError(f"double_gyre.yaml is missing required keys: {', '.join(missing)}")

    return DoubleGyreConfig(
        experiment_name=str(payload["experiment_name"]),
        nx=int(payload["nx"]),
        ny=int(payload["ny"]),
        layers=int(payload["layers"]),
        dx_m=float(payload["dx_m"]),
        dy_m=float(payload["dy_m"]),
        dt_seconds=float(payload["dt_seconds"]),
        duration_days=float(payload["duration_days"]),
        output_interval_days=float(payload["output_interval_days"]),
        viscosity=float(payload["viscosity"]),
        thickness_diffusivity=float(payload["thickness_diffusivity"]),
        linear_drag=float(payload["linear_drag"]),
        slip=float(payload["slip"]),
        hmin=float(payload["hmin"]),
        max_iterations=int(payload["max_iterations"]),
        pressure_tolerance=float(payload["pressure_tolerance"]),
        free_surface_factor=float(payload["free_surface_factor"]),
        thickness_error=float(payload["thickness_error"]),
        debug_level=int(payload["debug_level"]),
        mean_layer_thickness_m=float(payload["mean_layer_thickness_m"]),
        resting_depth_m=float(payload["resting_depth_m"]),
        reduced_gravity=float(payload["reduced_gravity"]),
        density=float(payload["density"]),
        f0=float(payload["f0"]),
        beta=float(payload["beta"]),
        wind_stress_max=float(payload.get("wind_stress_max", 0.05)),
        wind_shift_amplitude_m=float(payload.get("wind_shift_amplitude_m", 0.0)),
        wind_shift_period_days=(
            float(payload["wind_shift_period_days"])
            if payload.get("wind_shift_period_days") is not None
            else None
        ),
        wind_record_interval_hours=(
            float(payload["wind_record_interval_hours"])
            if payload.get("wind_record_interval_hours") is not None
            else None
        ),
        run_directory=Path(payload["run_directory"]),
        netcdf_output_path=Path(payload["netcdf_output_path"]),
        executable_name=str(payload.get("executable_name", "aronnax_core")),
        n_proc_x=int(payload.get("n_proc_x", 1)),
        n_proc_y=int(payload.get("n_proc_y", 1)),
        output_variables=tuple(payload.get("output_variables", ("snap.h", "snap.u", "snap.v"))),
        experiment_id=payload.get("experiment_id"),
    )


def timestamp_experiment_id(now: datetime | None = None) -> str:
    current = now or datetime.now()
    return current.strftime("%Y%m%dT%H%M%S")
