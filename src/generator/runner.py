from __future__ import annotations

import configparser
import shutil
from pathlib import Path

from .config import DoubleGyreConfig


def _layer_vector(value: float, layers: int) -> str:
    return ",".join([str(value)] * max(layers, 1))


def write_aronnax_configuration(config: DoubleGyreConfig, destination: Path) -> Path:
    parser = configparser.RawConfigParser()
    parser.optionxform = str

    parser["executable"] = {"exe": config.executable_name}
    parser["numerics"] = {
        "au": str(config.viscosity),
        "kh": str(config.thickness_diffusivity),
        "ar": str(config.linear_drag),
        "dt": str(config.dt_seconds),
        "slip": str(config.slip),
        "n_time_steps": str(config.n_time_steps),
        "dump_freq": str(config.dump_freq_seconds),
        "av_freq": str(config.averaging_freq_seconds),
        "diag_freq": str(config.dump_freq_seconds),
        "hmin": str(config.hmin),
        "maxits": str(config.max_iterations),
        "eps": str(config.pressure_tolerance),
        "freesurf_fac": str(config.free_surface_factor),
        "thickness_error": str(config.thickness_error),
        "debug_level": str(config.debug_level),
    }
    parser["model"] = {
        "hmean": _layer_vector(config.mean_layer_thickness_m, config.layers),
        "h0": str(config.resting_depth_m),
        "red_grav": "yes",
    }
    parser["pressure_solver"] = {
        "n_proc_x": str(config.n_proc_x),
        "n_proc_y": str(config.n_proc_y),
    }
    parser["physics"] = {
        "g_vec": _layer_vector(config.reduced_gravity, config.layers),
        "rho0": str(config.density),
    }
    parser["grid"] = {
        "nx": str(config.nx),
        "ny": str(config.ny),
        "layers": str(config.layers),
        "dx": str(config.dx_m),
        "dy": str(config.dy_m),
        "f_u_file": f":beta_plane_f_u:{config.f0},{config.beta}",
        "f_v_file": f":beta_plane_f_v:{config.f0},{config.beta}",
        "wet_mask_file": ":rectangular_pool:",
    }
    parser["initial_conditions"] = {
        "init_h_file": f":tracer_point_variable:{_layer_vector(config.mean_layer_thickness_m, config.layers)}",
    }
    parser["external_forcing"] = {
        "dump_wind": "no",
        "relative_wind": "no",
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        parser.write(handle)
    return destination


def prepare_run_directory(config: DoubleGyreConfig) -> Path:
    run_dir = config.run_directory
    if run_dir.exists():
        shutil.rmtree(run_dir)
    (run_dir / "input").mkdir(parents=True, exist_ok=True)
    (run_dir / "output").mkdir(parents=True, exist_ok=True)
    return run_dir
