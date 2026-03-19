from pathlib import Path

import numpy as np
import xarray as xr

from src.generator.config import load_double_gyre_config
from src.generator.netcdf_writer import write_netcdf_output


class FakeAronnaxModule:
    class Grid:
        def __init__(self, nx, ny, layers, dx, dy):
            self.nx = nx
            self.ny = ny
            self.layers = layers
            self.dx = dx
            self.dy = dy

    @staticmethod
    def open_mfdataarray(files, grid):
        base = Path(files[0]).name.split(".")
        variable = ".".join(base[:-1])
        shape = {
            "snap.h": ("iter", "layers", "y", "x"),
            "snap.u": ("iter", "layers", "y", "xp1"),
            "snap.v": ("iter", "layers", "yp1", "x"),
        }[variable]
        coords = {
            "iter": np.array([72.0, 144.0]),
            "layers": np.array([0]),
            "y": np.array([0.5, 1.5]),
            "x": np.array([0.5, 1.5]),
            "xp1": np.array([0.0, 1.0, 2.0]),
            "yp1": np.array([0.0, 1.0, 2.0]),
        }
        sizes = tuple(len(coords[dim]) for dim in shape)
        data = np.ones(sizes)
        return xr.DataArray(data=data, dims=shape, coords={dim: coords[dim] for dim in shape}, name=variable)


def test_write_netcdf_output_builds_time_stacked_file(tmp_path: Path, monkeypatch):
    run_dir = tmp_path / "run"
    output_dir = run_dir / "output"
    output_dir.mkdir(parents=True)
    for file_name in [
        "snap.h.0000072",
        "snap.h.0000144",
        "snap.u.0000072",
        "snap.u.0000144",
        "snap.v.0000072",
        "snap.v.0000144",
    ]:
        (output_dir / file_name).write_bytes(b"")

    monkeypatch.setattr("src.generator.netcdf_writer._require_aronnax", lambda: FakeAronnaxModule)

    config = load_double_gyre_config("config/generator/double_gyre.yaml").with_overrides(
        nx=2,
        ny=2,
        run_directory=run_dir,
        netcdf_output_path=tmp_path / "double_gyre.nc",
    )

    output_path = write_netcdf_output(config)
    dataset = xr.open_dataset(output_path)

    assert output_path.exists()
    assert set(dataset.data_vars) == {"layer_thickness", "zonal_velocity", "meridional_velocity"}
    assert np.allclose(dataset["time_days"].values, np.array([0.5, 1.0]))
    dataset.close()
