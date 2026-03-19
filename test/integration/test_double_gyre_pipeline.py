import shutil
from pathlib import Path

import pytest
import xarray as xr

from src.generator.aronnax_runtime import ensure_aronnax_checkout
from src.generator.config import load_double_gyre_config
from src.generator.double_gyre import run_double_gyre_pipeline


def _has_runtime_dependencies() -> bool:
    return all(shutil.which(command) for command in ("git", "mpif90", "mpirun"))


@pytest.mark.integration
def test_short_double_gyre_run_writes_test_netcdf():
    if not _has_runtime_dependencies():
        pytest.skip("Aronnax runtime requires git plus mpif90 and mpirun.")

    ensure_aronnax_checkout()

    output_path = Path("data/raw/double_gyre/test_double_gyre.nc")
    run_directory = Path("data/interim/test_double_gyre_run")
    if output_path.exists():
        output_path.unlink()
    if run_directory.exists():
        shutil.rmtree(run_directory)

    config = load_double_gyre_config("config/generator/double_gyre.yaml").with_overrides(
        nx=10,
        ny=10,
        dx_m=20_000.0,
        dy_m=20_000.0,
        duration_days=1.0,
        output_interval_days=0.5,
        max_iterations=1000,
        run_directory=Path("data/interim/test_double_gyre_run"),
        netcdf_output_path=Path("data/raw/double_gyre/test_double_gyre.nc"),
        experiment_id="integration-test",
    )

    try:
        written_path = run_double_gyre_pipeline(config)
        dataset = xr.open_dataset(written_path)

        assert written_path == output_path.parent / "integration-test" / output_path.name
        assert written_path.exists()
        assert dataset["time_days"].size >= 2
        assert "layer_thickness" in dataset
        dataset.close()
    finally:
        experiment_output_dir = output_path.parent / "integration-test"
        experiment_run_dir = run_directory / "integration-test"
        if experiment_output_dir.exists():
            shutil.rmtree(experiment_output_dir)
        if experiment_run_dir.exists():
            shutil.rmtree(experiment_run_dir)
