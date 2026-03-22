import shutil
from pathlib import Path

import pytest
import xarray as xr

from src.generator.aronnax_runtime import ensure_aronnax_checkout
from src.generator.config import SECONDS_PER_DAY, load_double_gyre_config
from src.generator.double_gyre import run_double_gyre_pipeline


def _has_runtime_dependencies() -> bool:
    return all(shutil.which(command) for command in ("git", "mpif90", "mpirun"))


@pytest.mark.integration
def test_short_double_gyre_run_writes_test_netcdf(tmp_path: Path):
    if not _has_runtime_dependencies():
        pytest.skip("Aronnax runtime requires git plus mpif90 and mpirun.")

    ensure_aronnax_checkout()

    output_path = tmp_path / "raw" / "double_gyre.nc"
    run_directory = tmp_path / "interim" / "double_gyre_run"

    config = load_double_gyre_config("config/generator/double_gyre.yaml").with_overrides(
        nx=6,
        ny=6,
        dx_m=20_000.0,
        dy_m=20_000.0,
        duration_days=(4 * 600.0) / SECONDS_PER_DAY,
        output_interval_days=(2 * 600.0) / SECONDS_PER_DAY,
        max_iterations=200,
        run_directory=run_directory,
        netcdf_output_path=output_path,
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
