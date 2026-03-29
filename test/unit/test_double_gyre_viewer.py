from pathlib import Path

import numpy as np
import xarray as xr

from src.viewer.double_gyre_viewer import list_experiments, select_time, wind_stress_figure


def test_list_experiments_finds_timestamped_runs(tmp_path: Path):
    newer_dir = tmp_path / "20260319T170000"
    older_dir = tmp_path / "20260318T170000"
    newer_dir.mkdir(parents=True)
    older_dir.mkdir(parents=True)
    xr.Dataset(
        {"layer_thickness": (("time_days", "y", "x"), np.ones((1, 2, 2)))},
        coords={"time_days": [1.0], "y": [0.0, 1.0], "x": [0.0, 1.0]},
    ).to_netcdf(newer_dir / "double_gyre.nc")
    xr.Dataset(
        {"layer_thickness": (("time_days", "y", "x"), np.ones((1, 2, 2)))},
        coords={"time_days": [1.0], "y": [0.0, 1.0], "x": [0.0, 1.0]},
    ).to_netcdf(older_dir / "double_gyre.nc")

    experiments = list_experiments(tmp_path)

    assert [experiment.experiment_id for experiment in experiments] == [
        "20260319T170000",
        "20260318T170000",
    ]
    assert all(experiment.physical_experiment_name == tmp_path.parent.name for experiment in experiments)


def test_list_experiments_aggregates_multiple_experiment_families(tmp_path: Path):
    static_dir = tmp_path / "double_gyre" / "generator" / "20260319T170000"
    shifting_dir = tmp_path / "double_gyre_shifting_wind" / "generator" / "20260319T180000"
    static_dir.mkdir(parents=True)
    shifting_dir.mkdir(parents=True)
    xr.Dataset(
        {"layer_thickness": (("time_days", "y", "x"), np.ones((1, 2, 2)))},
        coords={"time_days": [1.0], "y": [0.0, 1.0], "x": [0.0, 1.0]},
    ).to_netcdf(static_dir / "double_gyre.nc")
    xr.Dataset(
        {"layer_thickness": (("time_days", "y", "x"), np.ones((1, 2, 2)))},
        coords={"time_days": [1.0], "y": [0.0, 1.0], "x": [0.0, 1.0]},
    ).to_netcdf(shifting_dir / "double_gyre.nc")

    experiments = list_experiments(tmp_path)

    assert [(experiment.physical_experiment_name, experiment.experiment_id) for experiment in experiments] == [
        ("double_gyre_shifting_wind", "20260319T180000"),
        ("double_gyre", "20260319T170000"),
    ]


def test_select_time_picks_nearest_decimal_day():
    dataset = xr.Dataset(
        {
            "layer_thickness": (
                ("time_days", "y", "x"),
                np.arange(12).reshape(3, 2, 2),
            )
        },
        coords={"time_days": [0.0, 7.0, 14.0], "y": [0.0, 1.0], "x": [0.0, 1.0]},
    )

    selected_day, selected_dataset = select_time(dataset, 6.2)

    assert selected_day == 7.0
    assert selected_dataset["layer_thickness"].shape == (2, 2)


def test_wind_stress_figure_uses_expected_profile():
    dataset = xr.Dataset(coords={"y": [0.0, 1.0, 2.0]}, attrs={"experiment": "double_gyre", "wind_stress_max": "0.05"})

    figure = wind_stress_figure(dataset)

    assert list(figure.data[0].x) == [0.0, 1.0, 2.0]
    assert np.allclose(figure.data[0].y, [0.0, 0.1, 0.0])
