import argparse
from pathlib import Path

from src.pipelines import animate_double_gyre, run_double_gyre, train_cnn_thickness


def test_animate_double_gyre_resolve_netcdf_path_prefers_explicit_path():
    args = argparse.Namespace(netcdf_path="demo.nc", experiment_id=None)

    resolved = animate_double_gyre.resolve_netcdf_path(args)

    assert resolved == Path("demo.nc")


def test_animate_double_gyre_resolve_netcdf_path_uses_most_recent(monkeypatch):
    record = type("Record", (), {"experiment_id": "abc", "netcdf_path": Path("a.nc")})()
    monkeypatch.setattr("src.pipelines.animate_double_gyre.list_experiments", lambda: [record])

    resolved = animate_double_gyre.resolve_netcdf_path(argparse.Namespace(netcdf_path=None, experiment_id=None))

    assert resolved == Path("a.nc")


def test_train_cnn_thickness_main_prints_artifact_paths(monkeypatch, capsys):
    monkeypatch.setattr(
        "src.pipelines.train_cnn_thickness.parse_args",
        lambda: argparse.Namespace(
            config="config/emulator/cnn_thickness.yaml",
            source_experiment_id=None,
            experiment_id="demo",
            epochs=2,
        ),
    )
    monkeypatch.setattr(
        "src.pipelines.train_cnn_thickness.run_cnn_thickness_experiment",
        lambda config: {
            "metrics_path": "metrics.json",
            "rollout_path": "rollout.nc",
            "animation_path": "comparison.mp4",
        },
    )

    exit_code = train_cnn_thickness.main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "metrics.json" in captured.out
    assert "rollout.nc" in captured.out
    assert "comparison.mp4" in captured.out


def test_run_double_gyre_main_prints_output_path(monkeypatch, capsys):
    monkeypatch.setattr(
        "src.pipelines.run_double_gyre.parse_args",
        lambda: argparse.Namespace(
            config="config/generator/double_gyre.yaml",
            duration_days=None,
            output_interval_days=None,
            experiment_id="demo",
        ),
    )
    monkeypatch.setattr("src.pipelines.run_double_gyre.run_double_gyre_pipeline", lambda config: Path("output.nc"))

    exit_code = run_double_gyre.main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "output.nc" in captured.out
