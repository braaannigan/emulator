import json
from pathlib import Path
import os

from src.viewer.training_metrics_viewer import (
    RAW_EMULATOR_ROOT,
    REPO_ROOT,
    TrainingMetricsRecord,
    INTERIM_EMULATOR_ROOT,
    list_training_metric_runs,
    load_training_history,
)


def test_list_training_metric_runs_discovers_histories_and_metrics(tmp_path: Path):
    interim_root = tmp_path / "interim" / "emulator"
    raw_root = tmp_path / "raw"
    history_dir = interim_root / "residual_thickness" / "demo-run"
    history_dir.mkdir(parents=True)
    (history_dir / "training_history.json").write_text(json.dumps({"epoch_train_losses": [1.0]}), encoding="utf-8")
    metrics_dir = raw_root / "double_gyre_shifting_wind" / "emulator" / "residual_thickness" / "demo-run"
    metrics_dir.mkdir(parents=True)
    (metrics_dir / "metrics.json").write_text(json.dumps({"train_loss": 1.0}), encoding="utf-8")

    runs = list_training_metric_runs(interim_root=interim_root, raw_root=raw_root)

    assert len(runs) == 1
    assert runs[0].experiment_id == "demo-run"
    assert runs[0].emulator_name == "residual_thickness"
    assert runs[0].updated_at > 0


def test_load_training_history_falls_back_to_metrics(tmp_path: Path):
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({"train_loss": 0.25, "epochs": 100}), encoding="utf-8")
    record = TrainingMetricsRecord(
        emulator_name="residual_thickness",
        experiment_id="demo-run",
        training_history_path=None,
        metrics_path=metrics_path,
        updated_at=metrics_path.stat().st_mtime,
    )

    history = load_training_history(record)

    assert history["status"] == "completed"
    assert history["history_source"] == "metrics_fallback"
    assert history["epochs_completed"] == 100
    assert history["epochs_total"] == 100
    assert history["epoch_train_losses"] == [0.25]


def test_default_roots_are_repo_relative():
    assert INTERIM_EMULATOR_ROOT == REPO_ROOT / "data/interim/emulator"
    assert RAW_EMULATOR_ROOT == REPO_ROOT / "data/raw"


def test_list_training_metric_runs_orders_by_recency_not_experiment_name(tmp_path: Path):
    interim_root = tmp_path / "interim" / "emulator"
    raw_root = tmp_path / "raw"

    older_history_dir = interim_root / "unet_thickness" / "skyd-older"
    older_history_dir.mkdir(parents=True)
    older_history = older_history_dir / "training_history.json"
    older_history.write_text(json.dumps({"epoch_train_losses": [1.0]}), encoding="utf-8")
    older_metrics_dir = raw_root / "double_gyre_shifting_wind" / "emulator" / "unet_thickness" / "skyd-older"
    older_metrics_dir.mkdir(parents=True)
    older_metrics = older_metrics_dir / "metrics.json"
    older_metrics.write_text(json.dumps({"train_loss": 1.0}), encoding="utf-8")

    newer_history_dir = interim_root / "unet_thickness" / "20260330T235959"
    newer_history_dir.mkdir(parents=True)
    newer_history = newer_history_dir / "training_history.json"
    newer_history.write_text(json.dumps({"epoch_train_losses": [0.5]}), encoding="utf-8")
    newer_metrics_dir = raw_root / "double_gyre_shifting_wind" / "emulator" / "unet_thickness" / "20260330T235959"
    newer_metrics_dir.mkdir(parents=True)
    newer_metrics = newer_metrics_dir / "metrics.json"
    newer_metrics.write_text(json.dumps({"train_loss": 0.5}), encoding="utf-8")

    os.utime(older_history, (100, 100))
    os.utime(older_metrics, (100, 100))
    os.utime(newer_history, (200, 200))
    os.utime(newer_metrics, (200, 200))

    runs = list_training_metric_runs(interim_root=interim_root, raw_root=raw_root)

    assert [run.experiment_id for run in runs] == ["20260330T235959", "skyd-older"]
