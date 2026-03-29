import json
from pathlib import Path

from src.viewer.training_metrics_viewer import (
    TrainingMetricsRecord,
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


def test_load_training_history_falls_back_to_metrics(tmp_path: Path):
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({"train_loss": 0.25, "epochs": 100}), encoding="utf-8")
    record = TrainingMetricsRecord(
        emulator_name="residual_thickness",
        experiment_id="demo-run",
        training_history_path=None,
        metrics_path=metrics_path,
    )

    history = load_training_history(record)

    assert history["status"] == "completed"
    assert history["history_source"] == "metrics_fallback"
    assert history["epochs_completed"] == 100
    assert history["epochs_total"] == 100
    assert history["epoch_train_losses"] == [0.25]
