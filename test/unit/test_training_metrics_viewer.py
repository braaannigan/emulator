import json
from pathlib import Path
import os

from src.viewer.training_metrics_viewer import (
    RAW_EMULATOR_ROOT,
    REPO_ROOT,
    TrainingMetricsRecord,
    INTERIM_EMULATOR_ROOT,
    build_run_metadata,
    format_updated_at,
    list_training_metric_runs,
    load_training_history,
    summarize_autonomous_candidate_states,
    summarize_autonomous_stage_events,
    summarize_recent_runs,
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
    assert runs[0].physical_experiment_name == "double_gyre_shifting_wind"
    assert runs[0].updated_at > 0


def test_load_training_history_falls_back_to_metrics(tmp_path: Path):
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({"train_loss": 0.25, "epochs": 100}), encoding="utf-8")
    record = TrainingMetricsRecord(
        emulator_name="residual_thickness",
        physical_experiment_name="double_gyre_shifting_wind",
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


def test_summarize_recent_runs_filters_by_physical_experiment_and_reads_metrics(tmp_path: Path):
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "eval_mse_mean": 12.5,
                "train_loss": 0.1,
                "hypothesis": "try smaller kernel",
                "stop_reason": "threshold",
                "stopped_early": True,
            }
        ),
        encoding="utf-8",
    )
    runs = [
        TrainingMetricsRecord(
            emulator_name="unet_thickness",
            physical_experiment_name="double_gyre_shifting_wind_2layer",
            experiment_id="run-1",
            training_history_path=None,
            metrics_path=metrics_path,
            updated_at=123.0,
        ),
        TrainingMetricsRecord(
            emulator_name="unet_thickness",
            physical_experiment_name="double_gyre",
            experiment_id="run-2",
            training_history_path=None,
            metrics_path=metrics_path,
            updated_at=124.0,
        ),
    ]

    summaries = summarize_recent_runs(runs, physical_experiment_name="double_gyre_shifting_wind_2layer")

    assert len(summaries) == 1
    assert summaries[0]["experiment_id"] == "run-1"
    assert summaries[0]["eval_mse_mean"] == 12.5
    assert summaries[0]["hypothesis"] == "try smaller kernel"


def test_build_run_metadata_prefers_hypothesis_from_metrics(tmp_path: Path):
    history_path = tmp_path / "training_history.json"
    metrics_path = tmp_path / "metrics.json"
    history_path.write_text(
        json.dumps({"epoch_train_losses": [0.5], "status": "completed"}),
        encoding="utf-8",
    )
    metrics_path.write_text(
        json.dumps({"hypothesis": "use residual damping", "eval_mse_mean": 64.2}),
        encoding="utf-8",
    )
    record = TrainingMetricsRecord(
        emulator_name="unet_thickness",
        physical_experiment_name="double_gyre_shifting_wind_2layer",
        experiment_id="demo-run",
        training_history_path=history_path,
        metrics_path=metrics_path,
        updated_at=metrics_path.stat().st_mtime,
    )

    metadata = build_run_metadata(record)

    assert metadata["hypothesis"] == "use residual damping"
    assert metadata["eval_mse_mean"] == 64.2
    assert metadata["updated_at"].endswith("+00:00")


def test_format_updated_at_renders_iso_datetime():
    assert format_updated_at(0.0) == "1970-01-01T00:00:00+00:00"


def test_recent_runs_family_order_can_be_derived_by_latest_update():
    runs = [
        TrainingMetricsRecord(
            emulator_name="unet_thickness",
            physical_experiment_name="older_family",
            experiment_id="run-1",
            training_history_path=None,
            metrics_path=None,
            updated_at=100.0,
        ),
        TrainingMetricsRecord(
            emulator_name="unet_thickness",
            physical_experiment_name="newer_family",
            experiment_id="run-2",
            training_history_path=None,
            metrics_path=None,
            updated_at=200.0,
        ),
    ]

    experiment_names = [
        name
        for name, _updated_at in sorted(
            (
                (
                    name,
                    max(
                        run.updated_at
                        for run in runs
                        if run.physical_experiment_name == name
                    ),
                )
                for name in {run.physical_experiment_name for run in runs if run.physical_experiment_name is not None}
            ),
            key=lambda item: (item[1], item[0]),
            reverse=True,
        )
    ]

    assert experiment_names == ["newer_family", "older_family"]


def test_summarize_autonomous_stage_events_extracts_event_rows():
    rows = summarize_autonomous_stage_events(
        {
            "stage_events": [
                {
                    "timestamp": "2026-04-12T18:00:00+00:00",
                    "phase": "proposal",
                    "category": "phase",
                    "experiment_id": None,
                    "message": "Entering proposal generation.",
                    "details": {"proposal_count": 3},
                }
            ]
        }
    )

    assert rows == [
        {
            "timestamp": "2026-04-12T18:00:00+00:00",
            "phase": "proposal",
            "category": "phase",
            "experiment_id": None,
            "message": "Entering proposal generation.",
            "details": {"proposal_count": 3},
        }
    ]


def test_summarize_autonomous_candidate_states_extracts_candidate_rows():
    rows = summarize_autonomous_candidate_states(
        {
            "candidates": [
                {
                    "experiment_id": "exp-01",
                    "status": "running",
                    "ranking_score": 1.25,
                    "eval_mse_mean": None,
                    "result_class": None,
                    "is_competitive": False,
                    "artifact_severity": None,
                    "started_at": "2026-04-12T18:01:00+00:00",
                    "finished_at": None,
                    "hypothesis": "try residual damping",
                    "stop_reason": None,
                }
            ]
        }
    )

    assert rows == [
        {
            "experiment_id": "exp-01",
            "status": "running",
            "ranking_score": 1.25,
            "eval_mse_mean": None,
            "result_class": None,
            "is_competitive": False,
            "artifact_severity": None,
            "started_at": "2026-04-12T18:01:00+00:00",
            "finished_at": None,
            "hypothesis": "try residual damping",
            "stop_reason": None,
        }
    ]
