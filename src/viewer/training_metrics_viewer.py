from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from src.viewer.autoloop_viewer import load_autonomous_batch_details, list_autonomous_batches


REPO_ROOT = Path(__file__).resolve().parents[2]
INTERIM_EMULATOR_ROOT = REPO_ROOT / "data/interim/emulator"
RAW_EMULATOR_ROOT = REPO_ROOT / "data/raw"


@dataclass(frozen=True)
class TrainingMetricsRecord:
    emulator_name: str
    physical_experiment_name: str | None
    experiment_id: str
    training_history_path: Path | None
    metrics_path: Path | None
    updated_at: float


def list_training_metric_runs(interim_root: Path = INTERIM_EMULATOR_ROOT, raw_root: Path = RAW_EMULATOR_ROOT) -> list[TrainingMetricsRecord]:
    runs: list[TrainingMetricsRecord] = []
    if not interim_root.exists():
        return runs

    for model_root in sorted(path for path in interim_root.iterdir() if path.is_dir()):
        for experiment_root in sorted(path for path in model_root.iterdir() if path.is_dir()):
            history_path = experiment_root / "training_history.json"
            metrics_matches = sorted(raw_root.glob(f"*/emulator/{model_root.name}/{experiment_root.name}/metrics.json"))
            metrics_path = metrics_matches[0] if metrics_matches else None
            physical_experiment_name = None if metrics_path is None else metrics_path.parents[3].name
            if history_path.exists() or metrics_path is not None:
                updated_at = max(
                    path.stat().st_mtime
                    for path in (history_path, metrics_path)
                    if path is not None and path.exists()
                )
                runs.append(
                    TrainingMetricsRecord(
                        emulator_name=model_root.name,
                        physical_experiment_name=physical_experiment_name,
                        experiment_id=experiment_root.name,
                        training_history_path=history_path if history_path.exists() else None,
                        metrics_path=metrics_path,
                        updated_at=updated_at,
                    )
                )
    return sorted(runs, key=lambda run: (run.updated_at, run.experiment_id, run.emulator_name), reverse=True)


def load_training_history(record: TrainingMetricsRecord) -> dict[str, object]:
    if record.training_history_path is not None and record.training_history_path.exists():
        return json.loads(record.training_history_path.read_text(encoding="utf-8"))
    if record.metrics_path is not None and record.metrics_path.exists():
        metrics = json.loads(record.metrics_path.read_text(encoding="utf-8"))
        epochs_total = int(metrics.get("epochs", 0))
        return {
            "experiment_id": record.experiment_id,
            "status": "completed",
            "history_source": "metrics_fallback",
            "epochs_completed": epochs_total,
            "epochs_total": epochs_total,
            "epoch_train_losses": [float(metrics.get("train_loss", 0.0))],
        }
    return {
        "experiment_id": record.experiment_id,
        "status": "missing",
        "history_source": "missing",
        "epochs_completed": 0,
        "epochs_total": 0,
        "epoch_train_losses": [],
    }


def load_run_metrics(record: TrainingMetricsRecord) -> dict[str, object]:
    if record.metrics_path is None or not record.metrics_path.exists():
        return {}
    payload = json.loads(record.metrics_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def format_updated_at(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def build_run_metadata(record: TrainingMetricsRecord) -> dict[str, object]:
    history = load_training_history(record)
    metrics = load_run_metrics(record)
    return {
        "emulator_name": record.emulator_name,
        "physical_experiment_name": record.physical_experiment_name,
        "experiment_id": record.experiment_id,
        "training_history_path": None if record.training_history_path is None else str(record.training_history_path),
        "metrics_path": None if record.metrics_path is None else str(record.metrics_path),
        "updated_at": format_updated_at(record.updated_at),
        **metrics,
        **history,
        "hypothesis": metrics.get("hypothesis", history.get("hypothesis")),
        "stop_reason": metrics.get("stop_reason", history.get("stop_reason")),
    }


def summarize_recent_runs(
    runs: list[TrainingMetricsRecord],
    *,
    physical_experiment_name: str,
    limit: int = 25,
) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    filtered_runs = [
        run for run in runs if run.physical_experiment_name == physical_experiment_name
    ]
    for run in filtered_runs[:limit]:
        history = load_training_history(run)
        metrics = load_run_metrics(run)
        summaries.append(
            {
                "updated_at": format_updated_at(run.updated_at),
                "emulator_name": run.emulator_name,
                "experiment_id": run.experiment_id,
                "eval_mse_mean": metrics.get("eval_mse_mean"),
                "train_loss": metrics.get("train_loss"),
                "epochs_completed": history.get("epochs_completed"),
                "stopped_early": metrics.get("stopped_early"),
                "hypothesis": metrics.get("hypothesis", history.get("hypothesis")),
                "stop_reason": metrics.get("stop_reason", history.get("stop_reason")),
            }
        )
    return summaries


def training_loss_figure(history: dict[str, object]) -> go.Figure:
    losses = [float(value) for value in history.get("epoch_train_losses", [])]
    epochs = list(range(1, len(losses) + 1))
    figure = go.Figure(
        data=[
            go.Scatter(
                x=epochs,
                y=losses,
                mode="lines+markers",
                name="train_loss",
                line={"color": "#0f766e", "width": 3},
            )
        ]
    )
    figure.update_layout(
        title="Training Loss",
        xaxis_title="epoch",
        yaxis_title="loss",
    )
    return figure


def summarize_autonomous_stage_events(batch_details: dict[str, object]) -> list[dict[str, object]]:
    stage_events = batch_details.get("stage_events", [])
    if not isinstance(stage_events, list):
        return []
    rows: list[dict[str, object]] = []
    for event in stage_events:
        if not isinstance(event, dict):
            continue
        rows.append(
            {
                "timestamp": event.get("timestamp"),
                "phase": event.get("phase"),
                "category": event.get("category"),
                "experiment_id": event.get("experiment_id"),
                "message": event.get("message"),
                "details": event.get("details"),
            }
        )
    return rows


def summarize_autonomous_candidate_states(batch_details: dict[str, object]) -> list[dict[str, object]]:
    candidates = batch_details.get("candidates", [])
    if not isinstance(candidates, list):
        return []
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        rows.append(
            {
                "experiment_id": candidate.get("experiment_id"),
                "status": candidate.get("status"),
                "ranking_score": candidate.get("ranking_score"),
                "eval_mse_mean": candidate.get("eval_mse_mean"),
                "result_class": candidate.get("result_class"),
                "is_competitive": candidate.get("is_competitive"),
                "artifact_severity": candidate.get("artifact_severity"),
                "started_at": candidate.get("started_at"),
                "finished_at": candidate.get("finished_at"),
                "hypothesis": candidate.get("hypothesis"),
                "stop_reason": candidate.get("stop_reason"),
            }
        )
    return rows


def render_training_metrics_page(st_module=st) -> None:
    st_module.set_page_config(page_title="Training Metrics", layout="wide")
    st_module.title("Training Metrics")

    runs = list_training_metric_runs()
    if not runs:
        st_module.warning("No training histories or metrics found under data/interim/emulator.")
        st_module.stop()

    detail_tab, recent_tab, autoloop_tab = st_module.tabs(["Run Detail", "Recent Results", "Loop Activity"])

    with detail_tab:
        selected_run = st_module.sidebar.selectbox(
            "Run",
            runs,
            format_func=lambda run: f"{run.emulator_name} / {run.experiment_id}",
        )
        history = load_training_history(selected_run)
        metadata = build_run_metadata(selected_run)
        losses = history.get("epoch_train_losses", [])

        st_module.caption(
            f"Status: `{history.get('status', 'unknown')}` | "
            f"epochs: {history.get('epochs_completed', 0)} / {history.get('epochs_total', 0)}"
        )
        if history.get("history_source") == "metrics_fallback":
            st_module.info(
                "This run does not have an epoch-by-epoch history file. "
                "The chart shows only the final recorded train loss from metrics.json."
            )
        if losses:
            st_module.plotly_chart(training_loss_figure(history), width="stretch")
        else:
            st_module.info("No training loss history is available for this run.")
        st_module.subheader("Run Metadata")
        st_module.json(metadata)

    with recent_tab:
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
        if not experiment_names:
            st_module.info("No experiment families with metrics.json are available yet.")
        else:
            selected_experiment_name = st_module.sidebar.selectbox("Experiment Family", experiment_names)
            summary_rows = summarize_recent_runs(runs, physical_experiment_name=selected_experiment_name)
            st_module.caption(
                f"Recent runs for `{selected_experiment_name}`: {len(summary_rows)} shown"
            )
            st_module.dataframe(summary_rows, use_container_width=True)

    with autoloop_tab:
        batches = list_autonomous_batches()
        if not batches:
            st_module.info("No autonomous loop batches are available yet.")
        else:
            selected_batch = st_module.sidebar.selectbox(
                "Autonomous Batch",
                batches,
                format_func=lambda batch: f"{batch.batch_id} / {batch.phase} / candidates={batch.candidate_count}",
            )
            details = load_autonomous_batch_details(selected_batch.ledger_path)
            st_module.caption(
                f"Batch `{selected_batch.batch_id}` phase `{selected_batch.phase}`. "
                f"LLM calls `{details.get('llm_calls_used', 0)}`, evaluator calls `{details.get('evaluator_calls_used', 0)}`."
            )
            stage_rows = summarize_autonomous_stage_events(details)
            candidate_rows = summarize_autonomous_candidate_states(details)
            st_module.subheader("Stage Events")
            if stage_rows:
                st_module.dataframe(stage_rows, use_container_width=True)
            else:
                st_module.info("No stage events have been recorded for this batch yet.")
            st_module.subheader("Candidate States")
            if candidate_rows:
                st_module.dataframe(candidate_rows, use_container_width=True)
            else:
                st_module.info("No candidates have been materialized for this batch yet.")


def main(st_module=st) -> None:
    render_training_metrics_page(st_module)


if __name__ == "__main__":
    main()
