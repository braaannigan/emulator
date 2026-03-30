from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st


INTERIM_EMULATOR_ROOT = Path("data/interim/emulator")
RAW_EMULATOR_ROOT = Path("data/raw")


@dataclass(frozen=True)
class TrainingMetricsRecord:
    emulator_name: str
    experiment_id: str
    training_history_path: Path | None
    metrics_path: Path | None


def list_training_metric_runs(interim_root: Path = INTERIM_EMULATOR_ROOT, raw_root: Path = RAW_EMULATOR_ROOT) -> list[TrainingMetricsRecord]:
    runs: list[TrainingMetricsRecord] = []
    if not interim_root.exists():
        return runs

    for model_root in sorted(path for path in interim_root.iterdir() if path.is_dir()):
        for experiment_root in sorted(path for path in model_root.iterdir() if path.is_dir()):
            history_path = experiment_root / "training_history.json"
            metrics_matches = sorted(raw_root.glob(f"*/emulator/{model_root.name}/{experiment_root.name}/metrics.json"))
            metrics_path = metrics_matches[0] if metrics_matches else None
            if history_path.exists() or metrics_path is not None:
                runs.append(
                    TrainingMetricsRecord(
                        emulator_name=model_root.name,
                        experiment_id=experiment_root.name,
                        training_history_path=history_path if history_path.exists() else None,
                        metrics_path=metrics_path,
                    )
                )
    return sorted(runs, key=lambda run: (run.experiment_id, run.emulator_name), reverse=True)


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


def render_training_metrics_page(st_module=st) -> None:
    st_module.set_page_config(page_title="Training Metrics", layout="wide")
    st_module.title("Training Metrics")

    runs = list_training_metric_runs()
    if not runs:
        st_module.warning("No training histories or metrics found under data/interim/emulator.")
        st_module.stop()

    selected_run = st_module.sidebar.selectbox(
        "Run",
        runs,
        format_func=lambda run: f"{run.emulator_name} / {run.experiment_id}",
    )
    history = load_training_history(selected_run)
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
    st_module.json(
        {
            "emulator_name": selected_run.emulator_name,
            "experiment_id": selected_run.experiment_id,
            "training_history_path": None
            if selected_run.training_history_path is None
            else str(selected_run.training_history_path),
            "metrics_path": None if selected_run.metrics_path is None else str(selected_run.metrics_path),
            **history,
        }
    )


def main(st_module=st) -> None:
    render_training_metrics_page(st_module)


if __name__ == "__main__":
    main()
