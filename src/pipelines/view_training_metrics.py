from __future__ import annotations

import streamlit as st

from src.viewer.training_metrics_viewer import (
    list_training_metric_runs,
    load_training_history,
    training_loss_figure,
)


def main(st_module=st) -> None:
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
        st_module.info("This run does not have an epoch-by-epoch history file. The chart shows only the final recorded train loss from metrics.json.")
    if losses:
        st_module.plotly_chart(training_loss_figure(history), use_container_width=True)
    else:
        st_module.info("No training loss history is available for this run.")
    st_module.subheader("Run Metadata")
    st_module.json(
        {
            "emulator_name": selected_run.emulator_name,
            "experiment_id": selected_run.experiment_id,
            "training_history_path": None if selected_run.training_history_path is None else str(selected_run.training_history_path),
            "metrics_path": None if selected_run.metrics_path is None else str(selected_run.metrics_path),
            **history,
        }
    )


if __name__ == "__main__":
    main()
