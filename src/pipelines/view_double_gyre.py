from __future__ import annotations

import streamlit as st

from src.viewer.double_gyre_viewer import (
    available_fields,
    available_layers,
    field_to_heatmap,
    list_experiments,
    open_experiment_dataset,
    select_time,
    wind_stress_figure,
)
from src.viewer.emulator_viewer import (
    comparison_heatmap_figure,
    list_emulator_experiments,
    load_metrics,
    mse_timeseries_figure,
    open_rollout_dataset,
    rollout_color_limits,
)
from src.viewer.autoloop_viewer import load_autonomous_batch_details, list_autonomous_batches


def configure_app(st_module=st) -> None:
    st_module.set_page_config(page_title="Double Gyre Viewer", layout="wide")
    st_module.title("Double Gyre Viewer")


def _select_requested_day(st_module, dataset, label: str) -> float:
    available_days = [float(day) for day in dataset["time_days"].values.tolist()]
    min_day = min(available_days)
    max_day = max(available_days)
    if len(available_days) == 1:
        requested_day = available_days[0]
        st_module.sidebar.caption(f"Only one timestep available: {requested_day:.3f} days")
        return requested_day
    return st_module.sidebar.slider(
        label,
        min_value=min_day,
        max_value=max_day,
        value=min_day,
    )


def render_double_gyre_page(st_module=st) -> None:
    experiments = list_experiments()
    if not experiments:
        st_module.warning("No raw generator experiments found under data/raw.")
        st_module.stop()

    selected_experiment = st_module.sidebar.selectbox(
        "Experiment",
        experiments,
        format_func=lambda experiment: f"{experiment.physical_experiment_name} / {experiment.experiment_id}",
    )

    dataset = open_experiment_dataset(selected_experiment.netcdf_path)
    try:
        requested_day = _select_requested_day(st_module, dataset, "Timestep (days)")
        layer_options = available_layers(dataset)
        selected_layer = st_module.sidebar.selectbox("Layer", layer_options, index=0)
        fields = available_fields(dataset)
        selected_fields = st_module.sidebar.multiselect("Fields", fields, default=fields)

        selected_day, selected_dataset = select_time(dataset, requested_day)
        st_module.caption(f"Showing model state at {selected_day:.3f} days")

        if not selected_fields:
            st_module.info("Choose at least one field in the sidebar.")
            st_module.stop()

        columns = st_module.columns(min(2, len(selected_fields)))
        for index, field_name in enumerate(selected_fields):
            column = columns[index % len(columns)]
            with column:
                figure = field_to_heatmap(selected_dataset, field_name, layer_index=selected_layer)
                st_module.plotly_chart(figure, use_container_width=True)
        st_module.plotly_chart(wind_stress_figure(dataset), use_container_width=True)
        metadata = {
            "physical_experiment_name": selected_experiment.physical_experiment_name,
            "experiment_id": selected_experiment.experiment_id,
            "netcdf_path": str(selected_experiment.netcdf_path),
            **{key: value for key, value in dataset.attrs.items()},
        }
        st_module.subheader("Experiment Metadata")
        st_module.json(metadata)
    finally:
        dataset.close()


def render_emulator_evaluation_page(st_module=st) -> None:
    experiments = list_emulator_experiments()
    if not experiments:
        st_module.warning("No emulator evaluation outputs found under data/raw.")
        st_module.stop()

    selected_experiment = st_module.sidebar.selectbox(
        "Evaluation Experiment",
        experiments,
        format_func=lambda experiment: (
            f"{experiment.physical_experiment_name} / {experiment.emulator_name} / {experiment.experiment_id}"
        ),
    )

    dataset = open_rollout_dataset(selected_experiment.rollout_path)
    try:
        metrics = load_metrics(selected_experiment.metrics_path)
        requested_day = _select_requested_day(st_module, dataset, "Evaluation Timestep (days)")

        if metrics:
            overall_mse = float(metrics.get("eval_mse_mean", metrics.get("mse", 0.0)))
            st_module.caption(
                f"Eval experiment `{selected_experiment.experiment_id}` with overall MSE {overall_mse:.6f}"
            )
        else:
            st_module.caption(f"Eval experiment `{selected_experiment.experiment_id}`")

        st_module.plotly_chart(mse_timeseries_figure(dataset), use_container_width=True)

        selected_day, selected_dataset = select_time(dataset, requested_day)
        st_module.caption(f"Showing evaluation rollout at {selected_day:.3f} days")
        zmin, zmax = rollout_color_limits(dataset)
        st_module.caption(f"Shared color scale: {zmin:.3f} to {zmax:.3f}")
        columns = st_module.columns(2)
        with columns[0]:
            st_module.plotly_chart(
                comparison_heatmap_figure(
                    selected_dataset,
                    "truth_layer_thickness",
                    "Truth Layer Thickness",
                    zmin,
                    zmax,
                    False,
                ),
                use_container_width=True,
            )
        with columns[1]:
            st_module.plotly_chart(
                comparison_heatmap_figure(
                    selected_dataset,
                    "rollout_layer_thickness",
                    "Rollout Layer Thickness",
                    zmin,
                    zmax,
                    True,
                ),
                use_container_width=True,
            )
        metadata = {
            "physical_experiment_name": selected_experiment.physical_experiment_name,
            "emulator_name": selected_experiment.emulator_name,
            "evaluation_experiment_id": selected_experiment.experiment_id,
            "rollout_path": str(selected_experiment.rollout_path),
            **metrics,
        }
        st_module.subheader("Experiment Metadata")
        st_module.json(metadata)
    finally:
        dataset.close()


def render_autonomous_loop_page(st_module=st) -> None:
    batches = list_autonomous_batches()
    if not batches:
        st_module.warning("No autonomous loop batches found under data/interim/autoloop/unet.")
        st_module.stop()

    selected_batch = st_module.sidebar.selectbox(
        "Autonomous Batch",
        batches,
        format_func=lambda batch: f"{batch.batch_id} / {batch.phase} / candidates={batch.candidate_count}",
    )
    details = load_autonomous_batch_details(selected_batch.ledger_path)
    st_module.caption(
        f"Batch `{selected_batch.batch_id}` phase `{selected_batch.phase}` with "
        f"{selected_batch.candidate_count} candidates and {selected_batch.screened_out_count} screened-out proposals."
    )

    batch_rows = [
        {
            "batch_id": batch.batch_id,
            "phase": batch.phase,
            "updated_at": batch.updated_at,
            "llm_calls_used": batch.llm_calls_used,
            "candidate_count": batch.candidate_count,
            "completed_count": batch.completed_count,
            "screened_out_count": batch.screened_out_count,
            "best_eval_mse_mean": batch.best_eval_mse_mean,
            "failure_reason": batch.failure_reason,
        }
        for batch in batches
    ]
    st_module.subheader("Autonomous Batches")
    st_module.dataframe(batch_rows, use_container_width=True)

    candidate_rows = [
        {
            "experiment_id": candidate.get("experiment_id"),
            "status": candidate.get("status"),
            "eval_mse_mean": candidate.get("eval_mse_mean"),
            "stop_reason": candidate.get("stop_reason"),
            "started_at": candidate.get("started_at"),
            "finished_at": candidate.get("finished_at"),
            "hypothesis": candidate.get("hypothesis"),
        }
        for candidate in details.get("candidates", [])
        if isinstance(candidate, dict)
    ]
    st_module.subheader("Iteration Results")
    if candidate_rows:
        st_module.dataframe(candidate_rows, use_container_width=True)
    else:
        st_module.info("No candidate iterations recorded for this batch yet.")

    if details.get("failure_reason") is not None:
        st_module.subheader("Failure")
        st_module.json(
            {
                "failure_reason": details.get("failure_reason"),
                "failure_details": details.get("failure_details"),
            }
        )


def main(st_module=st) -> None:
    configure_app(st_module)
    page = st_module.sidebar.radio("Page", ["Double Gyre", "Emulator Evaluation", "Autonomous Loop"])
    if page == "Double Gyre":
        render_double_gyre_page(st_module)
    elif page == "Emulator Evaluation":
        render_emulator_evaluation_page(st_module)
    else:
        render_autonomous_loop_page(st_module)


if __name__ == "__main__":
    main()
