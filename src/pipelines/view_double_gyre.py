from __future__ import annotations

import streamlit as st

from src.viewer.double_gyre_viewer import (
    available_fields,
    available_layers,
    field_to_heatmap,
    list_experiments,
    open_experiment_dataset,
    select_time,
)
from src.viewer.emulator_viewer import (
    comparison_heatmap_figure,
    list_emulator_experiments,
    load_metrics,
    mse_timeseries_figure,
    open_rollout_dataset,
    rollout_color_limits,
)


st.set_page_config(page_title="Double Gyre Viewer", layout="wide")
st.title("Double Gyre Viewer")
page = st.sidebar.radio("Page", ["Double Gyre", "Emulator Evaluation"])

if page == "Double Gyre":
    experiments = list_experiments()
    if not experiments:
        st.warning("No experiments found under data/raw/double_gyre.")
        st.stop()

    selected_experiment = st.sidebar.selectbox(
        "Experiment",
        experiments,
        format_func=lambda experiment: experiment.experiment_id,
    )

    dataset = open_experiment_dataset(selected_experiment.netcdf_path)
    try:
        available_days = [float(day) for day in dataset["time_days"].values.tolist()]
        min_day = min(available_days)
        max_day = max(available_days)
        if len(available_days) == 1:
            requested_day = available_days[0]
            st.sidebar.caption(f"Only one timestep available: {requested_day:.3f} days")
        else:
            requested_day = st.sidebar.slider(
                "Timestep (days)",
                min_value=min_day,
                max_value=max_day,
                value=min_day,
            )
        layer_options = available_layers(dataset)
        selected_layer = st.sidebar.selectbox("Layer", layer_options, index=0)
        fields = available_fields(dataset)
        selected_fields = st.sidebar.multiselect("Fields", fields, default=fields)

        selected_day, selected_dataset = select_time(dataset, requested_day)
        st.caption(f"Showing model state at {selected_day:.3f} days")

        if not selected_fields:
            st.info("Choose at least one field in the sidebar.")
            st.stop()

        columns = st.columns(min(2, len(selected_fields)))
        for index, field_name in enumerate(selected_fields):
            column = columns[index % len(columns)]
            with column:
                figure = field_to_heatmap(selected_dataset, field_name, layer_index=selected_layer)
                st.plotly_chart(figure, use_container_width=True)
        metadata = {
            "experiment_id": selected_experiment.experiment_id,
            "netcdf_path": str(selected_experiment.netcdf_path),
            **{key: value for key, value in dataset.attrs.items()},
        }
        st.subheader("Experiment Metadata")
        st.json(metadata)
    finally:
        dataset.close()
else:
    experiments = list_emulator_experiments()
    if not experiments:
        st.warning("No emulator evaluation outputs found under data/raw/emulator/cnn_thickness.")
        st.stop()

    selected_experiment = st.sidebar.selectbox(
        "Evaluation Experiment",
        experiments,
        format_func=lambda experiment: experiment.experiment_id,
    )

    dataset = open_rollout_dataset(selected_experiment.rollout_path)
    try:
        metrics = load_metrics(selected_experiment.metrics_path)
        available_days = [float(day) for day in dataset["time_days"].values.tolist()]
        min_day = min(available_days)
        max_day = max(available_days)
        if len(available_days) == 1:
            requested_day = available_days[0]
            st.sidebar.caption(f"Only one timestep available: {requested_day:.3f} days")
        else:
            requested_day = st.sidebar.slider(
                "Evaluation Timestep (days)",
                min_value=min_day,
                max_value=max_day,
                value=min_day,
            )

        if metrics:
            st.caption(
                f"Eval experiment `{selected_experiment.experiment_id}` with overall MSE {float(metrics['mse']):.6f}"
            )
        else:
            st.caption(f"Eval experiment `{selected_experiment.experiment_id}`")

        st.plotly_chart(mse_timeseries_figure(dataset), use_container_width=True)

        selected_day, selected_dataset = select_time(dataset, requested_day)
        st.caption(f"Showing evaluation rollout at {selected_day:.3f} days")
        zmin, zmax = rollout_color_limits(dataset)
        st.caption(f"Shared color scale: {zmin:.3f} to {zmax:.3f}")
        columns = st.columns(2)
        with columns[0]:
            st.plotly_chart(
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
            st.plotly_chart(
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
            "evaluation_experiment_id": selected_experiment.experiment_id,
            "rollout_path": str(selected_experiment.rollout_path),
            **metrics,
        }
        st.subheader("Experiment Metadata")
        st.json(metadata)
    finally:
        dataset.close()
