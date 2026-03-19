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


st.set_page_config(page_title="Double Gyre Viewer", layout="wide")
st.title("Double Gyre Viewer")

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
finally:
    dataset.close()
