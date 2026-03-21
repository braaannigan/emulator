from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

from src.pipelines import view_double_gyre


class StreamlitStop(Exception):
    pass


@dataclass
class SidebarStub:
    radio_value: str = "Double Gyre"
    selectbox_values: list[object] | None = None
    slider_value: float = 0.0
    multiselect_value: list[str] | None = None
    captions: list[str] | None = None

    def __post_init__(self):
        if self.selectbox_values is None:
            self.selectbox_values = []
        if self.captions is None:
            self.captions = []

    def radio(self, label, options):
        return self.radio_value

    def selectbox(self, label, options, format_func=None, index=0):
        if self.selectbox_values:
            return self.selectbox_values.pop(0)
        return options[index]

    def slider(self, label, min_value, max_value, value):
        return self.slider_value or value

    def multiselect(self, label, options, default):
        if self.multiselect_value is None:
            return default
        return self.multiselect_value

    def caption(self, text):
        self.captions.append(text)


class StreamlitStub:
    def __init__(self, page: str = "Double Gyre"):
        self.sidebar = SidebarStub(radio_value=page)
        self.captions: list[str] = []
        self.plotly_calls: list[object] = []
        self.json_payloads: list[dict] = []
        self.subheaders: list[str] = []
        self.info_messages: list[str] = []
        self.warning_messages: list[str] = []
        self.page_config: dict | None = None
        self.title_text: str | None = None

    def set_page_config(self, **kwargs):
        self.page_config = kwargs

    def title(self, text):
        self.title_text = text

    def caption(self, text):
        self.captions.append(text)

    def plotly_chart(self, figure, use_container_width=True):
        self.plotly_calls.append(figure)

    def columns(self, count):
        return [nullcontext() for _ in range(count)]

    def subheader(self, text):
        self.subheaders.append(text)

    def json(self, payload):
        self.json_payloads.append(payload)

    def info(self, text):
        self.info_messages.append(text)

    def warning(self, text):
        self.warning_messages.append(text)

    def stop(self):
        raise StreamlitStop()


def _double_gyre_dataset():
    return xr.Dataset(
        data_vars={
            "layer_thickness": (("time_days", "layers", "y", "x"), np.ones((2, 1, 2, 2), dtype=np.float32)),
            "zonal_velocity": (("time_days", "layers", "y", "x"), np.ones((2, 1, 2, 2), dtype=np.float32)),
        },
        coords={"time_days": [1.0, 2.0], "layers": [0], "y": [0.0, 1.0], "x": [0.0, 1.0]},
        attrs={"model": "Aronnax"},
    )


def _rollout_dataset():
    return xr.Dataset(
        data_vars={
            "truth_layer_thickness": (("time_days", "y", "x"), np.ones((2, 2, 2), dtype=np.float32)),
            "rollout_layer_thickness": (("time_days", "y", "x"), np.zeros((2, 2, 2), dtype=np.float32)),
        },
        coords={"time_days": [1.0, 2.0], "y": [0.0, 1.0], "x": [0.0, 1.0]},
    )


def test_configure_app_sets_page_config_and_title():
    st = StreamlitStub()

    view_double_gyre.configure_app(st)

    assert st.page_config == {"page_title": "Double Gyre Viewer", "layout": "wide"}
    assert st.title_text == "Double Gyre Viewer"


def test_render_double_gyre_page_shows_plots_and_metadata(monkeypatch):
    st = StreamlitStub(page="Double Gyre")
    experiment = type("Experiment", (), {"experiment_id": "exp1", "netcdf_path": Path("demo.nc")})()
    st.sidebar.selectbox_values = [experiment, 0]

    monkeypatch.setattr("src.pipelines.view_double_gyre.list_experiments", lambda: [experiment])
    monkeypatch.setattr("src.pipelines.view_double_gyre.open_experiment_dataset", lambda path: _double_gyre_dataset())
    monkeypatch.setattr("src.pipelines.view_double_gyre.field_to_heatmap", lambda dataset, field_name, layer_index=0: field_name)

    view_double_gyre.render_double_gyre_page(st)

    assert st.plotly_calls == ["layer_thickness", "zonal_velocity"]
    assert st.subheaders == ["Experiment Metadata"]
    assert st.json_payloads[0]["experiment_id"] == "exp1"
    assert st.json_payloads[0]["model"] == "Aronnax"


def test_render_double_gyre_page_stops_when_no_fields_selected(monkeypatch):
    st = StreamlitStub(page="Double Gyre")
    experiment = type("Experiment", (), {"experiment_id": "exp1", "netcdf_path": Path("demo.nc")})()
    st.sidebar.selectbox_values = [experiment, 0]
    st.sidebar.multiselect_value = []

    monkeypatch.setattr("src.pipelines.view_double_gyre.list_experiments", lambda: [experiment])
    monkeypatch.setattr("src.pipelines.view_double_gyre.open_experiment_dataset", lambda path: _double_gyre_dataset())

    try:
        view_double_gyre.render_double_gyre_page(st)
    except StreamlitStop:
        pass

    assert st.info_messages == ["Choose at least one field in the sidebar."]


def test_render_emulator_evaluation_page_shows_timeseries_and_metadata(monkeypatch):
    st = StreamlitStub(page="Emulator Evaluation")
    experiment = type(
        "Experiment",
        (),
        {"experiment_id": "eval1", "rollout_path": Path("rollout.nc"), "metrics_path": Path("metrics.json")},
    )()
    st.sidebar.selectbox_values = [experiment]

    monkeypatch.setattr("src.pipelines.view_double_gyre.list_emulator_experiments", lambda: [experiment])
    monkeypatch.setattr("src.pipelines.view_double_gyre.open_rollout_dataset", lambda path: _rollout_dataset())
    monkeypatch.setattr("src.pipelines.view_double_gyre.load_metrics", lambda path: {"mse": 1.5, "epochs": 10})
    monkeypatch.setattr("src.pipelines.view_double_gyre.mse_timeseries_figure", lambda dataset: "timeseries")
    monkeypatch.setattr(
        "src.pipelines.view_double_gyre.comparison_heatmap_figure",
        lambda dataset, field_name, title, zmin, zmax, show_colorbar: title,
    )

    view_double_gyre.render_emulator_evaluation_page(st)

    assert st.plotly_calls == ["timeseries", "Truth Layer Thickness", "Rollout Layer Thickness"]
    assert st.subheaders == ["Experiment Metadata"]
    assert st.json_payloads[0]["evaluation_experiment_id"] == "eval1"
    assert st.json_payloads[0]["epochs"] == 10


def test_main_routes_to_emulator_page(monkeypatch):
    st = StreamlitStub(page="Emulator Evaluation")
    called = {"raw": 0, "eval": 0}

    monkeypatch.setattr("src.pipelines.view_double_gyre.render_double_gyre_page", lambda st_module: called.__setitem__("raw", called["raw"] + 1))
    monkeypatch.setattr("src.pipelines.view_double_gyre.render_emulator_evaluation_page", lambda st_module: called.__setitem__("eval", called["eval"] + 1))

    view_double_gyre.main(st)

    assert called == {"raw": 0, "eval": 1}
