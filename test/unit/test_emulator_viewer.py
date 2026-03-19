import numpy as np
import xarray as xr

from src.viewer.emulator_viewer import mse_timeseries


def test_mse_timeseries_computes_per_step_error():
    dataset = xr.Dataset(
        data_vars={
            "truth_layer_thickness": (("time_days", "y", "x"), np.array([[[1.0]], [[3.0]]])),
            "rollout_layer_thickness": (("time_days", "y", "x"), np.array([[[2.0]], [[5.0]]])),
        },
        coords={"time_days": [0.0, 1.0], "y": [0.0], "x": [0.0]},
    )

    values = mse_timeseries(dataset)

    assert values.tolist() == [1.0, 4.0]
