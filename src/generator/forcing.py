from __future__ import annotations

import math

import numpy as np

from .config import DoubleGyreConfig, SECONDS_PER_DAY


def build_static_double_gyre_wind(config: DoubleGyreConfig):
    def wind(_, y_coordinates):
        return config.wind_stress_max * (1.0 - np.cos(2.0 * np.pi * y_coordinates / np.max(y_coordinates)))

    return wind


def build_shifting_double_gyre_wind(config: DoubleGyreConfig):
    if config.wind_shift_period_days is None or config.wind_shift_period_days <= 0:
        raise ValueError("wind_shift_period_days must be set to a positive value for shifting-wind experiments.")

    def wind(_, y_coordinates, wind_n_records):
        y_max = float(np.max(y_coordinates))
        period_seconds = config.wind_shift_period_days * SECONDS_PER_DAY
        output = np.empty((wind_n_records, *y_coordinates.shape), dtype=float)
        for record in range(wind_n_records):
            phase = 2.0 * math.pi * (record / wind_n_records)
            shift_m = config.wind_shift_amplitude_m * math.sin(phase)
            shifted_y = np.clip(y_coordinates - shift_m, 0.0, y_max)
            output[record, :, :] = config.wind_stress_max * (
                1.0 - np.cos(2.0 * np.pi * shifted_y / y_max)
            )
        return output

    return wind
