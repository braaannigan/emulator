import numpy as np

from src.models.residual_thickness.training import (
    build_forcing_features,
    build_training_inputs,
    center_meridional_velocity,
    center_zonal_velocity,
    compute_relative_vorticity,
    fit_channel_standardizer,
    fit_forcing_standardizer,
    forcing_channel_count,
)


def test_forcing_channel_count_matches_supported_modes():
    assert forcing_channel_count("none") == 0
    assert forcing_channel_count("wind_current") == 1
    assert forcing_channel_count("wind_next") == 1
    assert forcing_channel_count("wind_delta") == 1
    assert forcing_channel_count("wind_current_plus_delta") == 2
    assert forcing_channel_count("wind_mean_12") == 1
    assert forcing_channel_count("wind_current_mean_4_12") == 3


def test_build_training_inputs_appends_current_wind_channel():
    frames = np.arange(3 * 2 * 2, dtype=np.float32).reshape(3, 2, 2)
    forcing = np.ones((3, 2, 2), dtype=np.float32) * 5.0

    forcing_features = build_forcing_features(forcing, "wind_current")
    inputs = build_training_inputs(frames, 1, forcing_features)

    assert inputs.shape == (2, 2, 2, 2)
    assert np.allclose(inputs[0, 0], frames[0])
    assert np.allclose(inputs[0, 1], forcing[0])


def test_build_training_inputs_uses_wind_delta_channel():
    frames = np.zeros((3, 2, 2), dtype=np.float32)
    forcing = np.array(
        [
            np.zeros((2, 2), dtype=np.float32),
            np.ones((2, 2), dtype=np.float32) * 2.0,
            np.ones((2, 2), dtype=np.float32) * 5.0,
        ]
    )

    forcing_features = build_forcing_features(forcing, "wind_delta")
    inputs = build_training_inputs(frames, 1, forcing_features)

    assert np.allclose(inputs[0, 1], np.ones((2, 2), dtype=np.float32) * 2.0)
    assert np.allclose(inputs[1, 1], np.ones((2, 2), dtype=np.float32) * 3.0)


def test_build_training_inputs_includes_state_history_channels():
    frames = np.arange(4 * 2 * 2, dtype=np.float32).reshape(4, 2, 2)

    inputs = build_training_inputs(frames, 3, None)

    assert inputs.shape == (3, 3, 2, 2)
    assert np.allclose(inputs[0, 0], frames[0])
    assert np.allclose(inputs[0, 1], frames[0])
    assert np.allclose(inputs[0, 2], frames[0])
    assert np.allclose(inputs[2, 0], frames[2])
    assert np.allclose(inputs[2, 1], frames[1])
    assert np.allclose(inputs[2, 2], frames[0])


def test_build_training_inputs_flattens_multichannel_state_history():
    frames = np.arange(4 * 2 * 2 * 2, dtype=np.float32).reshape(4, 2, 2, 2)

    inputs = build_training_inputs(frames, 2, None)

    assert inputs.shape == (3, 4, 2, 2)
    assert np.allclose(inputs[0, 0], frames[0, 0])
    assert np.allclose(inputs[0, 1], frames[0, 1])
    assert np.allclose(inputs[0, 2], frames[0, 0])
    assert np.allclose(inputs[0, 3], frames[0, 1])


def test_build_forcing_features_supports_multiscale_modes():
    forcing = np.arange(5 * 2 * 2, dtype=np.float32).reshape(5, 2, 2)

    features = build_forcing_features(forcing, "wind_current_mean_12_anom_12")

    assert features.shape == (5, 3, 2, 2)
    assert np.allclose(features[0, 0], forcing[0])
    assert np.allclose(features[0, 1], forcing[0])
    assert np.allclose(features[0, 2], np.zeros((2, 2), dtype=np.float32))


def test_fit_forcing_standardizer_round_trips_multichannel_features():
    forcing_features = np.array(
        [
            [[[1.0]], [[10.0]]],
            [[[3.0]], [[14.0]]],
            [[[5.0]], [[18.0]]],
        ],
        dtype=np.float32,
    )

    standardizer = fit_forcing_standardizer(forcing_features)
    normalized = standardizer.normalize(forcing_features)
    restored = standardizer.denormalize(normalized)

    assert np.allclose(restored, forcing_features)


def test_fit_channel_standardizer_round_trips_multichannel_state():
    frames = np.array(
        [
            [[[1.0]], [[10.0]]],
            [[[3.0]], [[14.0]]],
            [[[5.0]], [[18.0]]],
        ],
        dtype=np.float32,
    )

    standardizer = fit_channel_standardizer(frames)
    normalized = standardizer.normalize(frames)
    restored = standardizer.denormalize(normalized)

    assert np.allclose(restored, frames)


def test_velocity_centering_and_vorticity_shapes():
    zonal = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    meridional = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)
    y = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    x = np.array([0.0, 1.0, 2.0], dtype=np.float32)

    centered_u = center_zonal_velocity(zonal)
    centered_v = center_meridional_velocity(meridional)
    vorticity = compute_relative_vorticity(centered_u, centered_v, y, x)

    assert centered_u.shape == (2, 3, 3)
    assert centered_v.shape == (2, 3, 3)
    assert vorticity.shape == (2, 3, 3)
