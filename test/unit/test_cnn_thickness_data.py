import numpy as np

from src.models.cnn_thickness.data import fit_standardizer, split_sequence


def test_split_sequence_keeps_temporal_order():
    frames = np.arange(10 * 2 * 2, dtype=np.float32).reshape(10, 2, 2)
    time_days = np.arange(10, dtype=np.float32)

    split = split_sequence(frames, time_days, 0.8)

    assert split.train_frames.shape[0] == 8
    assert split.eval_frames.shape[0] == 2
    assert split.train_time_days.tolist() == list(range(8))
    assert split.eval_time_days.tolist() == [8.0, 9.0]


def test_fit_standardizer_round_trips_values():
    values = np.array([[[1.0]], [[3.0]], [[5.0]]], dtype=np.float32)
    standardizer = fit_standardizer(values)

    normalized = standardizer.normalize(values)
    restored = standardizer.denormalize(normalized)

    assert np.allclose(restored, values)
