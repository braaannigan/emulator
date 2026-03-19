import numpy as np
import torch

from src.models.cnn_thickness.data import Standardizer
from src.models.cnn_thickness.rollout import autoregressive_rollout


class AddOneModel(torch.nn.Module):
    def forward(self, inputs):
        return inputs + 1.0


def test_autoregressive_rollout_feeds_back_predictions():
    eval_frames = np.array(
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[1.0, 1.0], [1.0, 1.0]],
            [[2.0, 2.0], [2.0, 2.0]],
        ],
        dtype=np.float32,
    )
    standardizer = Standardizer(mean=0.0, std=1.0)

    rollout = autoregressive_rollout(AddOneModel(), eval_frames, standardizer, torch.device("cpu"))

    assert rollout.shape == (2, 2, 2)
    assert np.allclose(rollout[0], 1.0)
    assert np.allclose(rollout[1], 2.0)
