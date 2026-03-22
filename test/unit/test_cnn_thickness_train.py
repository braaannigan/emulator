import numpy as np

from src.models.cnn_thickness.config import load_cnn_thickness_config
from src.models.cnn_thickness.model import CnnThicknessModel
from src.models.cnn_thickness.train import train_model


def test_train_model_reports_steps_and_examples():
    config = load_cnn_thickness_config("config/emulator/cnn_thickness.yaml").with_overrides(
        epochs=1,
        batch_size=2,
    )
    model = CnnThicknessModel(hidden_channels=4, num_layers=2, kernel_size=3)
    normalized_train_frames = np.random.default_rng(0).normal(size=(4, 4, 4)).astype(np.float32)

    info = train_model(config, model, normalized_train_frames)

    assert info["training_examples"] == 3
    assert info["optimization_steps"] == 2
    assert "train_loss" in info
