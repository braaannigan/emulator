import torch

from src.models.cnn_thickness.model import CnnThicknessModel


def test_cnn_thickness_model_preserves_spatial_shape():
    model = CnnThicknessModel(hidden_channels=4, num_layers=3, kernel_size=3)
    inputs = torch.zeros((2, 1, 8, 8))

    outputs = model(inputs)

    assert outputs.shape == (2, 1, 8, 8)
