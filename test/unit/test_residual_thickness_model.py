import torch

from src.models.residual_thickness.model import ResidualThicknessModel


def test_residual_thickness_model_preserves_spatial_shape():
    model = ResidualThicknessModel(
        hidden_channels=4,
        num_blocks=3,
        kernel_size=3,
        block_type="standard",
        normalization="group",
        dilation_cycle=2,
    )
    inputs = torch.zeros((2, 1, 8, 8))

    outputs = model(inputs)

    assert outputs.shape == (2, 1, 8, 8)
