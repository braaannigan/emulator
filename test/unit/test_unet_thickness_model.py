import torch

from src.models.unet_thickness.model import UnetThicknessModel


def test_unet_thickness_model_preserves_spatial_shape():
    model = UnetThicknessModel(
        input_channels=3,
        hidden_channels=8,
        num_levels=2,
        kernel_size=3,
        block_type="convnext",
        state_channels=1,
        forcing_channels=2,
        fusion_mode="bottleneck",
        residual_connection=False,
    )
    inputs = torch.zeros((2, 3, 32, 32))

    outputs = model(inputs)

    assert outputs.shape == (2, 1, 32, 32)


def test_unet_thickness_model_supports_multichannel_output():
    model = UnetThicknessModel(
        input_channels=4,
        hidden_channels=8,
        num_levels=2,
        kernel_size=3,
        block_type="standard",
        state_channels=3,
        forcing_channels=1,
        fusion_mode="input",
        residual_connection=False,
        prognostic_channels=3,
    )
    inputs = torch.zeros((2, 4, 32, 32))

    outputs = model(inputs)

    assert outputs.shape == (2, 3, 32, 32)
