import torch
import pytest

from src.models.unet_thickness.model import UnetThicknessModel


def test_unet_thickness_model_preserves_spatial_shape():
    model = UnetThicknessModel(
        input_channels=3,
        hidden_channels=8,
        num_levels=2,
        kernel_size=3,
        stage_depth=2,
        dilation_cycle=2,
        norm_type="groupnorm",
        block_type="convnext",
        state_channels=1,
        output_steps=1,
        forcing_channels=2,
        fusion_mode="bottleneck",
        skip_fusion_mode="concat",
        upsample_mode="transpose",
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
        stage_depth=1,
        dilation_cycle=1,
        norm_type="groupnorm",
        block_type="standard",
        state_channels=3,
        output_steps=1,
        forcing_channels=1,
        fusion_mode="input",
        skip_fusion_mode="concat",
        upsample_mode="transpose",
        residual_connection=False,
        prognostic_channels=3,
    )
    inputs = torch.zeros((2, 4, 32, 32))

    outputs = model(inputs)

    assert outputs.shape == (2, 3, 32, 32)


def test_unet_thickness_model_supports_multistep_output():
    model = UnetThicknessModel(
        input_channels=4,
        hidden_channels=8,
        num_levels=2,
        kernel_size=5,
        stage_depth=2,
        dilation_cycle=3,
        norm_type="groupnorm",
        block_type="convnext",
        state_channels=3,
        output_steps=2,
        forcing_channels=1,
        fusion_mode="input",
        skip_fusion_mode="concat",
        upsample_mode="transpose",
        residual_connection=False,
        prognostic_channels=3,
    )
    inputs = torch.zeros((2, 4, 32, 32))

    outputs = model(inputs)

    assert outputs.shape == (2, 2, 3, 32, 32)


def test_unet_thickness_model_supports_per_scale_gated_bilinear_architecture():
    model = UnetThicknessModel(
        input_channels=4,
        hidden_channels=8,
        num_levels=3,
        kernel_size=5,
        stage_depth=2,
        dilation_cycle=2,
        norm_type="none",
        block_type="convnext",
        state_channels=3,
        output_steps=1,
        forcing_channels=1,
        fusion_mode="per_scale",
        skip_fusion_mode="gated",
        upsample_mode="bilinear",
        residual_connection=False,
        prognostic_channels=3,
    )
    inputs = torch.randn((2, 4, 32, 32))

    outputs = model(inputs)

    assert outputs.shape == (2, 3, 32, 32)


def test_unet_thickness_model_supports_nonzero_padding_modes():
    model = UnetThicknessModel(
        input_channels=3,
        hidden_channels=8,
        num_levels=2,
        kernel_size=3,
        stage_depth=1,
        dilation_cycle=1,
        norm_type="groupnorm",
        block_type="standard",
        state_channels=1,
        output_steps=1,
        forcing_channels=2,
        fusion_mode="bottleneck",
        skip_fusion_mode="concat",
        upsample_mode="transpose",
        residual_connection=False,
        boundary_padding_mode="reflect",
    )
    inputs = torch.zeros((2, 3, 32, 32))

    outputs = model(inputs)

    assert outputs.shape == (2, 1, 32, 32)


def test_unet_thickness_model_rejects_invalid_padding_mode():
    with pytest.raises(ValueError, match="Unsupported boundary_padding_mode"):
        UnetThicknessModel(
            input_channels=1,
            hidden_channels=4,
            boundary_padding_mode="invalid",
        )


def test_unet_thickness_model_supports_depthwise_separable_blocks():
    model = UnetThicknessModel(
        input_channels=3,
        hidden_channels=8,
        num_levels=2,
        kernel_size=5,
        stage_depth=2,
        dilation_cycle=2,
        norm_type="groupnorm",
        block_type="depthwise_separable",
        state_channels=1,
        output_steps=1,
        forcing_channels=2,
        fusion_mode="input",
        skip_fusion_mode="add",
        upsample_mode="bilinear",
        residual_connection=False,
        boundary_padding_mode="replicate",
    )
    inputs = torch.zeros((2, 3, 32, 32))

    outputs = model(inputs)

    assert outputs.shape == (2, 1, 32, 32)
