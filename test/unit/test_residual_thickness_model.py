import torch

from src.models.residual_thickness.model import ResidualThicknessModel


def test_residual_thickness_model_preserves_spatial_shape():
    model = ResidualThicknessModel(
        input_channels=3,
        hidden_channels=4,
        num_blocks=3,
        kernel_size=3,
        block_type="standard",
        normalization="group",
        dilation_cycle=2,
    )
    inputs = torch.zeros((2, 3, 8, 8))

    outputs = model(inputs)

    assert outputs.shape == (2, 1, 8, 8)


def test_residual_thickness_model_supports_additive_forcing_integration():
    model = ResidualThicknessModel(
        input_channels=2,
        hidden_channels=4,
        num_blocks=2,
        kernel_size=3,
        block_type="depthwise_separable",
        normalization="none",
        dilation_cycle=1,
        prognostic_channels=1,
        state_history=1,
        forcing_channels=1,
        forcing_integration="add",
    )
    inputs = torch.zeros((2, 2, 8, 8))

    outputs = model(inputs)

    assert outputs.shape == (2, 1, 8, 8)


def test_residual_thickness_model_supports_film_forcing_integration():
    model = ResidualThicknessModel(
        input_channels=4,
        hidden_channels=4,
        num_blocks=2,
        kernel_size=3,
        block_type="standard",
        normalization="group",
        dilation_cycle=1,
        prognostic_channels=1,
        state_history=1,
        forcing_channels=3,
        forcing_integration="film",
    )
    inputs = torch.zeros((2, 4, 8, 8))

    outputs = model(inputs)

    assert outputs.shape == (2, 1, 8, 8)


def test_residual_thickness_model_supports_multichannel_prognostic_state():
    model = ResidualThicknessModel(
        input_channels=6,
        hidden_channels=4,
        num_blocks=2,
        kernel_size=3,
        block_type="depthwise_separable",
        normalization="none",
        dilation_cycle=1,
        prognostic_channels=3,
        state_history=2,
        forcing_channels=0,
        forcing_integration="concat",
    )
    inputs = torch.zeros((2, 6, 8, 8))

    outputs = model(inputs)

    assert outputs.shape == (2, 3, 8, 8)


def test_residual_thickness_model_supports_transport_variant():
    model = ResidualThicknessModel(
        input_channels=2,
        hidden_channels=4,
        num_blocks=2,
        kernel_size=3,
        model_variant="transport",
        block_type="depthwise_separable",
        normalization="none",
        dilation_cycle=1,
        prognostic_channels=1,
        state_history=1,
        forcing_channels=1,
        forcing_integration="film",
    )
    inputs = torch.zeros((2, 2, 8, 8))

    outputs = model(inputs)

    assert outputs.shape == (2, 1, 8, 8)


def test_transport_variant_accepts_bounded_transport_scales():
    model = ResidualThicknessModel(
        input_channels=2,
        hidden_channels=4,
        num_blocks=2,
        kernel_size=3,
        model_variant="transport",
        block_type="depthwise_separable",
        normalization="none",
        dilation_cycle=1,
        prognostic_channels=1,
        state_history=1,
        forcing_channels=1,
        forcing_integration="film",
        transport_displacement_scale=0.5,
        transport_correction_scale=0.25,
    )
    inputs = torch.zeros((2, 2, 8, 8))

    outputs = model(inputs)

    assert outputs.shape == (2, 1, 8, 8)


def test_transport_variant_supports_dual_branch_forcing_fusion():
    model = ResidualThicknessModel(
        input_channels=2,
        hidden_channels=4,
        num_blocks=2,
        kernel_size=3,
        model_variant="transport",
        block_type="depthwise_separable",
        normalization="none",
        dilation_cycle=1,
        prognostic_channels=1,
        state_history=1,
        forcing_channels=1,
        forcing_integration="dual_branch",
    )
    inputs = torch.zeros((2, 2, 8, 8))

    outputs = model(inputs)

    assert outputs.shape == (2, 1, 8, 8)


def test_transport_variant_supports_blockwise_film_conditioning():
    model = ResidualThicknessModel(
        input_channels=2,
        hidden_channels=4,
        num_blocks=2,
        kernel_size=3,
        model_variant="transport",
        block_type="depthwise_separable",
        normalization="none",
        dilation_cycle=1,
        prognostic_channels=1,
        state_history=1,
        forcing_channels=1,
        forcing_integration="film_per_block",
    )
    inputs = torch.zeros((2, 2, 8, 8))

    outputs = model(inputs)

    assert outputs.shape == (2, 1, 8, 8)


def test_transport_variant_supports_separate_heads():
    model = ResidualThicknessModel(
        input_channels=2,
        hidden_channels=4,
        num_blocks=2,
        kernel_size=3,
        model_variant="transport",
        block_type="depthwise_separable",
        normalization="none",
        dilation_cycle=1,
        prognostic_channels=1,
        state_history=1,
        forcing_channels=1,
        forcing_integration="film_per_block",
        transport_head_mode="separate",
    )
    inputs = torch.zeros((2, 2, 8, 8))

    outputs = model(inputs)

    assert outputs.shape == (2, 1, 8, 8)
