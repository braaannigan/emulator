from src.models.residual_thickness.config import load_residual_thickness_config


def test_load_residual_thickness_config_reads_architecture_fields():
    config = load_residual_thickness_config("config/emulator/residual_thickness_depthwise.yaml")

    assert config.block_type == "depthwise_separable"
    assert config.normalization == "none"
    assert config.dilation_cycle == 1
    assert config.state_history == 1
    assert config.state_fields == ("layer_thickness",)
    assert config.model_variant == "residual"
    assert config.forcing_mode == "none"
    assert config.forcing_integration == "concat"
    assert config.gradient_loss_weight == 0.0
    assert config.eval_window_days is None
    assert config.transport_displacement_scale == 1.0
    assert config.transport_correction_scale == 1.0
    assert config.transport_head_mode == "shared"
    assert config.hidden_channels == 16
    assert config.num_blocks == 4
