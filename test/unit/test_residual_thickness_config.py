from src.models.residual_thickness.config import load_residual_thickness_config


def test_load_residual_thickness_config_reads_architecture_fields():
    config = load_residual_thickness_config("config/emulator/residual_thickness_depthwise.yaml")

    assert config.block_type == "depthwise_separable"
    assert config.normalization == "none"
    assert config.dilation_cycle == 1
    assert config.hidden_channels == 16
    assert config.num_blocks == 4
