from src.models.unet_thickness.config import load_unet_thickness_config


def test_load_unet_thickness_config_reads_architecture_fields():
    config = load_unet_thickness_config("config/emulator/unet_thickness.yaml")

    assert config.state_fields == ("layer_thickness",)
    assert config.hidden_channels == 16
    assert config.num_levels == 2
    assert config.kernel_size == 3
    assert config.block_type == "standard"
    assert config.state_history == 1
    assert config.forcing_mode == "none"
    assert config.fusion_mode == "input"
    assert config.residual_connection is True
    assert config.residual_step_scale == 1.0
    assert config.curriculum_rollout_steps == (1,)
    assert config.curriculum_transition_epochs == (0,)
    assert config.scheduled_sampling_max_prob == 0.0
    assert config.high_frequency_loss_weight == 0.0
    assert config.train_start_day == 0.0
    assert config.eval_window_days is None
