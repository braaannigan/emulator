import json
from pathlib import Path

import numpy as np
import torch

from src.models.unet_thickness.config import load_unet_thickness_config
from src.models.unet_thickness.model import UnetThicknessModel
from src.models.unet_thickness.training import build_rollout_curriculum, train_unet_model


def test_build_rollout_curriculum_selects_horizon_by_epoch():
    config = load_unet_thickness_config("config/emulator/unet_thickness.yaml").with_overrides(
        curriculum_rollout_steps=(1, 2, 4),
        curriculum_transition_epochs=(0, 10, 20),
    )

    curriculum = build_rollout_curriculum(config)

    assert curriculum.horizon_for_epoch(0) == 1
    assert curriculum.horizon_for_epoch(9) == 1
    assert curriculum.horizon_for_epoch(10) == 2
    assert curriculum.horizon_for_epoch(25) == 4


def test_train_unet_model_writes_training_history(tmp_path: Path):
    config = load_unet_thickness_config("config/emulator/unet_thickness.yaml").with_overrides(
        interim_output_root=tmp_path / "interim",
        raw_output_root=tmp_path / "raw",
        experiment_id="unit-unet-history",
        epochs=2,
        batch_size=1,
    )
    model = UnetThicknessModel(
        input_channels=1,
        hidden_channels=4,
        num_levels=1,
        kernel_size=3,
        state_channels=1,
        forcing_channels=0,
        fusion_mode="input",
        residual_connection=True,
        prognostic_channels=1,
    )
    frames = np.arange(4 * 1 * 4 * 4, dtype=np.float32).reshape(4, 1, 4, 4)

    train_unet_model(config, model, frames, None)

    history = json.loads(config.training_history_path.read_text(encoding="utf-8"))
    assert history["status"] == "completed"
    assert history["epochs_completed"] == 2
    assert history["epochs_total"] == 2
    assert len(history["epoch_train_losses"]) == 2


def test_train_unet_model_supports_stabilization_options(tmp_path: Path):
    config = load_unet_thickness_config("config/emulator/unet_thickness.yaml").with_overrides(
        interim_output_root=tmp_path / "interim",
        raw_output_root=tmp_path / "raw",
        experiment_id="unit-unet-stabilization",
        epochs=2,
        batch_size=1,
        residual_connection=True,
        residual_step_scale=0.5,
        curriculum_rollout_steps=(2,),
        curriculum_transition_epochs=(0,),
        scheduled_sampling_max_prob=0.5,
        high_frequency_loss_weight=0.01,
    )
    model = UnetThicknessModel(
        input_channels=1,
        hidden_channels=4,
        num_levels=1,
        kernel_size=3,
        state_channels=1,
        forcing_channels=0,
        fusion_mode="input",
        residual_connection=True,
        residual_step_scale=0.5,
        prognostic_channels=1,
    )
    frames = np.arange(5 * 1 * 4 * 4, dtype=np.float32).reshape(5, 1, 4, 4)

    train_info = train_unet_model(config, model, frames, None)

    assert train_info["curriculum_final_rollout_horizon"] == 2
    history = json.loads(config.training_history_path.read_text(encoding="utf-8"))
    assert history["status"] == "completed"
