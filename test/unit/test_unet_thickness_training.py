import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

from src.models.unet_thickness.config import load_unet_thickness_config
from src.models.unet_thickness.model import UnetThicknessModel
from src.models.unet_thickness.training import _assemble_model_inputs, build_rollout_curriculum, train_unet_model


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


def test_assemble_model_inputs_flattens_history_then_appends_forcing():
    state_history_tensor = torch.tensor(
        [
            [
                [[[1.0]], [[2.0]]],
                [[[3.0]], [[4.0]]],
            ]
        ],
        dtype=torch.float32,
    )
    forcing_tensor = torch.tensor([[[[5.0]], [[6.0]]]], dtype=torch.float32)

    assembled = _assemble_model_inputs(state_history_tensor, forcing_tensor)

    assert assembled.shape == (1, 6, 1, 1)
    assert assembled[:, :, 0, 0].tolist() == [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]


def test_assemble_model_inputs_supports_current_plus_residual_mode():
    state_history_tensor = torch.tensor(
        [
            [
                [[[10.0]], [[20.0]]],
                [[[7.0]], [[14.0]]],
            ]
        ],
        dtype=torch.float32,
    )
    forcing_tensor = torch.tensor([[[[1.0]]]], dtype=torch.float32)

    assembled = _assemble_model_inputs(state_history_tensor, forcing_tensor, state_input_mode="current_plus_residual")

    assert assembled.shape == (1, 5, 1, 1)
    assert assembled[:, :, 0, 0].tolist() == [[10.0, 20.0, 3.0, 6.0, 1.0]]


def test_assemble_model_inputs_supports_residual_only_mode():
    state_history_tensor = torch.tensor(
        [
            [
                [[[10.0]], [[20.0]]],
                [[[7.0]], [[14.0]]],
            ]
        ],
        dtype=torch.float32,
    )
    forcing_tensor = torch.tensor([[[[1.0]], [[2.0]]]], dtype=torch.float32)

    assembled = _assemble_model_inputs(state_history_tensor, forcing_tensor, state_input_mode="residual_only")

    assert assembled.shape == (1, 4, 1, 1)
    assert assembled[:, :, 0, 0].tolist() == [[3.0, 6.0, 1.0, 2.0]]


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
    assert history["hypothesis"] is None
    assert history["epochs_completed"] == 2
    assert history["epochs_total"] == 2
    assert len(history["epoch_train_losses"]) == 2
    assert isinstance(history["updated_at"], str)
    datetime.fromisoformat(history["updated_at"])
    assert isinstance(history["should_use_mps"], bool)
    assert isinstance(history["epoch_length_seconds"], float)
    assert len(history["epoch_length_seconds_per_epoch"]) == 2


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
    assert len(history["epoch_length_seconds_per_epoch"]) == 2


def test_train_unet_model_supports_multistep_operator(tmp_path: Path):
    config = load_unet_thickness_config("config/emulator/unet_thickness.yaml").with_overrides(
        interim_output_root=tmp_path / "interim",
        raw_output_root=tmp_path / "raw",
        experiment_id="unit-unet-multistep",
        epochs=1,
        batch_size=1,
        state_history=2,
        output_steps=2,
        curriculum_rollout_steps=(4,),
        curriculum_transition_epochs=(0,),
    )
    model = UnetThicknessModel(
        input_channels=2,
        hidden_channels=4,
        num_levels=1,
        kernel_size=3,
        stage_depth=1,
        dilation_cycle=2,
        state_channels=2,
        output_steps=2,
        forcing_channels=0,
        fusion_mode="input",
        residual_connection=False,
        prognostic_channels=1,
    )
    frames = np.arange(6 * 1 * 4 * 4, dtype=np.float32).reshape(6, 1, 4, 4)

    train_info = train_unet_model(config, model, frames, None)

    assert train_info["curriculum_final_rollout_horizon"] == 4
    history = json.loads(config.training_history_path.read_text(encoding="utf-8"))
    assert history["status"] == "completed"
    assert len(history["epoch_length_seconds_per_epoch"]) == 1


def test_train_unet_model_records_periodic_eval_results_and_stops_early(tmp_path: Path):
    config = load_unet_thickness_config("config/emulator/unet_thickness.yaml").with_overrides(
        interim_output_root=tmp_path / "interim",
        raw_output_root=tmp_path / "raw",
        experiment_id="unit-unet-early-stop",
        epochs=6,
        batch_size=1,
        early_stopping_eval_interval_epochs=2,
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
    frames = np.arange(8 * 1 * 4 * 4, dtype=np.float32).reshape(8, 1, 4, 4)

    def periodic_eval(epoch: int) -> dict[str, object]:
        return {
            "epoch": epoch,
            "eval_mse_mean": float(epoch),
            "eval_mse_last": float(epoch + 1),
            "stop_training": epoch >= 4,
            "stop_reason": "unit-test threshold crossed" if epoch >= 4 else None,
        }

    train_info = train_unet_model(config, model, frames, None, periodic_eval_callback=periodic_eval)

    history = json.loads(config.training_history_path.read_text(encoding="utf-8"))
    assert history["status"] == "stopped"
    assert history["epochs_completed"] == 4
    assert history["stopped_early"] is True
    assert history["stop_reason"] == "unit-test threshold crossed"
    assert [entry["epoch"] for entry in history["periodic_eval_results"]] == [2, 4]
    assert len(history["epoch_length_seconds_per_epoch"]) == 4
    assert train_info["epochs_completed"] == 4
    assert train_info["stopped_early"] is True
    assert isinstance(train_info["should_use_mps"], bool)
