from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import time
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.models.cnn_thickness.train import select_device, set_random_seed
from src.models.residual_thickness.training import build_input_channels

from .config import UnetThicknessConfig


@dataclass(frozen=True)
class RolloutCurriculum:
    rollout_steps: tuple[int, ...]
    transition_epochs: tuple[int, ...]

    def horizon_for_epoch(self, epoch_index: int) -> int:
        selected_horizon = self.rollout_steps[0]
        for transition_epoch, rollout_step in zip(self.transition_epochs, self.rollout_steps):
            if epoch_index >= transition_epoch:
                selected_horizon = rollout_step
        return selected_horizon


class RolloutStartDataset(Dataset[int]):
    def __init__(self, num_frames: int, rollout_horizon: int):
        self.num_examples = max(num_frames - rollout_horizon, 0)

    def __len__(self) -> int:
        return self.num_examples

    def __getitem__(self, index: int) -> int:
        return index


def build_rollout_curriculum(config: UnetThicknessConfig) -> RolloutCurriculum:
    if len(config.curriculum_rollout_steps) != len(config.curriculum_transition_epochs):
        raise ValueError("curriculum_rollout_steps and curriculum_transition_epochs must have the same length.")
    if len(config.curriculum_rollout_steps) == 0:
        raise ValueError("curriculum_rollout_steps must not be empty.")
    return RolloutCurriculum(
        rollout_steps=tuple(max(int(step), 1) for step in config.curriculum_rollout_steps),
        transition_epochs=tuple(max(int(epoch), 0) for epoch in config.curriculum_transition_epochs),
    )


def _assemble_state_history(
    frames: torch.Tensor,
    start_indices: torch.Tensor,
    state_history: int,
) -> torch.Tensor:
    history_frames: list[torch.Tensor] = []
    for offset in range(max(state_history, 1)):
        history_indices = torch.clamp(start_indices - offset, min=0)
        history_frames.append(frames.index_select(0, history_indices))
    return torch.stack(history_frames, dim=1)


def _assemble_model_inputs(
    state_history_tensor: torch.Tensor,
    current_forcing_features: torch.Tensor | None,
    state_input_mode: str = "history",
) -> torch.Tensor:
    if state_input_mode not in {"history", "current_plus_residual", "residual_only"}:
        raise ValueError(f"Unsupported state_input_mode: {state_input_mode}")

    if state_history_tensor.device.type == "cpu":
        inputs: list[np.ndarray] = []
        state_history_np = state_history_tensor.detach().numpy()
        current_forcing_np = None if current_forcing_features is None else current_forcing_features.detach().numpy()
        for batch_index in range(state_history_np.shape[0]):
            current = state_history_np[batch_index, 0]
            previous = state_history_np[batch_index, 1] if state_history_np.shape[1] > 1 else current
            residual = current - previous
            if state_input_mode == "history":
                state_frames = [state_history_np[batch_index, index] for index in range(state_history_np.shape[1])]
            elif state_input_mode == "current_plus_residual":
                state_frames = [current, residual]
            else:
                state_frames = [residual]
            inputs.append(
                build_input_channels(
                    state_frames,
                    None if current_forcing_np is None else current_forcing_np[batch_index],
                )
            )
        return torch.from_numpy(np.stack(inputs, axis=0))

    batch_size, history_length, channel_count, height, width = state_history_tensor.shape
    current = state_history_tensor[:, 0]
    previous = state_history_tensor[:, 1] if history_length > 1 else current
    residual = current - previous
    if state_input_mode == "history":
        state_inputs = state_history_tensor.reshape(batch_size, history_length * channel_count, height, width)
    elif state_input_mode == "current_plus_residual":
        state_inputs = torch.cat([current, residual], dim=1)
    else:
        state_inputs = residual

    if current_forcing_features is None:
        return state_inputs
    return torch.cat([state_inputs, current_forcing_features], dim=1)


def _laplacian_energy(values: torch.Tensor) -> torch.Tensor:
    horizontal = values[:, :, :, 2:] - (2.0 * values[:, :, :, 1:-1]) + values[:, :, :, :-2]
    vertical = values[:, :, 2:, :] - (2.0 * values[:, :, 1:-1, :]) + values[:, :, :-2, :]
    return horizontal.square().mean() + vertical.square().mean()


def _prediction_sequence(predictions: torch.Tensor) -> tuple[torch.Tensor, ...]:
    if predictions.ndim == 4:
        return (predictions,)
    if predictions.ndim == 5:
        return tuple(predictions[:, step_index] for step_index in range(predictions.shape[1]))
    raise ValueError(f"Unsupported prediction rank: {predictions.ndim}")


PeriodicEvalCallback = Callable[[int], dict[str, object] | None]


def train_unet_model(
    config: UnetThicknessConfig,
    model: torch.nn.Module,
    normalized_train_frames: np.ndarray,
    forcing_features: np.ndarray | None,
    periodic_eval_callback: PeriodicEvalCallback | None = None,
) -> dict[str, float]:
    set_random_seed(config.random_seed)
    device = select_device()
    should_use_mps = bool(torch.backends.mps.is_available())
    model.to(device)

    frames_tensor = torch.from_numpy(normalized_train_frames.astype(np.float32)).to(device)
    forcing_tensor = None if forcing_features is None else torch.from_numpy(forcing_features.astype(np.float32)).to(device)
    curriculum = build_rollout_curriculum(config)
    dataloader_generator = torch.Generator(device="cpu")
    dataloader_generator.manual_seed(config.random_seed)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_fn = nn.MSELoss()
    final_loss = 0.0
    optimization_steps = 0
    last_rollout_horizon = 1
    operator_output_steps = max(config.output_steps, 1)
    epoch_train_losses: list[float] = []
    epoch_length_seconds_per_epoch: list[float] = []
    periodic_eval_results: list[dict[str, object]] = []
    stopped_early = False
    stop_reason: str | None = None

    def _updated_at() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _write_training_history(status: str) -> None:
        config.interim_experiment_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "experiment_id": config.resolved_experiment_id,
            "hypothesis": config.hypothesis,
            "updated_at": _updated_at(),
            "status": status,
            "device": str(device),
            "should_use_mps": should_use_mps,
            "epochs_completed": len(epoch_train_losses),
            "epochs_total": config.epochs,
            "epoch_train_losses": epoch_train_losses,
            "epoch_length_seconds": (
                None
                if not epoch_length_seconds_per_epoch
                else float(sum(epoch_length_seconds_per_epoch) / len(epoch_length_seconds_per_epoch))
            ),
            "epoch_length_seconds_per_epoch": epoch_length_seconds_per_epoch,
            "early_stopping_eval_interval_epochs": config.early_stopping_eval_interval_epochs,
            "periodic_eval_results": periodic_eval_results,
            "stopped_early": stopped_early,
            "stop_reason": stop_reason,
        }
        config.training_history_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    model.train()
    _write_training_history(status="running")
    for epoch_index in range(config.epochs):
        epoch_started_at = time.perf_counter()
        rollout_horizon = curriculum.horizon_for_epoch(epoch_index)
        last_rollout_horizon = rollout_horizon
        scheduled_sampling_prob = config.scheduled_sampling_max_prob * float(epoch_index + 1) / float(max(config.epochs, 1))
        dataset = RolloutStartDataset(frames_tensor.shape[0], rollout_horizon)
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            generator=dataloader_generator,
        )
        epoch_loss_total = 0.0
        epoch_steps = 0
        for start_indices in dataloader:
            start_indices = start_indices.to(device=device, dtype=torch.long)
            optimizer.zero_grad()

            state_history_tensor = _assemble_state_history(frames_tensor, start_indices, config.state_history)
            total_loss = torch.tensor(0.0, device=device)
            rollout_offset = 0

            while rollout_offset < rollout_horizon:
                current_indices = start_indices + rollout_offset
                current_forcing = None if forcing_tensor is None else forcing_tensor.index_select(0, current_indices)
                model_inputs = _assemble_model_inputs(
                    state_history_tensor,
                    current_forcing,
                    state_input_mode=config.state_input_mode,
                )
                predictions = model(model_inputs)
                prediction_steps = _prediction_sequence(predictions)
                steps_to_apply = min(len(prediction_steps), rollout_horizon - rollout_offset, operator_output_steps)
                for prediction_step in range(steps_to_apply):
                    step_prediction = prediction_steps[prediction_step]
                    targets = frames_tensor.index_select(0, current_indices + prediction_step + 1)
                    total_loss = total_loss + loss_fn(step_prediction, targets)
                    if config.high_frequency_loss_weight > 0.0:
                        total_loss = total_loss + (config.high_frequency_loss_weight * _laplacian_energy(step_prediction))
                    next_state = step_prediction
                    if scheduled_sampling_prob > 0.0:
                        teacher_mask = (
                            torch.rand(step_prediction.shape[0], 1, 1, 1, device=device) < scheduled_sampling_prob
                        )
                        next_state = torch.where(teacher_mask, targets, step_prediction)
                    if config.state_history > 1:
                        state_history_tensor = torch.cat(
                            [next_state.unsqueeze(1), state_history_tensor[:, : config.state_history - 1]],
                            dim=1,
                        )
                    else:
                        state_history_tensor = next_state.unsqueeze(1)
                rollout_offset += steps_to_apply

            loss = total_loss / rollout_horizon
            loss.backward()
            optimizer.step()
            final_loss = float(loss.item())
            epoch_loss_total += final_loss
            epoch_steps += 1
            optimization_steps += 1
        epoch_train_losses.append(epoch_loss_total / max(epoch_steps, 1))
        epoch_length_seconds_per_epoch.append(time.perf_counter() - epoch_started_at)
        if (
            periodic_eval_callback is not None
            and config.early_stopping_eval_interval_epochs > 0
            and (epoch_index + 1) % config.early_stopping_eval_interval_epochs == 0
        ):
            model.eval()
            with torch.no_grad():
                eval_result = periodic_eval_callback(epoch_index + 1)
            model.train()
            if eval_result is not None:
                periodic_eval_results.append(dict(eval_result))
                if bool(eval_result.get("stop_training", False)):
                    stopped_early = True
                    reason = eval_result.get("stop_reason")
                    stop_reason = None if reason is None else str(reason)
        _write_training_history(status="stopped" if stopped_early else "running")
        if stopped_early:
            break

    _write_training_history(status="stopped" if stopped_early else "completed")

    return {
        "train_loss": final_loss,
        "device": str(device),
        "optimization_steps": optimization_steps,
        "training_examples": max(frames_tensor.shape[0] - last_rollout_horizon, 0),
        "curriculum_final_rollout_horizon": last_rollout_horizon,
        "epoch_train_losses": epoch_train_losses,
        "epoch_length_seconds": (
            0.0 if not epoch_length_seconds_per_epoch else float(sum(epoch_length_seconds_per_epoch) / len(epoch_length_seconds_per_epoch))
        ),
        "epoch_length_seconds_per_epoch": epoch_length_seconds_per_epoch,
        "epochs_completed": len(epoch_train_losses),
        "periodic_eval_results": periodic_eval_results,
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "should_use_mps": should_use_mps,
    }
