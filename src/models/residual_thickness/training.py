from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import xarray as xr

from src.models.cnn_thickness.data import Standardizer
from src.models.cnn_thickness.train import select_device, set_random_seed

from .config import ResidualThicknessConfig


@dataclass(frozen=True)
class ForcingStandardizer:
    mean: np.ndarray
    std: np.ndarray

    def normalize(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean[None, :, None, None]) / self.std[None, :, None, None]

    def denormalize(self, values: np.ndarray) -> np.ndarray:
        return (values * self.std[None, :, None, None]) + self.mean[None, :, None, None]


@dataclass(frozen=True)
class ChannelStandardizer:
    mean: np.ndarray
    std: np.ndarray

    def normalize(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean[None, :, None, None]) / self.std[None, :, None, None]

    def denormalize(self, values: np.ndarray) -> np.ndarray:
        return (values * self.std[None, :, None, None]) + self.mean[None, :, None, None]


def fit_channel_standardizer(train_frames: np.ndarray) -> ChannelStandardizer:
    mean = train_frames.mean(axis=(0, 2, 3), dtype=np.float64).astype(np.float32)
    std = train_frames.std(axis=(0, 2, 3), dtype=np.float64).astype(np.float32)
    std[std == 0.0] = 1.0
    return ChannelStandardizer(mean=mean, std=std)


def forcing_channel_count(forcing_mode: str) -> int:
    if forcing_mode == "none":
        return 0
    if forcing_mode in {"wind_current", "wind_next", "wind_delta"}:
        return 1
    if forcing_mode == "wind_current_plus_delta":
        return 2
    if forcing_mode in {"wind_mean_4", "wind_mean_12", "wind_mean_26", "wind_ema_4_12"}:
        return 1
    if forcing_mode in {"wind_current_mean_4_12", "wind_delta_mean_4_12", "wind_current_mean_12_anom_12"}:
        return 3
    raise ValueError(f"Unsupported forcing_mode: {forcing_mode}")


def load_forcing_dataset(netcdf_path: str, forcing_mode: str) -> np.ndarray | None:
    if forcing_mode == "none":
        return None
    dataset = xr.open_dataset(netcdf_path)
    try:
        time_days = np.asarray(dataset["time_days"].values, dtype=np.float32)
        y = np.asarray(dataset["y"].values, dtype=np.float32)
        x = np.asarray(dataset["x"].values, dtype=np.float32)
        y_grid = np.repeat(y[:, np.newaxis], x.size, axis=1)
        y_max = float(np.max(y))
        wind_stress_max = float(dataset.attrs["wind_stress_max"])
        shift_amplitude_m = float(dataset.attrs["wind_shift_amplitude_m"])
        shift_period_days = float(dataset.attrs["wind_shift_period_days"])

        wind = np.empty((time_days.size, y.size, x.size), dtype=np.float32)
        for index, time_day in enumerate(time_days):
            phase = 2.0 * np.pi * (float(time_day) / shift_period_days)
            shift_m = shift_amplitude_m * np.sin(phase)
            shifted_y = np.clip(y_grid - shift_m, 0.0, y_max)
            wind[index] = wind_stress_max * (1.0 - np.cos(2.0 * np.pi * shifted_y / y_max))
    finally:
        dataset.close()
    return wind


def _causal_mean(forcing: np.ndarray, window: int) -> np.ndarray:
    output = np.empty_like(forcing)
    cumulative = np.zeros_like(forcing[0], dtype=np.float32)
    for index in range(forcing.shape[0]):
        cumulative = cumulative + forcing[index]
        if index >= window:
            cumulative = cumulative - forcing[index - window]
        count = min(index + 1, window)
        output[index] = cumulative / float(count)
    return output


def _causal_ema(forcing: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / float(span + 1)
    output = np.empty_like(forcing)
    output[0] = forcing[0]
    for index in range(1, forcing.shape[0]):
        output[index] = alpha * forcing[index] + (1.0 - alpha) * output[index - 1]
    return output


def build_forcing_features(forcing: np.ndarray | None, forcing_mode: str) -> np.ndarray | None:
    if forcing_mode == "none":
        return None
    if forcing is None:
        raise ValueError("forcing is required when forcing_mode is not none")

    current = forcing.astype(np.float32)
    next_forcing = np.empty_like(current)
    next_forcing[:-1] = current[1:]
    next_forcing[-1] = current[-1]
    delta = next_forcing - current
    mean_4 = _causal_mean(current, 4)
    mean_12 = _causal_mean(current, 12)
    mean_26 = _causal_mean(current, 26)
    ema_4 = _causal_ema(current, 4)
    ema_12 = _causal_ema(current, 12)

    if forcing_mode == "wind_current":
        return current[:, np.newaxis, :, :]
    if forcing_mode == "wind_next":
        return next_forcing[:, np.newaxis, :, :]
    if forcing_mode == "wind_delta":
        return delta[:, np.newaxis, :, :]
    if forcing_mode == "wind_current_plus_delta":
        return np.stack([current, delta], axis=1)
    if forcing_mode == "wind_mean_4":
        return mean_4[:, np.newaxis, :, :]
    if forcing_mode == "wind_mean_12":
        return mean_12[:, np.newaxis, :, :]
    if forcing_mode == "wind_mean_26":
        return mean_26[:, np.newaxis, :, :]
    if forcing_mode == "wind_ema_4_12":
        return (ema_4 - ema_12)[:, np.newaxis, :, :]
    if forcing_mode == "wind_current_mean_4_12":
        return np.stack([current, mean_4, mean_12], axis=1)
    if forcing_mode == "wind_delta_mean_4_12":
        return np.stack([delta, mean_4, mean_12], axis=1)
    if forcing_mode == "wind_current_mean_12_anom_12":
        return np.stack([current, mean_12, current - mean_12], axis=1)
    raise ValueError(f"Unsupported forcing_mode: {forcing_mode}")


def fit_forcing_standardizer(forcing_features: np.ndarray | None) -> ForcingStandardizer | None:
    if forcing_features is None:
        return None
    mean = forcing_features.mean(axis=(0, 2, 3), dtype=np.float64).astype(np.float32)
    std = forcing_features.std(axis=(0, 2, 3), dtype=np.float64).astype(np.float32)
    std[std == 0.0] = 1.0
    return ForcingStandardizer(mean=mean, std=std)


def _state_channel_count(state_history: int) -> int:
    return max(state_history, 1)


def _ensure_channel_axis(frames: np.ndarray) -> np.ndarray:
    if frames.ndim == 3:
        return frames[:, np.newaxis, :, :]
    return frames


def center_zonal_velocity(values: np.ndarray) -> np.ndarray:
    return 0.5 * (values[..., :-1] + values[..., 1:])


def center_meridional_velocity(values: np.ndarray) -> np.ndarray:
    return 0.5 * (values[..., :-1, :] + values[..., 1:, :])


def compute_relative_vorticity(
    zonal_velocity_centered: np.ndarray,
    meridional_velocity_centered: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    dudy = np.gradient(zonal_velocity_centered, y, axis=-2, edge_order=2)
    dvdx = np.gradient(meridional_velocity_centered, x, axis=-1, edge_order=2)
    return (dvdx - dudy).astype(np.float32)


def load_state_fields(
    netcdf_path: str,
    state_fields: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dataset = xr.open_dataset(netcdf_path)
    try:
        time_days = np.asarray(dataset["time_days"].values, dtype=np.float32)
        y = np.asarray(dataset["y"].values, dtype=np.float32)
        x = np.asarray(dataset["x"].values, dtype=np.float32)

        cached_centered_u: np.ndarray | None = None
        cached_centered_v: np.ndarray | None = None
        channels: list[np.ndarray] = []
        for field_name in state_fields:
            if field_name == "layer_thickness":
                values = np.asarray(dataset["layer_thickness"].values, dtype=np.float32)
            elif field_name == "zonal_velocity_centered":
                if cached_centered_u is None:
                    raw_u = np.asarray(dataset["zonal_velocity"].values, dtype=np.float32)
                    cached_centered_u = center_zonal_velocity(raw_u)
                values = cached_centered_u
            elif field_name == "meridional_velocity_centered":
                if cached_centered_v is None:
                    raw_v = np.asarray(dataset["meridional_velocity"].values, dtype=np.float32)
                    cached_centered_v = center_meridional_velocity(raw_v)
                values = cached_centered_v
            elif field_name == "relative_vorticity":
                if cached_centered_u is None:
                    raw_u = np.asarray(dataset["zonal_velocity"].values, dtype=np.float32)
                    cached_centered_u = center_zonal_velocity(raw_u)
                if cached_centered_v is None:
                    raw_v = np.asarray(dataset["meridional_velocity"].values, dtype=np.float32)
                    cached_centered_v = center_meridional_velocity(raw_v)
                values = compute_relative_vorticity(cached_centered_u, cached_centered_v, y, x)
            else:
                raise ValueError(f"Unsupported state field: {field_name}")
            values = values.astype(np.float32)
            if values.ndim == 4:
                channels.extend([values[:, layer_index] for layer_index in range(values.shape[1])])
            else:
                channels.append(values)
    finally:
        dataset.close()

    frames = np.stack(channels, axis=1)
    return frames, time_days, y, x


def field_channel_indices(
    netcdf_path: str,
    state_fields: tuple[str, ...],
    field_name: str,
) -> tuple[int, ...]:
    dataset = xr.open_dataset(netcdf_path)
    try:
        if "layers" in dataset.coords:
            layer_count = int(dataset.sizes["layers"])
        else:
            layer_count = 1
    finally:
        dataset.close()

    indices: list[int] = []
    channel_offset = 0
    for current_field in state_fields:
        channels_for_field = layer_count if current_field in {
            "layer_thickness",
            "zonal_velocity_centered",
            "meridional_velocity_centered",
            "relative_vorticity",
        } else 1
        if current_field == field_name:
            indices.extend(range(channel_offset, channel_offset + channels_for_field))
        channel_offset += channels_for_field
    if not indices:
        raise ValueError(f"field_name {field_name!r} is not present in state_fields={state_fields!r}")
    return tuple(indices)


def _history_frame(frames: np.ndarray, index: int, offset: int) -> np.ndarray:
    history_index = max(index - offset, 0)
    return _ensure_channel_axis(frames)[history_index].astype(np.float32)


def build_input_channels(
    state_frames: list[np.ndarray],
    forcing_features: np.ndarray | None,
) -> np.ndarray:
    channels: list[np.ndarray] = []
    for frame in state_frames:
        channels.extend([channel.astype(np.float32) for channel in frame])
    if forcing_features is not None:
        channels.extend([feature.astype(np.float32) for feature in forcing_features])
    return np.stack(channels, axis=0)


def build_training_inputs(
    normalized_train_frames: np.ndarray,
    state_history: int,
    forcing_features: np.ndarray | None,
) -> np.ndarray:
    normalized_train_frames = _ensure_channel_axis(normalized_train_frames)
    inputs: list[np.ndarray] = []
    for index in range(normalized_train_frames.shape[0] - 1):
        state_frames = [_history_frame(normalized_train_frames, index, offset) for offset in range(_state_channel_count(state_history))]
        inputs.append(
            build_input_channels(
                state_frames,
                None if forcing_features is None else forcing_features[index],
            )
        )
    return np.stack(inputs, axis=0)


class ResidualAutoregressiveDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, inputs: np.ndarray, targets: np.ndarray):
        self.inputs = inputs
        self.targets = targets

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.inputs[index]), torch.from_numpy(self.targets[index])


def train_residual_model(
    config: ResidualThicknessConfig,
    model: torch.nn.Module,
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
) -> dict[str, float]:
    set_random_seed(config.random_seed)
    device = select_device()
    model.to(device)

    dataset = ResidualAutoregressiveDataset(train_inputs.astype(np.float32), train_targets.astype(np.float32))
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_fn = nn.MSELoss()
    final_loss = 0.0
    optimization_steps = 0
    epoch_train_losses: list[float] = []

    def _write_training_history(status: str) -> None:
        config.interim_experiment_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "experiment_id": config.resolved_experiment_id,
            "status": status,
            "epochs_completed": len(epoch_train_losses),
            "epochs_total": config.epochs,
            "epoch_train_losses": epoch_train_losses,
        }
        config.training_history_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _gradient_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pred_field = predictions[:, :1]
        target_field = targets[:, :1]
        pred_dx = pred_field[:, :, :, 1:] - pred_field[:, :, :, :-1]
        target_dx = target_field[:, :, :, 1:] - target_field[:, :, :, :-1]
        pred_dy = pred_field[:, :, 1:, :] - pred_field[:, :, :-1, :]
        target_dy = target_field[:, :, 1:, :] - target_field[:, :, :-1, :]
        return loss_fn(pred_dx, target_dx) + loss_fn(pred_dy, target_dy)

    model.train()
    _write_training_history(status="running")
    for _ in range(config.epochs):
        epoch_loss_total = 0.0
        epoch_steps = 0
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            predictions = model(inputs)
            loss = loss_fn(predictions, targets)
            if config.gradient_loss_weight > 0.0:
                loss = loss + (config.gradient_loss_weight * _gradient_loss(predictions, targets))
            loss.backward()
            optimizer.step()
            final_loss = float(loss.item())
            epoch_loss_total += final_loss
            epoch_steps += 1
            optimization_steps += 1
        epoch_train_losses.append(epoch_loss_total / max(epoch_steps, 1))
        _write_training_history(status="running")

    _write_training_history(status="completed")

    return {
        "train_loss": final_loss,
        "device": str(device),
        "optimization_steps": optimization_steps,
        "training_examples": len(dataset),
        "epoch_train_losses": epoch_train_losses,
    }


def autoregressive_rollout_with_forcing(
    model: torch.nn.Module,
    eval_frames: np.ndarray,
    state_history: int,
    forcing_features: np.ndarray | None,
    standardizer: Standardizer | ChannelStandardizer,
    device: torch.device,
) -> np.ndarray:
    if eval_frames.shape[0] < 2:
        raise ValueError("Evaluation segment must contain at least two timesteps.")

    model.eval()
    eval_frames = _ensure_channel_axis(eval_frames)
    current = standardizer.normalize(eval_frames[[0]])[0].astype(np.float32)
    state_buffer: deque[np.ndarray] = deque(
        [current.copy() for _ in range(_state_channel_count(state_history))],
        maxlen=_state_channel_count(state_history),
    )
    rollout_frames: list[np.ndarray] = []

    with torch.no_grad():
        index = 0
        while index < eval_frames.shape[0] - 1:
            model_inputs = build_input_channels(
                list(state_buffer),
                None if forcing_features is None else forcing_features[index],
            )
            input_tensor = torch.from_numpy(model_inputs).unsqueeze(0).to(device)
            prediction = model(input_tensor).squeeze(0).cpu().numpy()
            if prediction.ndim == 3:
                prediction_steps = (prediction,)
            elif prediction.ndim == 4:
                prediction_steps = tuple(prediction[step_index] for step_index in range(prediction.shape[0]))
            else:
                raise ValueError(f"Unsupported prediction rank during rollout: {prediction.ndim}")
            for step_prediction in prediction_steps:
                if index >= eval_frames.shape[0] - 1:
                    break
                rollout_frames.append(standardizer.denormalize(step_prediction[None, ...])[0])
                current = step_prediction.astype(np.float32)
                state_buffer.appendleft(current.copy())
                index += 1

    return np.stack(rollout_frames, axis=0)
