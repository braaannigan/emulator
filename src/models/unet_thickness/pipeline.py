from __future__ import annotations

from dataclasses import asdict

import torch

from src.models.cnn_thickness.animation import create_rollout_comparison_animation
from src.models.cnn_thickness.data import split_sequence
from src.models.cnn_thickness.evaluate import mean_squared_error, mse_per_timestep, save_metrics, save_rollout_dataset
from src.models.cnn_thickness.train import set_random_seed
from src.models.residual_thickness.training import (
    autoregressive_rollout_with_forcing,
    build_forcing_features,
    build_training_inputs,
    fit_channel_standardizer,
    fit_forcing_standardizer,
    forcing_channel_count,
    load_forcing_dataset,
    load_state_fields,
)

from .config import UnetThicknessConfig, load_unet_thickness_config
from .model import UnetThicknessModel
from .training import build_rollout_curriculum, train_unet_model


def _trim_initial_spinup(
    frames: torch.Tensor | object,
    time_days: torch.Tensor | object,
    train_start_day: float,
):
    if train_start_day <= 0.0:
        return frames, time_days
    eval_mask = time_days >= train_start_day
    if not bool(eval_mask.any()):
        raise ValueError(f"No timesteps remain after applying train_start_day={train_start_day}.")
    first_index = int(eval_mask.argmax())
    return frames[first_index:], time_days[first_index:]


def run_unet_thickness_experiment(config: UnetThicknessConfig | str) -> dict[str, str | float | int]:
    if not isinstance(config, UnetThicknessConfig):
        config = load_unet_thickness_config(config)
    config = config.resolve_experiment()

    set_random_seed(config.random_seed)
    frames, time_days, y, x = load_state_fields(str(config.source_netcdf_path), config.state_fields)
    frames, time_days = _trim_initial_spinup(frames, time_days, config.train_start_day)
    split = split_sequence(frames, time_days, config.train_fraction)
    standardizer = fit_channel_standardizer(split.train_frames)
    forcing = load_forcing_dataset(str(config.source_netcdf_path), config.forcing_mode)
    forcing_features = build_forcing_features(forcing, config.forcing_mode)
    forcing_standardizer = fit_forcing_standardizer(
        None if forcing_features is None else forcing_features[: split.train_frames.shape[0]]
    )
    normalized_forcing_features = (
        None if forcing_standardizer is None or forcing_features is None else forcing_standardizer.normalize(forcing_features)
    )

    model = UnetThicknessModel(
        input_channels=(len(config.state_fields) * config.state_history) + forcing_channel_count(config.forcing_mode),
        hidden_channels=config.hidden_channels,
        num_levels=config.num_levels,
        kernel_size=config.kernel_size,
        block_type=config.block_type,
        state_channels=len(config.state_fields) * config.state_history,
        forcing_channels=forcing_channel_count(config.forcing_mode),
        fusion_mode=config.fusion_mode,
        residual_connection=config.residual_connection,
        residual_step_scale=config.residual_step_scale,
        prognostic_channels=len(config.state_fields),
    )
    train_info = train_unet_model(
        config,
        model,
        standardizer.normalize(split.train_frames),
        None if normalized_forcing_features is None else normalized_forcing_features[: split.train_frames.shape[0]],
    )
    device = torch.device(train_info["device"])

    eval_forcing = None if normalized_forcing_features is None else normalized_forcing_features[split.train_frames.shape[0] :]
    rollout = autoregressive_rollout_with_forcing(
        model,
        split.eval_frames,
        config.state_history,
        eval_forcing,
        standardizer,
        device,
    )
    field_index = config.state_fields.index(config.field_name)
    truth = split.eval_frames[1:, field_index]
    rollout_field = rollout[:, field_index]
    eval_time_days = split.eval_time_days[1:]
    if config.eval_window_days is not None:
        window_limit = float(eval_time_days[0] + config.eval_window_days)
        eval_mask = eval_time_days <= window_limit
        truth = truth[eval_mask]
        rollout_field = rollout_field[eval_mask]
        eval_time_days = eval_time_days[eval_mask]
    mse = mean_squared_error(truth, rollout_field)
    per_timestep_mse = mse_per_timestep(truth, rollout_field)

    config.interim_experiment_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "normalization": {"mean": standardizer.mean.tolist(), "std": standardizer.std.tolist()},
            "forcing_normalization": None
            if forcing_standardizer is None
            else {
                "mean": forcing_standardizer.mean.tolist(),
                "std": forcing_standardizer.std.tolist(),
            },
        },
        config.checkpoint_path,
    )
    save_rollout_dataset(config.rollout_path, truth=truth, rollout=rollout_field, time_days=eval_time_days, y=y, x=x)
    if config.animation_fps > 0:
        create_rollout_comparison_animation(config.rollout_path, config.animation_path, fps=config.animation_fps)

    metrics = {
        "source_experiment_id": config.source_experiment_id,
        "emulator_experiment_id": config.resolved_experiment_id,
        "architecture": "unet_thickness",
        "hidden_channels": config.hidden_channels,
        "num_levels": config.num_levels,
        "kernel_size": config.kernel_size,
        "block_type": config.block_type,
        "state_history": config.state_history,
        "state_fields": list(config.state_fields),
        "forcing_mode": config.forcing_mode,
        "fusion_mode": config.fusion_mode,
        "residual_connection": config.residual_connection,
        "residual_step_scale": config.residual_step_scale,
        "scheduled_sampling_max_prob": config.scheduled_sampling_max_prob,
        "high_frequency_loss_weight": config.high_frequency_loss_weight,
        "eval_window_days": config.eval_window_days,
        "curriculum_rollout_steps": [int(value) for value in config.curriculum_rollout_steps],
        "curriculum_transition_epochs": [int(value) for value in config.curriculum_transition_epochs],
        "curriculum_final_rollout_horizon": int(train_info["curriculum_final_rollout_horizon"]),
        "train_start_day": config.train_start_day,
        "epochs": config.epochs,
        "train_timesteps": int(split.train_frames.shape[0]),
        "eval_timesteps": int(truth.shape[0]),
        "training_examples": int(train_info["training_examples"]),
        "optimization_steps": int(train_info["optimization_steps"]),
        "mse": mse,
        "eval_mse_mean": float(per_timestep_mse.mean()),
        "eval_mse_std": float(per_timestep_mse.std()),
        "eval_mse_min": float(per_timestep_mse.min()),
        "eval_mse_max": float(per_timestep_mse.max()),
        "train_loss": float(train_info["train_loss"]),
        "eval_time_days": [float(value) for value in eval_time_days.tolist()],
        "eval_mse_per_timestep": [float(value) for value in per_timestep_mse.tolist()],
    }
    save_metrics(config.metrics_path, metrics)
    return {
        "metrics_path": str(config.metrics_path),
        "rollout_path": str(config.rollout_path),
        "animation_path": str(config.animation_path),
        "checkpoint_path": str(config.checkpoint_path),
        "mse": mse,
        "eval_mse_mean": float(per_timestep_mse.mean()),
        "eval_mse_last": float(per_timestep_mse[-1]),
        "train_loss": float(train_info["train_loss"]),
    }
