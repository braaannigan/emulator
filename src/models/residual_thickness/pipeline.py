from __future__ import annotations

from dataclasses import asdict

import torch

from src.models.cnn_thickness.data import split_sequence
from src.models.cnn_thickness.evaluate import mean_squared_error, mse_per_timestep, save_metrics, save_rollout_dataset
from src.models.cnn_thickness.train import set_random_seed

from .config import ResidualThicknessConfig, load_residual_thickness_config
from .model import ResidualThicknessModel
from .training import (
    autoregressive_rollout_with_forcing,
    build_forcing_features,
    build_training_inputs,
    fit_channel_standardizer,
    fit_forcing_standardizer,
    forcing_channel_count,
    load_forcing_dataset,
    load_state_fields,
    train_residual_model,
)


def run_residual_thickness_experiment(config: ResidualThicknessConfig | str) -> dict[str, str | float | int]:
    if not isinstance(config, ResidualThicknessConfig):
        config = load_residual_thickness_config(config)
    config = config.resolve_experiment()

    set_random_seed(config.random_seed)
    frames, time_days, y, x = load_state_fields(str(config.source_netcdf_path), config.state_fields)
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

    model = ResidualThicknessModel(
        input_channels=(len(config.state_fields) * config.state_history) + forcing_channel_count(config.forcing_mode),
        hidden_channels=config.hidden_channels,
        num_blocks=config.num_blocks,
        kernel_size=config.kernel_size,
        model_variant=config.model_variant,
        block_type=config.block_type,
        normalization=config.normalization,
        dilation_cycle=config.dilation_cycle,
        prognostic_channels=len(config.state_fields),
        state_history=config.state_history,
        forcing_channels=forcing_channel_count(config.forcing_mode),
        forcing_integration=config.forcing_integration,
        transport_displacement_scale=config.transport_displacement_scale,
        transport_correction_scale=config.transport_correction_scale,
        transport_head_mode=config.transport_head_mode,
    )
    train_inputs = build_training_inputs(
        standardizer.normalize(split.train_frames),
        config.state_history,
        None if normalized_forcing_features is None else normalized_forcing_features[: split.train_frames.shape[0]],
    )
    train_targets = standardizer.normalize(split.train_frames[1:])
    train_info = train_residual_model(config, model, train_inputs, train_targets)
    device = torch.device(train_info["device"])

    eval_forcing_features = (
        None if normalized_forcing_features is None else normalized_forcing_features[split.train_frames.shape[0] :]
    )
    rollout = autoregressive_rollout_with_forcing(
        model,
        split.eval_frames,
        config.state_history,
        eval_forcing_features,
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
        from src.models.cnn_thickness.animation import create_rollout_comparison_animation

        create_rollout_comparison_animation(config.rollout_path, config.animation_path, fps=config.animation_fps)

    metrics = {
        "source_experiment_id": config.source_experiment_id,
        "emulator_experiment_id": config.resolved_experiment_id,
        "architecture": "residual_thickness",
        "hidden_channels": config.hidden_channels,
        "num_blocks": config.num_blocks,
        "kernel_size": config.kernel_size,
        "model_variant": config.model_variant,
        "block_type": config.block_type,
        "normalization": config.normalization,
        "dilation_cycle": config.dilation_cycle,
        "state_history": config.state_history,
        "state_fields": list(config.state_fields),
        "forcing_mode": config.forcing_mode,
        "forcing_integration": config.forcing_integration,
        "gradient_loss_weight": config.gradient_loss_weight,
        "eval_window_days": config.eval_window_days,
        "transport_displacement_scale": config.transport_displacement_scale,
        "transport_correction_scale": config.transport_correction_scale,
        "transport_head_mode": config.transport_head_mode,
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
