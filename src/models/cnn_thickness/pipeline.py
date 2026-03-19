from __future__ import annotations

from dataclasses import asdict

import numpy as np
import torch

from .animation import create_rollout_comparison_animation
from .config import CnnThicknessConfig, load_cnn_thickness_config
from .data import fit_standardizer, load_field_dataset, split_sequence
from .evaluate import mean_squared_error, mse_per_timestep, save_metrics, save_rollout_dataset
from .model import CnnThicknessModel
from .rollout import autoregressive_rollout
from .train import select_device, set_random_seed, train_model


def run_cnn_thickness_experiment(config: CnnThicknessConfig | str) -> dict[str, str | float | int]:
    if not isinstance(config, CnnThicknessConfig):
        config = load_cnn_thickness_config(config)
    config = config.resolve_experiment()

    set_random_seed(config.random_seed)
    frames, time_days, y, x = load_field_dataset(str(config.source_netcdf_path), config.field_name)
    split = split_sequence(frames, time_days, config.train_fraction)
    standardizer = fit_standardizer(split.train_frames)

    model = CnnThicknessModel(
        hidden_channels=config.hidden_channels,
        num_layers=config.num_layers,
        kernel_size=config.kernel_size,
    )
    train_info = train_model(config, model, standardizer.normalize(split.train_frames))
    device = torch.device(train_info["device"])

    rollout = autoregressive_rollout(model, split.eval_frames, standardizer, device)
    truth = split.eval_frames[1:]
    eval_time_days = split.eval_time_days[1:]
    mse = mean_squared_error(truth, rollout)
    per_timestep_mse = mse_per_timestep(truth, rollout)

    config.interim_experiment_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "normalization": {"mean": standardizer.mean, "std": standardizer.std},
        },
        config.checkpoint_path,
    )
    save_rollout_dataset(config.rollout_path, truth=truth, rollout=rollout, time_days=eval_time_days, y=y, x=x)
    create_rollout_comparison_animation(config.rollout_path, config.animation_path, fps=config.animation_fps)

    metrics = {
        "source_experiment_id": config.source_experiment_id,
        "emulator_experiment_id": config.resolved_experiment_id,
        "architecture": "cnn_thickness",
        "hidden_channels": config.hidden_channels,
        "num_layers": config.num_layers,
        "kernel_size": config.kernel_size,
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
    }
