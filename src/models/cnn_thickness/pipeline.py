from __future__ import annotations

from dataclasses import asdict

import numpy as np
import torch

from .animation import create_rollout_comparison_animation
from .config import CnnThicknessConfig, load_cnn_thickness_config
from .data import fit_standardizer, load_field_dataset, split_sequence
from .evaluate import mean_squared_error, save_metrics, save_rollout_dataset
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
        "train_timesteps": int(split.train_frames.shape[0]),
        "eval_timesteps": int(truth.shape[0]),
        "mse": mse,
        "train_loss": float(train_info["train_loss"]),
    }
    save_metrics(config.metrics_path, metrics)
    return {
        "metrics_path": str(config.metrics_path),
        "rollout_path": str(config.rollout_path),
        "animation_path": str(config.animation_path),
        "checkpoint_path": str(config.checkpoint_path),
        "mse": mse,
    }
