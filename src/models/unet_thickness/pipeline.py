from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from src.models.cnn_thickness.data import split_sequence
from src.models.cnn_thickness.evaluate import mean_squared_error, mse_per_timestep, save_metrics, save_rollout_dataset
from src.models.cnn_thickness.train import select_device, set_random_seed
from src.models.residual_thickness.training import (
    autoregressive_rollout_with_forcing,
    build_forcing_features,
    build_training_inputs,
    field_channel_indices,
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


def _state_input_channels(prognostic_channels: int, config: UnetThicknessConfig) -> int:
    if config.state_input_mode == "history":
        return prognostic_channels * config.state_history
    if config.state_input_mode == "current_plus_residual":
        return prognostic_channels * 2
    if config.state_input_mode == "residual_only":
        return prognostic_channels
    raise ValueError(f"Unsupported state_input_mode: {config.state_input_mode}")


def _first_visualization_channel(
    netcdf_path: str,
    state_fields: tuple[str, ...],
    requested_field: str,
) -> int | None:
    if requested_field not in state_fields:
        return None
    return field_channel_indices(netcdf_path, state_fields, requested_field)[0]


def _evaluate_rollout(
    config: UnetThicknessConfig,
    split,
    rollout: np.ndarray,
) -> dict[str, object]:
    field_indices = field_channel_indices(str(config.source_netcdf_path), config.state_fields, config.field_name)
    truth = split.eval_frames[1:, field_indices]
    rollout_field = rollout[:, field_indices]
    eval_time_days = split.eval_time_days[1:]
    eval_mask = None
    if config.eval_window_days is not None:
        window_limit = float(eval_time_days[0] + config.eval_window_days)
        eval_mask = eval_time_days <= window_limit
        truth = truth[eval_mask]
        rollout_field = rollout_field[eval_mask]
        eval_time_days = eval_time_days[eval_mask]
    per_timestep_mse = mse_per_timestep(truth, rollout_field)
    return {
        "field_indices": field_indices,
        "truth": truth,
        "rollout_field": rollout_field,
        "eval_time_days": eval_time_days,
        "eval_mask": eval_mask,
        "mse": mean_squared_error(truth, rollout_field),
        "per_timestep_mse": per_timestep_mse,
    }


def _load_early_stopping_reference(config: UnetThicknessConfig) -> dict[str, object] | None:
    metrics_path = config.early_stopping_best_metrics_path
    if metrics_path is None:
        return None
    payload = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    per_timestep = payload.get("eval_mse_per_timestep")
    if "eval_mse_mean" not in payload:
        raise ValueError(f"Reference metrics file {metrics_path} is missing eval_mse_mean.")
    periodic_eval_results = payload.get("periodic_eval_results", [])
    epoch_curve: dict[int, dict[str, float | None]] = {}
    if isinstance(periodic_eval_results, list):
        for entry in periodic_eval_results:
            if not isinstance(entry, dict) or "epoch" not in entry or "eval_mse_mean" not in entry:
                continue
            epoch_curve[int(entry["epoch"])] = {
                "eval_mse_mean": float(entry["eval_mse_mean"]),
                "eval_mse_last": None if entry.get("eval_mse_last") is None else float(entry["eval_mse_last"]),
            }
    return {
        "metrics_path": str(metrics_path),
        "eval_mse_mean": float(payload["eval_mse_mean"]),
        "eval_mse_last": None
        if not isinstance(per_timestep, list) or len(per_timestep) == 0
        else float(per_timestep[-1]),
        "epoch_curve": epoch_curve,
    }


def _scheduled_early_stopping_margin(config: UnetThicknessConfig, epoch: int) -> float | None:
    if config.early_stopping_margin_start is None:
        return None
    if config.early_stopping_margin_decay is not None:
        if config.early_stopping_eval_interval_epochs <= 0:
            raise ValueError("early_stopping_eval_interval_epochs must be positive when using margin decay.")
        checkpoint_index = max((int(epoch) // int(config.early_stopping_eval_interval_epochs)) - 1, 0)
        return float(config.early_stopping_margin_start) * (float(config.early_stopping_margin_decay) ** checkpoint_index)
    if config.early_stopping_margin_end is None:
        return None
    if config.epochs <= 1:
        return float(config.early_stopping_margin_end)
    progress = float(max(epoch - 1, 0)) / float(config.epochs - 1)
    return float(config.early_stopping_margin_start) + (
        (float(config.early_stopping_margin_end) - float(config.early_stopping_margin_start)) * progress
    )


def _build_periodic_eval_callback(
    config: UnetThicknessConfig,
    model: UnetThicknessModel,
    split,
    standardizer,
    eval_forcing: np.ndarray | None,
    device: torch.device,
) -> tuple[callable | None, dict[str, object] | None]:
    interval = int(config.early_stopping_eval_interval_epochs)
    if interval <= 0:
        return None, None

    reference = _load_early_stopping_reference(config)
    if reference is not None:
        has_linear_schedule = (
            config.early_stopping_margin_start is not None and config.early_stopping_margin_end is not None
        )
        has_decay_schedule = (
            config.early_stopping_margin_start is not None and config.early_stopping_margin_decay is not None
        )
        if not (has_linear_schedule or has_decay_schedule):
            raise ValueError(
                "Early stopping margins must provide either start/end or start/decay when "
                "early_stopping_best_metrics_path is set."
            )

    def _callback(epoch: int) -> dict[str, object]:
        if config.state_input_mode == "history":
            rollout = autoregressive_rollout_with_forcing(
                model,
                split.eval_frames,
                config.state_history,
                eval_forcing,
                standardizer,
                device,
            )
        else:
            rollout = autoregressive_rollout_with_forcing(
                model,
                split.eval_frames,
                config.state_history,
                eval_forcing,
                standardizer,
                device,
                state_input_mode=config.state_input_mode,
            )
        evaluation = _evaluate_rollout(config, split, rollout)
        per_timestep_mse = evaluation["per_timestep_mse"]
        eval_mse_mean = float(per_timestep_mse.mean())
        eval_mse_last = float(per_timestep_mse[-1])
        result: dict[str, object] = {
            "epoch": epoch,
            "eval_mse_mean": eval_mse_mean,
            "eval_mse_last": eval_mse_last,
            "stop_training": False,
            "stop_reason": None,
        }
        if reference is None:
            return result

        epoch_curve = reference.get("epoch_curve", {})
        benchmark = epoch_curve.get(epoch)
        reference_eval_mse_mean = float(reference["eval_mse_mean"])
        reference_eval_mse_last = reference["eval_mse_last"]
        reference_epoch = None
        if isinstance(benchmark, dict):
            reference_eval_mse_mean = float(benchmark["eval_mse_mean"])
            reference_eval_mse_last = benchmark["eval_mse_last"]
            reference_epoch = epoch
        margin_ratio = _scheduled_early_stopping_margin(config, epoch)
        stop_threshold = reference_eval_mse_mean * (1.0 + float(margin_ratio))
        should_stop = eval_mse_mean > stop_threshold
        result.update(
            {
                "reference_metrics_path": reference["metrics_path"],
                "reference_epoch": reference_epoch,
                "reference_eval_mse_mean": reference_eval_mse_mean,
                "reference_eval_mse_last": reference_eval_mse_last,
                "margin_ratio": float(margin_ratio),
                "stop_threshold": stop_threshold,
                "stop_training": should_stop,
                "stop_reason": None
                if not should_stop
                else (
                    "Periodic eval MSE exceeded the early-stopping threshold: "
                    f"{eval_mse_mean:.4f} > {stop_threshold:.4f}."
                ),
            }
        )
        return result

    return _callback, reference


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
    eval_forcing = None if normalized_forcing_features is None else normalized_forcing_features[split.train_frames.shape[0] :]
    prognostic_channels = int(split.train_frames.shape[1])
    if config.state_input_mode == "residual_only" and config.residual_connection:
        raise ValueError("state_input_mode=residual_only requires residual_connection=false.")

    model = UnetThicknessModel(
        input_channels=_state_input_channels(prognostic_channels, config) + forcing_channel_count(config.forcing_mode),
        hidden_channels=config.hidden_channels,
        num_levels=config.num_levels,
        kernel_size=config.kernel_size,
        stage_depth=config.stage_depth,
        dilation_cycle=config.dilation_cycle,
        norm_type=config.norm_type,
        block_type=config.block_type,
        state_channels=_state_input_channels(prognostic_channels, config),
        output_steps=config.output_steps,
        forcing_channels=forcing_channel_count(config.forcing_mode),
        fusion_mode=config.fusion_mode,
        skip_fusion_mode=config.skip_fusion_mode,
        upsample_mode=config.upsample_mode,
        residual_connection=config.residual_connection,
        residual_step_scale=config.residual_step_scale,
        prognostic_channels=prognostic_channels,
    )
    train_device = select_device()
    periodic_eval_callback, early_stopping_reference = _build_periodic_eval_callback(
        config,
        model,
        split,
        standardizer,
        eval_forcing,
        train_device,
    )
    train_info = train_unet_model(
        config,
        model,
        standardizer.normalize(split.train_frames),
        None if normalized_forcing_features is None else normalized_forcing_features[: split.train_frames.shape[0]],
        periodic_eval_callback=periodic_eval_callback,
    )
    device = torch.device(train_info["device"])

    if config.state_input_mode == "history":
        rollout = autoregressive_rollout_with_forcing(
            model,
            split.eval_frames,
            config.state_history,
            eval_forcing,
            standardizer,
            device,
        )
    else:
        rollout = autoregressive_rollout_with_forcing(
            model,
            split.eval_frames,
            config.state_history,
            eval_forcing,
            standardizer,
            device,
            state_input_mode=config.state_input_mode,
        )
    evaluation = _evaluate_rollout(config, split, rollout)
    field_indices = evaluation["field_indices"]
    truth = evaluation["truth"]
    rollout_field = evaluation["rollout_field"]
    eval_time_days = evaluation["eval_time_days"]
    eval_mask = evaluation["eval_mask"]
    mse = float(evaluation["mse"])
    per_timestep_mse = evaluation["per_timestep_mse"]

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
    truth_animation = truth[:, 0] if truth.ndim == 4 else truth
    rollout_animation = rollout_field[:, 0] if rollout_field.ndim == 4 else rollout_field
    truth_zonal_velocity = None
    rollout_zonal_velocity = None
    zonal_velocity_index = _first_visualization_channel(
        str(config.source_netcdf_path),
        config.state_fields,
        "zonal_velocity_centered",
    )
    if zonal_velocity_index is not None:
        truth_zonal_velocity = split.eval_frames[1:, zonal_velocity_index]
        rollout_zonal_velocity = rollout[:, zonal_velocity_index]
        if config.eval_window_days is not None:
            truth_zonal_velocity = truth_zonal_velocity[eval_mask]
            rollout_zonal_velocity = rollout_zonal_velocity[eval_mask]

    truth_meridional_velocity = None
    rollout_meridional_velocity = None
    meridional_velocity_index = _first_visualization_channel(
        str(config.source_netcdf_path),
        config.state_fields,
        "meridional_velocity_centered",
    )
    if meridional_velocity_index is not None:
        truth_meridional_velocity = split.eval_frames[1:, meridional_velocity_index]
        rollout_meridional_velocity = rollout[:, meridional_velocity_index]
        if config.eval_window_days is not None:
            truth_meridional_velocity = truth_meridional_velocity[eval_mask]
            rollout_meridional_velocity = rollout_meridional_velocity[eval_mask]
    save_rollout_dataset(
        config.rollout_path,
        truth=truth_animation,
        rollout=rollout_animation,
        time_days=eval_time_days,
        y=y,
        x=x,
        truth_zonal_velocity=truth_zonal_velocity,
        rollout_zonal_velocity=rollout_zonal_velocity,
        truth_meridional_velocity=truth_meridional_velocity,
        rollout_meridional_velocity=rollout_meridional_velocity,
    )
    if config.animation_fps > 0:
        from src.models.cnn_thickness.animation import create_rollout_comparison_animation

        create_rollout_comparison_animation(config.rollout_path, config.animation_path, fps=config.animation_fps)

    metrics = {
        "source_experiment_id": config.source_experiment_id,
        "emulator_experiment_id": config.resolved_experiment_id,
        "hypothesis": config.hypothesis,
        "architecture": "unet_thickness",
        "hidden_channels": config.hidden_channels,
        "num_levels": config.num_levels,
        "kernel_size": config.kernel_size,
        "block_type": config.block_type,
        "stage_depth": config.stage_depth,
        "dilation_cycle": config.dilation_cycle,
        "norm_type": config.norm_type,
        "state_history": config.state_history,
        "state_input_mode": config.state_input_mode,
        "output_steps": config.output_steps,
        "state_fields": list(config.state_fields),
        "evaluated_field_channel_count": len(field_indices),
        "forcing_mode": config.forcing_mode,
        "fusion_mode": config.fusion_mode,
        "skip_fusion_mode": config.skip_fusion_mode,
        "upsample_mode": config.upsample_mode,
        "residual_connection": config.residual_connection,
        "residual_step_scale": config.residual_step_scale,
        "scheduled_sampling_max_prob": config.scheduled_sampling_max_prob,
        "high_frequency_loss_weight": config.high_frequency_loss_weight,
        "early_stopping_eval_interval_epochs": config.early_stopping_eval_interval_epochs,
        "early_stopping_best_metrics_path": None
        if config.early_stopping_best_metrics_path is None
        else str(config.early_stopping_best_metrics_path),
        "early_stopping_margin_start": config.early_stopping_margin_start,
        "early_stopping_margin_end": config.early_stopping_margin_end,
        "early_stopping_margin_decay": config.early_stopping_margin_decay,
        "early_stopping_reference_eval_mse_mean": None
        if early_stopping_reference is None
        else float(early_stopping_reference["eval_mse_mean"]),
        "eval_window_days": config.eval_window_days,
        "curriculum_rollout_steps": [int(value) for value in config.curriculum_rollout_steps],
        "curriculum_transition_epochs": [int(value) for value in config.curriculum_transition_epochs],
        "curriculum_final_rollout_horizon": int(train_info["curriculum_final_rollout_horizon"]),
        "epoch_length_seconds": float(train_info["epoch_length_seconds"]),
        "epoch_length_seconds_per_epoch": [float(value) for value in train_info["epoch_length_seconds_per_epoch"]],
        "epochs_completed": int(train_info["epochs_completed"]),
        "stopped_early": bool(train_info["stopped_early"]),
        "stop_reason": train_info["stop_reason"],
        "periodic_eval_results": train_info["periodic_eval_results"],
        "device": str(train_info["device"]),
        "should_use_mps": bool(train_info["should_use_mps"]),
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
        "epochs_completed": int(train_info["epochs_completed"]),
        "stopped_early": bool(train_info["stopped_early"]),
    }
