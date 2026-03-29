from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ray import tune

from .config import ResidualThicknessConfig, load_residual_thickness_config
from .pipeline import run_residual_thickness_experiment


def _resolve_hpo_base_config(config: ResidualThicknessConfig) -> ResidualThicknessConfig:
    project_root = Path.cwd().resolve()
    source_data_root = config.source_data_root if config.source_data_root.is_absolute() else (project_root / config.source_data_root).resolve()
    raw_output_root = config.raw_output_root if config.raw_output_root.is_absolute() else (project_root / config.raw_output_root).resolve()
    interim_output_root = (
        config.interim_output_root if config.interim_output_root.is_absolute() else (project_root / config.interim_output_root).resolve()
    )
    return config.with_overrides(
        source_data_root=source_data_root,
        raw_output_root=raw_output_root,
        interim_output_root=interim_output_root,
        animation_fps=0,
    )


def build_residual_thickness_search_space(base_config: ResidualThicknessConfig) -> dict[str, Any]:
    return {
        "learning_rate": tune.loguniform(base_config.learning_rate / 3.0, base_config.learning_rate * 3.0),
        "weight_decay": tune.loguniform(max(base_config.weight_decay / 5.0, 1.0e-5), base_config.weight_decay * 5.0),
    }


def _run_trial(
    trial_params: dict[str, float],
    *,
    base_config: ResidualThicknessConfig,
    experiment_prefix: str,
) -> None:
    trial_context = tune.get_context()
    trial_id = trial_context.get_trial_id()
    trial_name = trial_context.get_trial_name()
    trial_experiment_id = f"{experiment_prefix}-{trial_id}"
    config = base_config.with_overrides(
        learning_rate=float(trial_params["learning_rate"]),
        weight_decay=float(trial_params["weight_decay"]),
        experiment_id=trial_experiment_id,
    )
    outputs = run_residual_thickness_experiment(config)
    tune.report(
        {
            "trial_name": trial_name,
            "trial_experiment_id": trial_experiment_id,
            "metrics_path": outputs["metrics_path"],
            "rollout_path": outputs["rollout_path"],
            "animation_path": outputs["animation_path"],
            "checkpoint_path": outputs["checkpoint_path"],
            "learning_rate": float(trial_params["learning_rate"]),
            "weight_decay": float(trial_params["weight_decay"]),
            "eval_mse_mean": float(outputs["eval_mse_mean"]),
            "eval_mse_last": float(outputs["eval_mse_last"]),
            "train_loss": float(outputs["train_loss"]),
        }
    )


def run_residual_thickness_hpo(
    config: ResidualThicknessConfig | str,
    *,
    experiment_name: str,
    num_samples: int,
) -> dict[str, Any]:
    if not isinstance(config, ResidualThicknessConfig):
        config = load_residual_thickness_config(config)
    config = _resolve_hpo_base_config(config)

    results_root = Path("data/interim/ray_results")
    results_root.mkdir(parents=True, exist_ok=True)
    search_space = build_residual_thickness_search_space(config)
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(
                _run_trial,
                base_config=config,
                experiment_prefix=experiment_name,
            ),
            resources={"cpu": 1},
        ),
        tune_config=tune.TuneConfig(
            metric="eval_mse_mean",
            mode="min",
            num_samples=num_samples,
            max_concurrent_trials=1,
        ),
        run_config=tune.RunConfig(
            name=experiment_name,
            storage_path=str(results_root.resolve()),
        ),
        param_space=search_space,
    )
    result_grid = tuner.fit()
    best_result = result_grid.get_best_result(metric="eval_mse_mean", mode="min")
    best_metrics = dict(best_result.metrics)
    summary = {
        "experiment_name": experiment_name,
        "num_samples": int(num_samples),
        "search_space": {
            "learning_rate": [
                float(config.learning_rate / 3.0),
                float(config.learning_rate * 3.0),
            ],
            "weight_decay": [
                float(max(config.weight_decay / 5.0, 1.0e-5)),
                float(config.weight_decay * 5.0),
            ],
        },
        "best_config": {
            "learning_rate": float(best_result.config["learning_rate"]),
            "weight_decay": float(best_result.config["weight_decay"]),
        },
        "best_metrics": {
            "eval_mse_mean": float(best_metrics["eval_mse_mean"]),
            "eval_mse_last": float(best_metrics["eval_mse_last"]),
            "train_loss": float(best_metrics["train_loss"]),
        },
        "best_artifacts": {
            "metrics_path": str(best_metrics["metrics_path"]),
            "rollout_path": str(best_metrics["rollout_path"]),
            "animation_path": str(best_metrics["animation_path"]),
            "checkpoint_path": str(best_metrics["checkpoint_path"]),
        },
    }
    summary_path = Path(best_metrics["metrics_path"]).with_name("tuning_summary.json")
    summary_path.write_text(f"{json.dumps(summary, indent=2)}\n")
    summary["summary_path"] = str(summary_path)
    return summary
