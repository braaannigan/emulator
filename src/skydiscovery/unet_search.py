from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import types
import uuid
from pathlib import Path
from typing import Any

from skydiscover import run_discovery
from skydiscover.config import Config, EvaluatorConfig, LLMConfig, LLMModelConfig, SearchConfig

from src.models.unet_thickness.config import load_unet_thickness_config
from src.models.unet_thickness.pipeline import run_unet_thickness_experiment

OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"

ALLOWED_OVERRIDE_KEYS = {
    "learning_rate",
    "weight_decay",
    "hidden_channels",
    "num_levels",
    "kernel_size",
    "block_type",
    "stage_depth",
    "norm_type",
    "batch_size",
    "train_start_day",
    "forcing_mode",
    "fusion_mode",
    "skip_fusion_mode",
    "upsample_mode",
    "residual_connection",
}

ALLOWED_FORCING_MODES = {"wind_current", "wind_delta", "wind_current_plus_delta"}
ALLOWED_FUSION_MODES = {"input", "bottleneck", "per_scale"}
ALLOWED_BLOCK_TYPES = {"standard", "convnext"}
ALLOWED_NORM_TYPES = {"none", "groupnorm"}
ALLOWED_SKIP_FUSION_MODES = {"concat", "add", "gated"}
ALLOWED_UPSAMPLE_MODES = {"transpose", "bilinear"}


def load_env_key(env_var_name: str, env_file: str | Path = ".env") -> str:
    current = os.environ.get(env_var_name)
    if current:
        return current

    env_path = Path(env_file)
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            name, value = stripped.split("=", 1)
            if name == env_var_name:
                resolved = value.strip().strip("'").strip('"')
                os.environ[env_var_name] = resolved
                return resolved

    raise ValueError(f"Missing required environment variable: {env_var_name}")


def configure_openrouter(env_var_name: str) -> str:
    key = load_env_key(env_var_name)
    os.environ["OPENAI_API_KEY"] = key
    return key


def load_candidate_overrides(program_path: str | Path) -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location("skydiscover_candidate", str(program_path))
    if spec is None or spec.loader is None:
        raise ValueError(f"Unable to load candidate program from {program_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    overrides = getattr(module, "CONFIG_OVERRIDES", None)
    if not isinstance(overrides, dict):
        raise ValueError("Candidate program must define CONFIG_OVERRIDES as a dictionary.")
    return overrides


def validate_candidate_overrides(overrides: dict[str, Any]) -> dict[str, Any]:
    invalid = sorted(set(overrides) - ALLOWED_OVERRIDE_KEYS)
    if invalid:
        raise ValueError(f"Unsupported override keys: {invalid}")

    validated = dict(overrides)
    if "learning_rate" in validated:
        value = float(validated["learning_rate"])
        if not 1.0e-5 <= value <= 5.0e-3:
            raise ValueError("learning_rate out of allowed range")
        validated["learning_rate"] = value
    if "weight_decay" in validated:
        value = float(validated["weight_decay"])
        if not 0.0 <= value <= 1.0e-2:
            raise ValueError("weight_decay out of allowed range")
        validated["weight_decay"] = value
    if "hidden_channels" in validated:
        value = int(validated["hidden_channels"])
        if value not in {16, 24, 32, 40}:
            raise ValueError("hidden_channels out of allowed range")
        validated["hidden_channels"] = value
    if "num_levels" in validated:
        value = int(validated["num_levels"])
        if value not in {3, 4, 5}:
            raise ValueError("num_levels out of allowed range")
        validated["num_levels"] = value
    if "kernel_size" in validated:
        value = int(validated["kernel_size"])
        if value not in {3, 5, 7}:
            raise ValueError("kernel_size out of allowed range")
        validated["kernel_size"] = value
    if "block_type" in validated:
        value = str(validated["block_type"])
        if value not in ALLOWED_BLOCK_TYPES:
            raise ValueError("block_type out of allowed range")
        validated["block_type"] = value
    if "stage_depth" in validated:
        value = int(validated["stage_depth"])
        if value not in {1, 2, 3}:
            raise ValueError("stage_depth out of allowed range")
        validated["stage_depth"] = value
    if "norm_type" in validated:
        value = str(validated["norm_type"])
        if value not in ALLOWED_NORM_TYPES:
            raise ValueError("norm_type out of allowed range")
        validated["norm_type"] = value
    if "batch_size" in validated:
        value = int(validated["batch_size"])
        if value not in {1, 2, 4}:
            raise ValueError("batch_size out of allowed range")
        validated["batch_size"] = value
    if "train_start_day" in validated:
        value = float(validated["train_start_day"])
        if not 0.0 <= value <= 500.0:
            raise ValueError("train_start_day out of allowed range")
        validated["train_start_day"] = value
    if "forcing_mode" in validated:
        value = str(validated["forcing_mode"])
        if value not in ALLOWED_FORCING_MODES:
            raise ValueError("forcing_mode out of allowed range")
        validated["forcing_mode"] = value
    if "fusion_mode" in validated:
        value = str(validated["fusion_mode"])
        if value not in ALLOWED_FUSION_MODES:
            raise ValueError("fusion_mode out of allowed range")
        validated["fusion_mode"] = value
    if "skip_fusion_mode" in validated:
        value = str(validated["skip_fusion_mode"])
        if value not in ALLOWED_SKIP_FUSION_MODES:
            raise ValueError("skip_fusion_mode out of allowed range")
        validated["skip_fusion_mode"] = value
    if "upsample_mode" in validated:
        value = str(validated["upsample_mode"])
        if value not in ALLOWED_UPSAMPLE_MODES:
            raise ValueError("upsample_mode out of allowed range")
        validated["upsample_mode"] = value
    if "residual_connection" in validated:
        validated["residual_connection"] = bool(validated["residual_connection"])

    return validated


def evaluate_unet_candidate(program_path: str) -> dict[str, Any]:
    try:
        overrides = validate_candidate_overrides(load_candidate_overrides(program_path))
    except Exception as exc:
        return {
            "combined_score": -1.0e12,
            "error": str(exc),
            "eval_mse_mean": 1.0e12,
            "eval_mse_last": 1.0e12,
            "train_loss": 1.0e12,
        }

    base_config_path = os.environ["SKYDISCOVER_BASE_CONFIG"]
    source_experiment_id = os.environ.get("SKYDISCOVER_SOURCE_EXPERIMENT_ID")
    discovery_epochs = int(os.environ.get("SKYDISCOVER_DISCOVERY_EPOCHS", "5"))
    eval_window_days = float(os.environ.get("SKYDISCOVER_EVAL_WINDOW_DAYS", "250"))

    config = load_unet_thickness_config(base_config_path)
    experiment_id = f"skyd-{uuid.uuid4().hex[:10]}"
    config = config.with_overrides(
        experiment_id=experiment_id,
        epochs=discovery_epochs,
        eval_window_days=eval_window_days,
        animation_fps=0,
        **overrides,
    )
    if source_experiment_id is not None:
        config = config.with_overrides(source_experiment_id=source_experiment_id)

    outputs = run_unet_thickness_experiment(config)
    eval_mse_mean = float(outputs["eval_mse_mean"])
    eval_mse_last = float(outputs["eval_mse_last"])
    train_loss = float(outputs["train_loss"])
    combined_score = -eval_mse_mean

    return {
        "combined_score": combined_score,
        "eval_mse_mean": eval_mse_mean,
        "eval_mse_last": eval_mse_last,
        "train_loss": train_loss,
        "metrics_path": str(outputs["metrics_path"]),
        "rollout_path": str(outputs["rollout_path"]),
        "checkpoint_path": str(outputs["checkpoint_path"]),
    }


def build_discovery_config(model_name: str) -> Config:
    return Config(
        max_iterations=4,
        llm=LLMConfig(
            models=[LLMModelConfig(name=model_name)],
            evaluator_models=[LLMModelConfig(name=model_name)],
            guide_models=[LLMModelConfig(name=model_name)],
            temperature=0.6,
            top_p=0.95,
            max_tokens=12000,
            timeout=180,
            retries=3,
            retry_delay=5,
        ),
        search=SearchConfig(type="topk", num_inspirations=3),
        evaluator=EvaluatorConfig(
            timeout=3600,
            max_retries=0,
            parallel_evaluations=1,
            cascade_evaluation=False,
        ),
        diff_based_generation=False,
    )


def run_unet_discovery(
    *,
    env_var_name: str,
    model_name: str,
    base_config_path: str | Path,
    source_experiment_id: str | None,
    discovery_epochs: int,
    iterations: int,
    output_dir: str | None = None,
) -> Any:
    configure_openrouter(env_var_name)
    os.environ["SKYDISCOVER_BASE_CONFIG"] = str(base_config_path)
    os.environ["SKYDISCOVER_DISCOVERY_EPOCHS"] = str(discovery_epochs)
    if source_experiment_id is not None:
        os.environ["SKYDISCOVER_SOURCE_EXPERIMENT_ID"] = source_experiment_id

    initial_program = Path("src/skydiscovery/unet_candidate_template.py")
    config = build_discovery_config(model_name)
    config.max_iterations = iterations

    return run_discovery(
        initial_program=initial_program,
        evaluator=evaluate_unet_candidate,
        config=config,
        output_dir=output_dir,
        cleanup=False,
        codebase=".",
        model=model_name,
        api_base=OPENROUTER_API_BASE,
        system_prompt=(
            "You are optimizing a bounded architecture-and-training dictionary for a U-Net ocean emulator. "
            "Only edit CONFIG_OVERRIDES. You may change architecture choices like block type, forcing fusion, skip fusion, "
            "upsampling, normalization, and stage depth, in addition to optimization hyperparameters. "
            "Favor changes that plausibly improve eval_mse_mean on the current dataset. "
            "Keep values within the existing search space and do not add new keys."
        ),
    )


def run_openrouter_smoke_test(env_var_name: str, model_name: str) -> dict[str, Any]:
    configure_openrouter(env_var_name)

    config = build_discovery_config(model_name)
    config.max_iterations = 1
    config.search = SearchConfig(type="best_of_n")
    config.evaluator = EvaluatorConfig(
        timeout=30,
        max_retries=0,
        parallel_evaluations=1,
        cascade_evaluation=False,
    )

    result = run_discovery(
        initial_program=Path("src/skydiscovery/unet_candidate_template.py"),
        evaluator=lambda program_path: {"combined_score": 1.0},
        config=config,
        output_dir=tempfile.mkdtemp(prefix="skydiscover_smoke_"),
        cleanup=False,
        model=model_name,
        api_base=OPENROUTER_API_BASE,
        system_prompt="Return a valid Python file that only mutates CONFIG_OVERRIDES.",
    )
    return {
        "best_score": result.best_score,
        "best_solution_preview": result.best_solution[:400],
        "output_dir": result.output_dir,
    }
