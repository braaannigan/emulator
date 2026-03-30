from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from src.generator.config import timestamp_experiment_id


@dataclass(frozen=True)
class UnetThicknessConfig:
    source_data_root: Path
    source_experiment_id: str
    source_output_filename: str
    field_name: str
    state_fields: tuple[str, ...]
    train_fraction: float
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    hidden_channels: int
    num_levels: int
    kernel_size: int
    block_type: str
    stage_depth: int
    dilation_cycle: int
    norm_type: str
    state_history: int
    output_steps: int
    forcing_mode: str
    fusion_mode: str
    skip_fusion_mode: str
    upsample_mode: str
    residual_connection: bool
    residual_step_scale: float
    curriculum_rollout_steps: tuple[int, ...]
    curriculum_transition_epochs: tuple[int, ...]
    scheduled_sampling_max_prob: float
    high_frequency_loss_weight: float
    train_start_day: float
    eval_window_days: float | None
    random_seed: int
    animation_fps: int
    raw_output_root: Path
    interim_output_root: Path
    experiment_id: str | None = None

    def with_overrides(self, **overrides: Any) -> "UnetThicknessConfig":
        normalized: dict[str, Any] = {}
        for key, value in overrides.items():
            if key in {"source_data_root", "raw_output_root", "interim_output_root"} and value is not None:
                normalized[key] = Path(value)
            else:
                normalized[key] = value
        return replace(self, **normalized)

    def resolve_experiment(self, experiment_id: str | None = None) -> "UnetThicknessConfig":
        resolved_id = experiment_id or self.experiment_id or timestamp_experiment_id()
        return replace(self, experiment_id=resolved_id)

    @property
    def source_netcdf_path(self) -> Path:
        source_output_path = Path(self.source_output_filename)
        if source_output_path.is_absolute():
            return source_output_path
        return self.source_data_root / self.source_experiment_id / source_output_path

    @property
    def resolved_experiment_id(self) -> str:
        if self.experiment_id is None:
            raise ValueError("experiment_id has not been resolved.")
        return self.experiment_id

    @property
    def raw_experiment_dir(self) -> Path:
        return self.raw_output_root / self.resolved_experiment_id

    @property
    def interim_experiment_dir(self) -> Path:
        return self.interim_output_root / self.resolved_experiment_id

    @property
    def checkpoint_path(self) -> Path:
        return self.interim_experiment_dir / "model.pt"

    @property
    def training_history_path(self) -> Path:
        return self.interim_experiment_dir / "training_history.json"

    @property
    def metrics_path(self) -> Path:
        return self.raw_experiment_dir / "metrics.json"

    @property
    def rollout_path(self) -> Path:
        return self.raw_experiment_dir / "rollout.nc"

    @property
    def animation_path(self) -> Path:
        return self.raw_experiment_dir / "comparison.mp4"


def load_unet_thickness_config(path: str | Path) -> UnetThicknessConfig:
    payload = yaml.safe_load(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("unet_thickness.yaml must contain a top-level mapping.")

    return UnetThicknessConfig(
        source_data_root=Path(payload.get("source_data_root", "data/raw/double_gyre/generator")),
        source_experiment_id=str(payload["source_experiment_id"]),
        source_output_filename=str(payload.get("source_output_filename", "double_gyre.nc")),
        field_name=str(payload.get("field_name", "layer_thickness")),
        state_fields=tuple(payload.get("state_fields", [payload.get("field_name", "layer_thickness")])),
        train_fraction=float(payload["train_fraction"]),
        batch_size=int(payload["batch_size"]),
        epochs=int(payload["epochs"]),
        learning_rate=float(payload["learning_rate"]),
        weight_decay=float(payload.get("weight_decay", 0.0)),
        hidden_channels=int(payload["hidden_channels"]),
        num_levels=int(payload.get("num_levels", 2)),
        kernel_size=int(payload.get("kernel_size", 3)),
        block_type=str(payload.get("block_type", "standard")),
        stage_depth=int(payload.get("stage_depth", 1)),
        dilation_cycle=int(payload.get("dilation_cycle", 1)),
        norm_type=str(payload.get("norm_type", "groupnorm")),
        state_history=int(payload.get("state_history", 1)),
        output_steps=int(payload.get("output_steps", 1)),
        forcing_mode=str(payload.get("forcing_mode", "none")),
        fusion_mode=str(payload.get("fusion_mode", "input")),
        skip_fusion_mode=str(payload.get("skip_fusion_mode", "concat")),
        upsample_mode=str(payload.get("upsample_mode", "transpose")),
        residual_connection=bool(payload.get("residual_connection", True)),
        residual_step_scale=float(payload.get("residual_step_scale", 1.0)),
        curriculum_rollout_steps=tuple(int(value) for value in payload.get("curriculum_rollout_steps", [1])),
        curriculum_transition_epochs=tuple(int(value) for value in payload.get("curriculum_transition_epochs", [0])),
        scheduled_sampling_max_prob=float(payload.get("scheduled_sampling_max_prob", 0.0)),
        high_frequency_loss_weight=float(payload.get("high_frequency_loss_weight", 0.0)),
        train_start_day=float(payload.get("train_start_day", 0.0)),
        eval_window_days=None if payload.get("eval_window_days") is None else float(payload.get("eval_window_days")),
        random_seed=int(payload["random_seed"]),
        animation_fps=int(payload.get("animation_fps", 4)),
        raw_output_root=Path(payload["raw_output_root"]),
        interim_output_root=Path(payload["interim_output_root"]),
        experiment_id=payload.get("experiment_id"),
    )
