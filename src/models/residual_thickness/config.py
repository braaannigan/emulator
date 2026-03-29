from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from src.generator.config import timestamp_experiment_id


@dataclass(frozen=True)
class ResidualThicknessConfig:
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
    num_blocks: int
    kernel_size: int
    model_variant: str
    block_type: str
    normalization: str
    dilation_cycle: int
    state_history: int
    forcing_mode: str
    forcing_integration: str
    gradient_loss_weight: float
    eval_window_days: float | None
    transport_displacement_scale: float
    transport_correction_scale: float
    transport_head_mode: str
    random_seed: int
    animation_fps: int
    raw_output_root: Path
    interim_output_root: Path
    experiment_id: str | None = None

    def with_overrides(self, **overrides: Any) -> "ResidualThicknessConfig":
        normalized: dict[str, Any] = {}
        for key, value in overrides.items():
            if key in {"source_data_root", "raw_output_root", "interim_output_root"} and value is not None:
                normalized[key] = Path(value)
            else:
                normalized[key] = value
        return replace(self, **normalized)

    def resolve_experiment(self, experiment_id: str | None = None) -> "ResidualThicknessConfig":
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


def load_residual_thickness_config(path: str | Path) -> ResidualThicknessConfig:
    payload = yaml.safe_load(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("residual_thickness.yaml must contain a top-level mapping.")

    return ResidualThicknessConfig(
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
        num_blocks=int(payload["num_blocks"]),
        kernel_size=int(payload["kernel_size"]),
        model_variant=str(payload.get("model_variant", "residual")),
        block_type=str(payload.get("block_type", "standard")),
        normalization=str(payload.get("normalization", "none")),
        dilation_cycle=int(payload.get("dilation_cycle", 1)),
        state_history=int(payload.get("state_history", 1)),
        forcing_mode=str(payload.get("forcing_mode", "none")),
        forcing_integration=str(payload.get("forcing_integration", "concat")),
        gradient_loss_weight=float(payload.get("gradient_loss_weight", 0.0)),
        eval_window_days=None if payload.get("eval_window_days") is None else float(payload.get("eval_window_days")),
        transport_displacement_scale=float(payload.get("transport_displacement_scale", 1.0)),
        transport_correction_scale=float(payload.get("transport_correction_scale", 1.0)),
        transport_head_mode=str(payload.get("transport_head_mode", "shared")),
        random_seed=int(payload["random_seed"]),
        animation_fps=int(payload.get("animation_fps", 4)),
        raw_output_root=Path(payload["raw_output_root"]),
        interim_output_root=Path(payload["interim_output_root"]),
        experiment_id=payload.get("experiment_id"),
    )
