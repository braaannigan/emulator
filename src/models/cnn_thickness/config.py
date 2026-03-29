from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from src.generator.config import timestamp_experiment_id


@dataclass(frozen=True)
class CnnThicknessConfig:
    source_data_root: Path
    source_experiment_id: str
    source_output_filename: str
    field_name: str
    train_fraction: float
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    hidden_channels: int
    num_layers: int
    kernel_size: int
    random_seed: int
    animation_fps: int
    raw_output_root: Path
    interim_output_root: Path
    experiment_id: str | None = None

    def with_overrides(self, **overrides: Any) -> "CnnThicknessConfig":
        normalized: dict[str, Any] = {}
        for key, value in overrides.items():
            if key in {"source_data_root", "raw_output_root", "interim_output_root"} and value is not None:
                normalized[key] = Path(value)
            else:
                normalized[key] = value
        return replace(self, **normalized)

    def resolve_experiment(self, experiment_id: str | None = None) -> "CnnThicknessConfig":
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
    def metrics_path(self) -> Path:
        return self.raw_experiment_dir / "metrics.json"

    @property
    def rollout_path(self) -> Path:
        return self.raw_experiment_dir / "rollout.nc"

    @property
    def animation_path(self) -> Path:
        return self.raw_experiment_dir / "comparison.mp4"


def load_cnn_thickness_config(path: str | Path) -> CnnThicknessConfig:
    payload = yaml.safe_load(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("cnn_thickness.yaml must contain a top-level mapping.")

    return CnnThicknessConfig(
        source_data_root=Path(payload.get("source_data_root", "data/raw/double_gyre/generator")),
        source_experiment_id=str(payload["source_experiment_id"]),
        source_output_filename=str(payload.get("source_output_filename", "double_gyre.nc")),
        field_name=str(payload.get("field_name", "layer_thickness")),
        train_fraction=float(payload["train_fraction"]),
        batch_size=int(payload["batch_size"]),
        epochs=int(payload["epochs"]),
        learning_rate=float(payload["learning_rate"]),
        weight_decay=float(payload.get("weight_decay", 0.0)),
        hidden_channels=int(payload["hidden_channels"]),
        num_layers=int(payload["num_layers"]),
        kernel_size=int(payload["kernel_size"]),
        random_seed=int(payload["random_seed"]),
        animation_fps=int(payload.get("animation_fps", 4)),
        raw_output_root=Path(payload["raw_output_root"]),
        interim_output_root=Path(payload["interim_output_root"]),
        experiment_id=payload.get("experiment_id"),
    )
