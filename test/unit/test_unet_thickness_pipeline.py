import json
from pathlib import Path

import numpy as np
import torch

from src.models.unet_thickness.config import load_unet_thickness_config
from src.models.unet_thickness.pipeline import run_unet_thickness_experiment


def test_run_unet_thickness_experiment_writes_expected_artifacts(monkeypatch, tmp_path: Path):
    config = load_unet_thickness_config("config/emulator/unet_thickness.yaml").with_overrides(
        raw_output_root=tmp_path / "raw",
        interim_output_root=tmp_path / "interim",
        experiment_id="unit-test",
        train_start_day=2.0,
    )

    frames = np.arange(6 * 1 * 2 * 2, dtype=np.float32).reshape(6, 1, 2, 2)
    time_days = np.arange(6, dtype=np.float32)
    y = np.array([0.0, 1.0], dtype=np.float32)
    x = np.array([0.0, 1.0], dtype=np.float32)

    observed: dict[str, tuple[int, ...] | None] = {"train_shape": None}

    monkeypatch.setattr(
        "src.models.unet_thickness.pipeline.load_state_fields",
        lambda netcdf_path, state_fields: (frames, time_days, y, x),
    )
    monkeypatch.setattr(
        "src.models.unet_thickness.pipeline.fit_channel_standardizer",
        lambda train_frames: type(
            "ChannelStandardizerStub",
            (),
            {
                "mean": np.array([0.0], dtype=np.float32),
                "std": np.array([1.0], dtype=np.float32),
                "normalize": staticmethod(lambda values: values),
                "denormalize": staticmethod(lambda values: values),
            },
        )(),
    )
    monkeypatch.setattr(
        "src.models.unet_thickness.pipeline.load_forcing_dataset",
        lambda netcdf_path, forcing_mode: None,
    )
    monkeypatch.setattr(
        "src.models.unet_thickness.pipeline.build_forcing_features",
        lambda forcing, forcing_mode: None,
    )
    monkeypatch.setattr(
        "src.models.unet_thickness.pipeline.fit_forcing_standardizer",
        lambda forcing_features: None,
    )
    monkeypatch.setattr(
        "src.models.unet_thickness.pipeline.train_unet_model",
        lambda config, model, normalized_train_frames, forcing_features: observed.update(
            {"train_shape": normalized_train_frames.shape}
        )
        or {
            "train_loss": 0.5,
            "device": "cpu",
            "optimization_steps": 3,
            "training_examples": 1,
            "curriculum_final_rollout_horizon": 1,
        },
    )
    monkeypatch.setattr(
        "src.models.unet_thickness.pipeline.autoregressive_rollout_with_forcing",
        lambda model, eval_frames, state_history, forcing_features, standardizer, device: eval_frames[1:] + 1.0,
    )
    monkeypatch.setattr(
        "src.models.unet_thickness.pipeline.create_rollout_comparison_animation",
        lambda rollout_path, output_path, fps: output_path.write_bytes(b"mp4") or output_path,
    )
    monkeypatch.setattr("torch.save", lambda payload, path: Path(path).write_bytes(b"pt"))

    outputs = run_unet_thickness_experiment(config)

    assert Path(outputs["metrics_path"]).exists()
    assert Path(outputs["rollout_path"]).exists()
    assert Path(outputs["animation_path"]).exists()
    assert Path(outputs["checkpoint_path"]).exists()
    assert outputs["mse"] == 1.0
    assert observed["train_shape"] == (2, 1, 2, 2)


def test_run_unet_thickness_experiment_supports_multilayer_targets(monkeypatch, tmp_path: Path):
    config = load_unet_thickness_config("config/emulator/unet_thickness.yaml").with_overrides(
        source_experiment_id="multilayer-source",
        raw_output_root=tmp_path / "raw",
        interim_output_root=tmp_path / "interim",
        experiment_id="unit-test-multilayer",
        state_fields=("layer_thickness", "zonal_velocity_centered", "meridional_velocity_centered"),
        field_name="layer_thickness",
        state_history=2,
    )

    frames = np.arange(6 * 6 * 2 * 2, dtype=np.float32).reshape(6, 6, 2, 2)
    time_days = np.arange(6, dtype=np.float32)
    y = np.array([0.0, 1.0], dtype=np.float32)
    x = np.array([0.0, 1.0], dtype=np.float32)

    monkeypatch.setattr(
        "src.models.unet_thickness.pipeline.load_state_fields",
        lambda netcdf_path, state_fields: (frames, time_days, y, x),
    )
    monkeypatch.setattr(
        "src.models.unet_thickness.pipeline.fit_channel_standardizer",
        lambda train_frames: type(
            "ChannelStandardizerStub",
            (),
            {
                "mean": np.zeros(train_frames.shape[1], dtype=np.float32),
                "std": np.ones(train_frames.shape[1], dtype=np.float32),
                "normalize": staticmethod(lambda values: values),
                "denormalize": staticmethod(lambda values: values),
            },
        )(),
    )
    monkeypatch.setattr("src.models.unet_thickness.pipeline.load_forcing_dataset", lambda netcdf_path, forcing_mode: None)
    monkeypatch.setattr("src.models.unet_thickness.pipeline.build_forcing_features", lambda forcing, forcing_mode: None)
    monkeypatch.setattr("src.models.unet_thickness.pipeline.fit_forcing_standardizer", lambda forcing_features: None)
    monkeypatch.setattr(
        "src.models.unet_thickness.pipeline.train_unet_model",
        lambda config, model, normalized_train_frames, forcing_features: {
            "train_loss": 0.25,
            "device": "cpu",
            "optimization_steps": 3,
            "training_examples": 1,
            "curriculum_final_rollout_horizon": 1,
        },
    )
    monkeypatch.setattr(
        "src.models.unet_thickness.pipeline.autoregressive_rollout_with_forcing",
        lambda model, eval_frames, state_history, forcing_features, standardizer, device: eval_frames[1:] + 1.0,
    )
    monkeypatch.setattr(
        "src.models.unet_thickness.pipeline.field_channel_indices",
        lambda netcdf_path, state_fields, field_name: (0, 1),
    )
    monkeypatch.setattr(
        "src.models.unet_thickness.pipeline.create_rollout_comparison_animation",
        lambda rollout_path, output_path, fps: output_path.write_bytes(b"mp4") or output_path,
    )
    monkeypatch.setattr("torch.save", lambda payload, path: Path(path).write_bytes(b"pt"))

    outputs = run_unet_thickness_experiment(config)

    metrics = json.loads(Path(outputs["metrics_path"]).read_text(encoding="utf-8"))
    assert metrics["evaluated_field_channel_count"] == 2
    assert outputs["mse"] == 1.0
