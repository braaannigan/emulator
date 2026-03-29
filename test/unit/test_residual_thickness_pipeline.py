from pathlib import Path

import numpy as np
import torch

from src.models.residual_thickness.config import load_residual_thickness_config
from src.models.residual_thickness.pipeline import run_residual_thickness_experiment


def test_run_residual_thickness_experiment_writes_expected_artifacts(monkeypatch, tmp_path: Path):
    config = load_residual_thickness_config("config/emulator/residual_thickness.yaml").with_overrides(
        raw_output_root=tmp_path / "raw",
        interim_output_root=tmp_path / "interim",
        experiment_id="unit-test",
    )

    frames = np.arange(6 * 1 * 2 * 2, dtype=np.float32).reshape(6, 1, 2, 2)
    time_days = np.arange(6, dtype=np.float32)
    y = np.array([0.0, 1.0], dtype=np.float32)
    x = np.array([0.0, 1.0], dtype=np.float32)

    monkeypatch.setattr(
        "src.models.residual_thickness.pipeline.load_state_fields",
        lambda netcdf_path, state_fields: (frames, time_days, y, x),
    )
    monkeypatch.setattr(
        "src.models.residual_thickness.pipeline.fit_channel_standardizer",
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
        "src.models.residual_thickness.pipeline.load_forcing_dataset",
        lambda netcdf_path, forcing_mode: None,
    )
    monkeypatch.setattr(
        "src.models.residual_thickness.pipeline.build_forcing_features",
        lambda forcing, forcing_mode: None,
    )
    monkeypatch.setattr(
        "src.models.residual_thickness.pipeline.fit_forcing_standardizer",
        lambda forcing_features: None,
    )
    monkeypatch.setattr(
        "src.models.residual_thickness.pipeline.train_residual_model",
        lambda config, model, train_inputs, train_targets: {
            "train_loss": 0.5,
            "device": "cpu",
            "optimization_steps": 3,
            "training_examples": 3,
        },
    )
    monkeypatch.setattr(
        "src.models.residual_thickness.pipeline.autoregressive_rollout_with_forcing",
        lambda model, eval_frames, state_history, forcing_features, standardizer, device: eval_frames[1:] + 1.0,
    )
    monkeypatch.setattr(
        "src.models.residual_thickness.pipeline.create_rollout_comparison_animation",
        lambda rollout_path, output_path, fps: output_path.write_bytes(b"mp4") or output_path,
    )
    monkeypatch.setattr("torch.save", lambda payload, path: Path(path).write_bytes(b"pt"))

    outputs = run_residual_thickness_experiment(config)

    assert Path(outputs["metrics_path"]).exists()
    assert Path(outputs["rollout_path"]).exists()
    assert Path(outputs["animation_path"]).exists()
    assert Path(outputs["checkpoint_path"]).exists()
    assert outputs["mse"] == 1.0


def test_run_residual_thickness_experiment_skips_animation_when_disabled(monkeypatch, tmp_path: Path):
    config = load_residual_thickness_config("config/emulator/residual_thickness.yaml").with_overrides(
        raw_output_root=tmp_path / "raw",
        interim_output_root=tmp_path / "interim",
        experiment_id="unit-test-no-animation",
        animation_fps=0,
    )

    frames = np.arange(6 * 1 * 2 * 2, dtype=np.float32).reshape(6, 1, 2, 2)
    time_days = np.arange(6, dtype=np.float32)
    y = np.array([0.0, 1.0], dtype=np.float32)
    x = np.array([0.0, 1.0], dtype=np.float32)

    monkeypatch.setattr(
        "src.models.residual_thickness.pipeline.load_state_fields",
        lambda netcdf_path, state_fields: (frames, time_days, y, x),
    )
    monkeypatch.setattr(
        "src.models.residual_thickness.pipeline.fit_channel_standardizer",
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
        "src.models.residual_thickness.pipeline.load_forcing_dataset",
        lambda netcdf_path, forcing_mode: None,
    )
    monkeypatch.setattr(
        "src.models.residual_thickness.pipeline.build_forcing_features",
        lambda forcing, forcing_mode: None,
    )
    monkeypatch.setattr(
        "src.models.residual_thickness.pipeline.fit_forcing_standardizer",
        lambda forcing_features: None,
    )
    monkeypatch.setattr(
        "src.models.residual_thickness.pipeline.train_residual_model",
        lambda config, model, train_inputs, train_targets: {
            "train_loss": 0.5,
            "device": "cpu",
            "optimization_steps": 3,
            "training_examples": 3,
        },
    )
    monkeypatch.setattr(
        "src.models.residual_thickness.pipeline.autoregressive_rollout_with_forcing",
        lambda model, eval_frames, state_history, forcing_features, standardizer, device: eval_frames[1:] + 1.0,
    )
    monkeypatch.setattr("torch.save", lambda payload, path: Path(path).write_bytes(b"pt"))

    def fail_if_called(*args, **kwargs):
        raise AssertionError("animation should be skipped when animation_fps is 0")

    monkeypatch.setattr("src.models.residual_thickness.pipeline.create_rollout_comparison_animation", fail_if_called)

    outputs = run_residual_thickness_experiment(config)

    assert Path(outputs["metrics_path"]).exists()
    assert Path(outputs["rollout_path"]).exists()
    assert not Path(outputs["animation_path"]).exists()


def test_run_residual_thickness_experiment_applies_eval_window(monkeypatch, tmp_path: Path):
    config = load_residual_thickness_config("config/emulator/residual_thickness.yaml").with_overrides(
        raw_output_root=tmp_path / "raw",
        interim_output_root=tmp_path / "interim",
        experiment_id="unit-test-window",
        eval_window_days=10.0,
    )

    frames = np.arange(6 * 1 * 2 * 2, dtype=np.float32).reshape(6, 1, 2, 2)
    time_days = np.array([0.0, 7.0, 14.0, 21.0, 28.0, 35.0], dtype=np.float32)
    y = np.array([0.0, 1.0], dtype=np.float32)
    x = np.array([0.0, 1.0], dtype=np.float32)

    monkeypatch.setattr(
        "src.models.residual_thickness.pipeline.load_state_fields",
        lambda netcdf_path, state_fields: (frames, time_days, y, x),
    )
    monkeypatch.setattr(
        "src.models.residual_thickness.pipeline.fit_channel_standardizer",
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
        "src.models.residual_thickness.pipeline.load_forcing_dataset",
        lambda netcdf_path, forcing_mode: None,
    )
    monkeypatch.setattr(
        "src.models.residual_thickness.pipeline.build_forcing_features",
        lambda forcing, forcing_mode: None,
    )
    monkeypatch.setattr(
        "src.models.residual_thickness.pipeline.fit_forcing_standardizer",
        lambda forcing_features: None,
    )
    monkeypatch.setattr(
        "src.models.residual_thickness.pipeline.train_residual_model",
        lambda config, model, train_inputs, train_targets: {
            "train_loss": 0.5,
            "device": "cpu",
            "optimization_steps": 3,
            "training_examples": 3,
        },
    )
    monkeypatch.setattr(
        "src.models.residual_thickness.pipeline.autoregressive_rollout_with_forcing",
        lambda model, eval_frames, state_history, forcing_features, standardizer, device: eval_frames[1:] + 1.0,
    )
    monkeypatch.setattr("torch.save", lambda payload, path: Path(path).write_bytes(b"pt"))
    monkeypatch.setattr(
        "src.models.residual_thickness.pipeline.create_rollout_comparison_animation",
        lambda rollout_path, output_path, fps: output_path.write_bytes(b"mp4") or output_path,
    )

    outputs = run_residual_thickness_experiment(config)

    assert outputs["mse"] == 1.0
    metrics = Path(outputs["metrics_path"]).read_text()
    assert '"eval_timesteps": 1' in metrics
