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
        hypothesis="unit-test hypothesis",
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
        lambda config, model, normalized_train_frames, forcing_features, periodic_eval_callback=None: observed.update(
            {"train_shape": normalized_train_frames.shape}
        )
        or {
            "train_loss": 0.5,
            "device": "cpu",
            "should_use_mps": False,
            "optimization_steps": 3,
            "training_examples": 1,
            "curriculum_final_rollout_horizon": 1,
            "epoch_length_seconds": 1.25,
            "epoch_length_seconds_per_epoch": [1.25] * 20,
            "epochs_completed": 20,
            "periodic_eval_results": [],
            "stopped_early": False,
            "stop_reason": None,
        },
    )
    monkeypatch.setattr(
        "src.models.unet_thickness.pipeline.autoregressive_rollout_with_forcing",
        lambda model, eval_frames, state_history, forcing_features, standardizer, device: eval_frames[1:] + 1.0,
    )
    monkeypatch.setattr(
        "src.models.cnn_thickness.animation.create_rollout_comparison_animation",
        lambda rollout_path, output_path, fps: output_path.write_bytes(b"mp4") or output_path,
    )
    monkeypatch.setattr("torch.save", lambda payload, path: Path(path).write_bytes(b"pt"))

    outputs = run_unet_thickness_experiment(config)

    assert Path(outputs["metrics_path"]).exists()
    assert Path(outputs["rollout_path"]).exists()
    assert Path(outputs["animation_path"]).exists()
    assert Path(outputs["checkpoint_path"]).exists()
    assert outputs["mse"] == 1.0
    assert outputs["epochs_completed"] == 20
    assert outputs["stopped_early"] is False
    assert observed["train_shape"] == (2, 1, 2, 2)
    metrics = json.loads(Path(outputs["metrics_path"]).read_text(encoding="utf-8"))
    assert metrics["hypothesis"] == "unit-test hypothesis"


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
        lambda config, model, normalized_train_frames, forcing_features, periodic_eval_callback=None: {
            "train_loss": 0.25,
            "device": "cpu",
            "should_use_mps": False,
            "optimization_steps": 3,
            "training_examples": 1,
            "curriculum_final_rollout_horizon": 1,
            "epoch_length_seconds": 1.0,
            "epoch_length_seconds_per_epoch": [1.0] * 20,
            "epochs_completed": 20,
            "periodic_eval_results": [],
            "stopped_early": False,
            "stop_reason": None,
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
        "src.models.cnn_thickness.animation.create_rollout_comparison_animation",
        lambda rollout_path, output_path, fps: output_path.write_bytes(b"mp4") or output_path,
    )
    monkeypatch.setattr("torch.save", lambda payload, path: Path(path).write_bytes(b"pt"))

    outputs = run_unet_thickness_experiment(config)

    metrics = json.loads(Path(outputs["metrics_path"]).read_text(encoding="utf-8"))
    assert metrics["evaluated_field_channel_count"] == 2
    assert outputs["mse"] == 1.0


def test_run_unet_thickness_experiment_builds_periodic_eval_and_stops_against_reference(monkeypatch, tmp_path: Path):
    reference_metrics_path = tmp_path / "best_metrics.json"
    reference_metrics_path.write_text(
        json.dumps(
            {
                "eval_mse_mean": 6.0,
                "eval_mse_per_timestep": [2.0, 10.0],
                "periodic_eval_results": [
                    {
                        "epoch": 5,
                        "eval_mse_mean": 4.0,
                        "eval_mse_last": 8.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    config = load_unet_thickness_config("config/emulator/unet_thickness.yaml").with_overrides(
        raw_output_root=tmp_path / "raw",
        interim_output_root=tmp_path / "interim",
        experiment_id="unit-test-early-stop-pipeline",
        early_stopping_eval_interval_epochs=5,
        early_stopping_best_metrics_path=reference_metrics_path,
        early_stopping_margin_start=0.5,
        early_stopping_margin_end=0.1,
    )

    frames = np.arange(6 * 1 * 2 * 2, dtype=np.float32).reshape(6, 1, 2, 2)
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
                "mean": np.array([0.0], dtype=np.float32),
                "std": np.array([1.0], dtype=np.float32),
                "normalize": staticmethod(lambda values: values),
                "denormalize": staticmethod(lambda values: values),
            },
        )(),
    )
    monkeypatch.setattr("src.models.unet_thickness.pipeline.load_forcing_dataset", lambda netcdf_path, forcing_mode: None)
    monkeypatch.setattr("src.models.unet_thickness.pipeline.build_forcing_features", lambda forcing, forcing_mode: None)
    monkeypatch.setattr("src.models.unet_thickness.pipeline.fit_forcing_standardizer", lambda forcing_features: None)
    monkeypatch.setattr(
        "src.models.unet_thickness.pipeline.autoregressive_rollout_with_forcing",
        lambda model, eval_frames, state_history, forcing_features, standardizer, device: eval_frames[1:] + 20.0,
    )

    def train_stub(config, model, normalized_train_frames, forcing_features, periodic_eval_callback=None):
        assert periodic_eval_callback is not None
        eval_result = periodic_eval_callback(5)
        config.interim_experiment_dir.mkdir(parents=True, exist_ok=True)
        config.training_history_path.write_text(
            json.dumps(
                {
                    "experiment_id": config.resolved_experiment_id,
                    "status": "stopped",
                    "epochs_completed": 5,
                    "epochs_total": config.epochs,
                    "epoch_train_losses": [1.0] * 5,
                    "periodic_eval_results": [eval_result],
                    "stopped_early": True,
                    "stop_reason": eval_result["stop_reason"],
                }
            ),
            encoding="utf-8",
        )
        return {
            "train_loss": 0.5,
            "device": "cpu",
            "should_use_mps": False,
            "optimization_steps": 3,
            "training_examples": 1,
            "curriculum_final_rollout_horizon": 1,
            "epoch_length_seconds": 1.0,
            "epoch_length_seconds_per_epoch": [1.0] * 5,
            "epochs_completed": 5,
            "periodic_eval_results": [eval_result],
            "stopped_early": True,
            "stop_reason": eval_result["stop_reason"],
        }

    monkeypatch.setattr("src.models.unet_thickness.pipeline.train_unet_model", train_stub)
    monkeypatch.setattr(
        "src.models.cnn_thickness.animation.create_rollout_comparison_animation",
        lambda rollout_path, output_path, fps: output_path.write_bytes(b"mp4") or output_path,
    )
    monkeypatch.setattr("torch.save", lambda payload, path: Path(path).write_bytes(b"pt"))

    outputs = run_unet_thickness_experiment(config)

    metrics = json.loads(Path(outputs["metrics_path"]).read_text(encoding="utf-8"))
    assert outputs["stopped_early"] is True
    assert outputs["epochs_completed"] == 5
    assert metrics["stopped_early"] is True
    assert metrics["periodic_eval_results"][0]["stop_training"] is True
    assert metrics["periodic_eval_results"][0]["reference_epoch"] == 5
    assert metrics["periodic_eval_results"][0]["reference_eval_mse_mean"] == 4.0


def test_run_unet_thickness_experiment_uses_checkpoint_decay_margin_schedule(monkeypatch, tmp_path: Path):
    reference_metrics_path = tmp_path / "best_metrics.json"
    reference_metrics_path.write_text(
        json.dumps(
            {
                "eval_mse_mean": 10.0,
                "eval_mse_per_timestep": [5.0, 15.0],
                "periodic_eval_results": [
                    {"epoch": 5, "eval_mse_mean": 100.0, "eval_mse_last": 300.0},
                    {"epoch": 10, "eval_mse_mean": 80.0, "eval_mse_last": 240.0},
                    {"epoch": 15, "eval_mse_mean": 60.0, "eval_mse_last": 180.0},
                ],
            }
        ),
        encoding="utf-8",
    )
    config = load_unet_thickness_config("config/emulator/unet_thickness.yaml").with_overrides(
        raw_output_root=tmp_path / "raw",
        interim_output_root=tmp_path / "interim",
        experiment_id="unit-test-early-stop-decay",
        early_stopping_eval_interval_epochs=5,
        early_stopping_best_metrics_path=reference_metrics_path,
        early_stopping_margin_start=0.5,
        early_stopping_margin_decay=0.8,
    )

    frames = np.arange(6 * 1 * 2 * 2, dtype=np.float32).reshape(6, 1, 2, 2)
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
                "mean": np.array([0.0], dtype=np.float32),
                "std": np.array([1.0], dtype=np.float32),
                "normalize": staticmethod(lambda values: values),
                "denormalize": staticmethod(lambda values: values),
            },
        )(),
    )
    monkeypatch.setattr("src.models.unet_thickness.pipeline.load_forcing_dataset", lambda netcdf_path, forcing_mode: None)
    monkeypatch.setattr("src.models.unet_thickness.pipeline.build_forcing_features", lambda forcing, forcing_mode: None)
    monkeypatch.setattr("src.models.unet_thickness.pipeline.fit_forcing_standardizer", lambda forcing_features: None)
    monkeypatch.setattr(
        "src.models.unet_thickness.pipeline.autoregressive_rollout_with_forcing",
        lambda model, eval_frames, state_history, forcing_features, standardizer, device: eval_frames[1:] + 1.0,
    )

    def train_stub(config, model, normalized_train_frames, forcing_features, periodic_eval_callback=None):
        assert periodic_eval_callback is not None
        eval_epoch5 = periodic_eval_callback(5)
        eval_epoch10 = periodic_eval_callback(10)
        eval_epoch15 = periodic_eval_callback(15)
        return {
            "train_loss": 0.5,
            "device": "cpu",
            "should_use_mps": False,
            "optimization_steps": 3,
            "training_examples": 1,
            "curriculum_final_rollout_horizon": 1,
            "epoch_length_seconds": 1.0,
            "epoch_length_seconds_per_epoch": [1.0] * 15,
            "epochs_completed": 15,
            "periodic_eval_results": [eval_epoch5, eval_epoch10, eval_epoch15],
            "stopped_early": False,
            "stop_reason": None,
        }

    monkeypatch.setattr("src.models.unet_thickness.pipeline.train_unet_model", train_stub)
    monkeypatch.setattr("torch.save", lambda payload, path: Path(path).write_bytes(b"pt"))

    outputs = run_unet_thickness_experiment(config)

    metrics = json.loads(Path(outputs["metrics_path"]).read_text(encoding="utf-8"))
    results = metrics["periodic_eval_results"]
    assert results[0]["margin_ratio"] == 0.5
    assert results[1]["margin_ratio"] == 0.4
    assert results[2]["margin_ratio"] == 0.32000000000000006
