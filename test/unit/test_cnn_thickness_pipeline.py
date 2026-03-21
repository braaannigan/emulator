from pathlib import Path

import numpy as np
import torch

from src.models.cnn_thickness.config import load_cnn_thickness_config
from src.models.cnn_thickness.pipeline import run_cnn_thickness_experiment


def test_run_cnn_thickness_experiment_writes_expected_artifacts(monkeypatch, tmp_path: Path):
    config = load_cnn_thickness_config("config/emulator/cnn_thickness.yaml").with_overrides(
        raw_output_root=tmp_path / "raw",
        interim_output_root=tmp_path / "interim",
        experiment_id="unit-test",
    )

    frames = np.arange(6 * 2 * 2, dtype=np.float32).reshape(6, 2, 2)
    time_days = np.arange(6, dtype=np.float32)
    y = np.array([0.0, 1.0], dtype=np.float32)
    x = np.array([0.0, 1.0], dtype=np.float32)

    monkeypatch.setattr(
        "src.models.cnn_thickness.pipeline.load_field_dataset",
        lambda netcdf_path, field_name: (frames, time_days, y, x),
    )
    monkeypatch.setattr(
        "src.models.cnn_thickness.pipeline.train_model",
        lambda config, model, normalized_train_frames: {
            "train_loss": 0.5,
            "device": "cpu",
            "optimization_steps": 3,
            "training_examples": 3,
        },
    )
    monkeypatch.setattr(
        "src.models.cnn_thickness.pipeline.autoregressive_rollout",
        lambda model, eval_frames, standardizer, device: eval_frames[1:] + 1.0,
    )
    monkeypatch.setattr(
        "src.models.cnn_thickness.pipeline.create_rollout_comparison_animation",
        lambda rollout_path, output_path, fps: output_path.write_bytes(b"mp4") or output_path,
    )
    monkeypatch.setattr("torch.save", lambda payload, path: Path(path).write_bytes(b"pt"))

    outputs = run_cnn_thickness_experiment(config)

    assert Path(outputs["metrics_path"]).exists()
    assert Path(outputs["rollout_path"]).exists()
    assert Path(outputs["animation_path"]).exists()
    assert Path(outputs["checkpoint_path"]).exists()
    assert outputs["mse"] == 1.0
