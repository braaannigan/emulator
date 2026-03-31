from __future__ import annotations

from pathlib import Path

import pytest

from src.skydiscovery.unet_search import (
    ALLOWED_OVERRIDE_KEYS,
    configure_openrouter,
    evaluate_unet_candidate,
    load_candidate_overrides,
    validate_candidate_overrides,
)


def test_load_candidate_overrides_reads_dict(tmp_path: Path):
    candidate_path = tmp_path / "candidate.py"
    candidate_path.write_text("CONFIG_OVERRIDES = {'hidden_channels': 24}\n", encoding="utf-8")

    overrides = load_candidate_overrides(candidate_path)

    assert overrides == {"hidden_channels": 24}


def test_validate_candidate_overrides_rejects_unknown_keys():
    with pytest.raises(ValueError):
        validate_candidate_overrides({"bad_key": 1})


def test_validate_candidate_overrides_normalizes_values():
    overrides = validate_candidate_overrides(
        {
            "learning_rate": 1e-3,
            "weight_decay": 5e-4,
            "hidden_channels": 24,
            "num_levels": 4,
            "kernel_size": 3,
            "block_type": "convnext",
            "stage_depth": 2,
            "norm_type": "groupnorm",
            "skip_fusion_mode": "gated",
            "upsample_mode": "bilinear",
        }
    )

    assert set(overrides) <= ALLOWED_OVERRIDE_KEYS
    assert overrides["stage_depth"] == 2


def test_configure_openrouter_maps_named_key(monkeypatch, tmp_path: Path):
    env_path = tmp_path / ".env"
    env_path.write_text("OPENROUTER_GEMMA3_27B_IT_API_KEY=test-key\n", encoding="utf-8")
    monkeypatch.delenv("OPENROUTER_GEMMA3_27B_IT_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)

    key = configure_openrouter("OPENROUTER_GEMMA3_27B_IT_API_KEY")

    assert key == "test-key"
    assert key == "test-key"


def test_evaluate_unet_candidate_runs_pipeline(monkeypatch, tmp_path: Path):
    candidate_path = tmp_path / "candidate.py"
    candidate_path.write_text(
        "CONFIG_OVERRIDES = {'hidden_channels': 24, 'block_type': 'convnext'}\n",
        encoding="utf-8",
    )

    config_path = tmp_path / "base.yaml"
    config_path.write_text("unused\n", encoding="utf-8")

    class ConfigStub:
        def __init__(self):
            self.overrides = {}

        def with_overrides(self, **kwargs):
            clone = ConfigStub()
            clone.overrides = self.overrides | kwargs
            return clone

    monkeypatch.setenv("SKYDISCOVER_BASE_CONFIG", str(config_path))
    monkeypatch.setenv("SKYDISCOVER_DISCOVERY_EPOCHS", "5")
    monkeypatch.setenv("SKYDISCOVER_EVAL_WINDOW_DAYS", "250")
    monkeypatch.setattr("src.skydiscovery.unet_search.load_unet_thickness_config", lambda path: ConfigStub())
    monkeypatch.setattr(
        "src.skydiscovery.unet_search.run_unet_thickness_experiment",
        lambda config: {
            "eval_mse_mean": 12.5,
            "eval_mse_last": 25.0,
            "train_loss": 0.1,
            "metrics_path": "metrics.json",
            "rollout_path": "rollout.nc",
            "checkpoint_path": "model.pt",
        },
    )

    metrics = evaluate_unet_candidate(str(candidate_path))

    assert metrics["combined_score"] == -12.5
    assert metrics["eval_mse_last"] == 25.0


def test_evaluate_unet_candidate_penalizes_invalid_program(tmp_path: Path, monkeypatch):
    candidate_path = tmp_path / "candidate.py"
    candidate_path.write_text("CONFIG_OVERRIDES = {'hidden_channels': 99}\n", encoding="utf-8")
    monkeypatch.setenv("SKYDISCOVER_BASE_CONFIG", str(tmp_path / "base.yaml"))

    metrics = evaluate_unet_candidate(str(candidate_path))

    assert metrics["combined_score"] < -1.0e11
    assert "out of allowed range" in metrics["error"]
