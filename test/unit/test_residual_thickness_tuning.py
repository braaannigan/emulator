from pathlib import Path

from src.models.residual_thickness.config import load_residual_thickness_config
from src.models.residual_thickness.tuning import (
    _resolve_hpo_base_config,
    build_residual_thickness_search_space,
    run_residual_thickness_hpo,
)


def test_build_residual_thickness_search_space_uses_incumbent_scale():
    config = load_residual_thickness_config("config/emulator/residual_thickness_depthwise_kernel5.yaml")

    search_space = build_residual_thickness_search_space(config)

    assert set(search_space) == {"learning_rate", "weight_decay"}


def test_resolve_hpo_base_config_uses_absolute_repo_paths():
    config = load_residual_thickness_config("config/emulator/residual_thickness_depthwise_kernel5.yaml")

    resolved = _resolve_hpo_base_config(config)

    assert Path(resolved.source_output_filename).is_absolute()
    assert resolved.raw_output_root.is_absolute()
    assert resolved.interim_output_root.is_absolute()


def test_run_residual_thickness_hpo_writes_summary(monkeypatch, tmp_path: Path):
    config = load_residual_thickness_config("config/emulator/residual_thickness_depthwise_kernel5.yaml")

    class DummyResult:
        config = {"learning_rate": 8.0e-4, "weight_decay": 7.0e-4}
        metrics = {
            "eval_mse_mean": 2.5,
            "eval_mse_last": 7.5,
            "train_loss": 0.2,
            "metrics_path": str(tmp_path / "raw" / "trial" / "metrics.json"),
            "rollout_path": str(tmp_path / "raw" / "trial" / "rollout.nc"),
            "animation_path": str(tmp_path / "raw" / "trial" / "comparison.mp4"),
            "checkpoint_path": str(tmp_path / "interim" / "trial" / "model.pt"),
        }

    class DummyResultGrid:
        def get_best_result(self, metric: str, mode: str) -> DummyResult:
            assert metric == "eval_mse_mean"
            assert mode == "min"
            return DummyResult()

    class DummyTuner:
        def fit(self) -> DummyResultGrid:
            metrics_path = Path(DummyResult.metrics["metrics_path"])
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text("{}\n")
            return DummyResultGrid()

    monkeypatch.setattr("src.models.residual_thickness.tuning.tune.Tuner", lambda *args, **kwargs: DummyTuner())

    summary = run_residual_thickness_hpo(
        config,
        experiment_name="20260322T010000-hpo-residual-depthwise-kernel5",
        num_samples=4,
    )

    assert summary["best_metrics"]["eval_mse_mean"] == 2.5
    assert Path(summary["summary_path"]).exists()
