from pathlib import Path

from src.models.cnn_thickness.config import load_cnn_thickness_config


def test_cnn_thickness_config_builds_expected_output_paths():
    config = load_cnn_thickness_config("config/emulator/cnn_thickness.yaml").resolve_experiment("demo-run")

    assert config.source_netcdf_path == Path("data/raw/double_gyre/generator/20260319T200102/double_gyre.nc")
    assert config.rollout_path == Path("data/raw/double_gyre/emulator/cnn_thickness/demo-run/rollout.nc")
    assert config.animation_path == Path("data/raw/double_gyre/emulator/cnn_thickness/demo-run/comparison.mp4")
