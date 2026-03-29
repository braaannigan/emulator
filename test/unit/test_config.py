import re
from pathlib import Path

import pytest

from src.generator.config import load_double_gyre_config, timestamp_experiment_id
from src.generator.runner import write_aronnax_configuration


def test_load_double_gyre_config_derives_expected_schedule():
    config = load_double_gyre_config("config/generator/double_gyre.yaml")

    assert config.n_time_steps == 10080
    assert config.dump_freq_seconds == 604800.0
    assert config.expected_output_count == 10


def test_load_double_gyre_config_rejects_missing_required_keys(tmp_path: Path):
    config_path = tmp_path / "broken.yaml"
    config_path.write_text("nx: 10\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required keys"):
        load_double_gyre_config(config_path)


def test_write_aronnax_configuration_persists_expected_sections(tmp_path: Path):
    config = load_double_gyre_config("config/generator/double_gyre.yaml").with_overrides(
        run_directory=tmp_path / "run"
    )
    conf_path = write_aronnax_configuration(config, tmp_path / "aronnax.conf")
    contents = conf_path.read_text(encoding="utf-8")

    assert "[numerics]" in contents
    assert "n_time_steps = 10080" in contents
    assert "dump_freq = 604800.0" in contents
    assert "f_u_file = :beta_plane_f_u:1e-05,2e-11" in contents


def test_resolve_experiment_nests_outputs_under_timestamped_directory():
    config = load_double_gyre_config("config/generator/double_gyre.yaml")
    resolved = config.resolve_experiment("20260319T120000")

    assert resolved.experiment_id == "20260319T120000"
    assert resolved.netcdf_output_path == Path("data/raw/double_gyre/generator/20260319T120000/double_gyre.nc")
    assert resolved.run_directory == Path("data/interim/double_gyre_run/20260319T120000")


def test_timestamp_experiment_id_uses_expected_format():
    experiment_id = timestamp_experiment_id()
    assert re.fullmatch(r"\d{8}T\d{6}", experiment_id)
