from __future__ import annotations

import pytest

from aoi_detection.config import Config, load_config


def test_defaults_are_usable():
    cfg = Config()
    assert cfg.input_size == (512, 512)
    assert cfg.detector.name == "threshold"
    assert cfg.report.max_defects == 0


def test_from_dict_coerces_types():
    cfg = Config.from_dict({"input_size": ["256", "128"], "detector": {"sensitivity": 0.5}})
    assert cfg.input_size == (256, 128)
    assert cfg.detector.sensitivity == 0.5
    # Unspecified sections keep their defaults.
    assert cfg.report.save_overlays is True


def test_load_config_round_trip(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("input_size: [32, 32]\ngrayscale: true\n", encoding="utf-8")

    cfg = load_config(path)
    assert cfg.input_size == (32, 32)
    assert cfg.grayscale is True


def test_load_config_rejects_non_mapping(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("- not\n- a mapping\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_config(path)
