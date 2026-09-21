from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from aoi_detection.cli import main


@pytest.fixture
def cfg_path(tmp_path):
    """A config matching the 64x64 test fixtures, so no rescaling occurs."""
    path = tmp_path / "cfg.yaml"
    path.write_text(
        "input_size: [64, 64]\ndetector:\n  sensitivity: 0.2\n  min_defect_area: 4\n",
        encoding="utf-8",
    )
    return path


def _write(path, image):
    cv2.imwrite(str(path), image)


def test_cli_passes_on_clean_images(tmp_path, cfg_path, clean_image):
    _write(tmp_path / "clean.png", clean_image)
    assert main(["--input", str(tmp_path / "clean.png"), "--config", str(cfg_path)]) == 0


def test_cli_fails_and_writes_report(tmp_path, cfg_path, defective_image):
    images = tmp_path / "images"
    images.mkdir()
    out = tmp_path / "out"
    _write(images / "bad.png", defective_image)

    assert main(["--input", str(images), "--output", str(out), "--config", str(cfg_path)]) == 1

    report = json.loads((out / "report.json").read_text())
    assert len(report) == 1
    assert report[0]["passed"] is False
    assert report[0]["defects"]
    assert (out / "bad_overlay.png").exists()


def test_cli_reports_when_no_images_found(tmp_path):
    assert main(["--input", str(tmp_path)]) == 1


def test_cli_honours_max_defects_threshold(tmp_path, defective_image):
    _write(tmp_path / "bad.png", defective_image)
    cfg = tmp_path / "lenient.yaml"
    cfg.write_text(
        "input_size: [64, 64]\n"
        "detector:\n  sensitivity: 0.2\n  min_defect_area: 4\n"
        "report:\n  max_defects: 10\n",
        encoding="utf-8",
    )

    # The blemish is still found, but the lenient threshold accepts it.
    assert main(["--input", str(tmp_path / "bad.png"), "--config", str(cfg)]) == 0


def test_overlay_marks_the_defect(tmp_path, cfg_path, defective_image):
    images = tmp_path / "images"
    images.mkdir()
    out = tmp_path / "out"
    _write(images / "bad.png", defective_image)

    main(["--input", str(images), "--output", str(out), "--config", str(cfg_path)])

    overlay = cv2.imread(str(out / "bad_overlay.png"))
    original = cv2.imread(str(images / "bad.png"))
    assert not np.array_equal(overlay, original)
