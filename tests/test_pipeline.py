from __future__ import annotations

import numpy as np
import pytest

from aoi_detection.models import ThresholdDetector, build_detector
from aoi_detection.pipeline.inspect import Inspector
from aoi_detection.pipeline.preprocess import preprocess


def test_preprocess_resizes_to_configured_size(config, clean_image):
    prepared = preprocess(clean_image, config)
    assert prepared.shape[:2] == (64, 64)


def test_preprocess_grayscale_drops_channels(config, clean_image):
    cfg = type(config)(input_size=(32, 32), grayscale=True, detector=config.detector)
    prepared = preprocess(clean_image, cfg)
    assert prepared.shape == (32, 32)


def test_clean_surface_has_no_defects(config, clean_image):
    result = Inspector(config).inspect(clean_image)
    assert result.defects == ()
    assert Inspector(config).passed(result)


def test_blemish_is_detected(config, defective_image):
    inspector = Inspector(config)
    result = inspector.inspect(defective_image)

    assert result.defect_count == 1
    x, y, w, h = result.defects[0].bbox
    # The blemish sits at rows 20:26, cols 30:36 of the already-64x64 image.
    assert (x, y) == (30, 20)
    assert (w, h) == (6, 6)
    assert not inspector.passed(result)


def test_small_specks_are_filtered_by_min_area(clean_image, config):
    image = clean_image.copy()
    image[10, 10] = 0  # a single-pixel speck, below min_defect_area=4

    result = Inspector(config).inspect(image)
    assert result.defects == ()


def test_detector_rejects_bad_sensitivity():
    with pytest.raises(ValueError):
        ThresholdDetector(sensitivity=0.0)


def test_registry_rejects_unknown_detector(config):
    bad = type(config.detector)(name="magic")
    with pytest.raises(ValueError, match="unknown detector"):
        build_detector(bad)


def test_mask_is_binary_uint8(config, defective_image):
    result = Inspector(config).inspect(defective_image)
    assert result.mask.dtype == np.uint8
    assert set(np.unique(result.mask)) <= {0, 255}
