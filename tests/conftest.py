from __future__ import annotations

import numpy as np
import pytest

from aoi_detection.config import Config, DetectorConfig


@pytest.fixture
def config() -> Config:
    return Config(
        input_size=(64, 64),
        detector=DetectorConfig(sensitivity=0.2, min_defect_area=4),
    )


@pytest.fixture
def clean_image() -> np.ndarray:
    """A uniform 64x64 BGR surface with no defects."""
    return np.full((64, 64, 3), 180, dtype=np.uint8)


@pytest.fixture
def defective_image(clean_image: np.ndarray) -> np.ndarray:
    """The same surface with one dark 6x6 blemish."""
    image = clean_image.copy()
    image[20:26, 30:36] = 20
    return image
