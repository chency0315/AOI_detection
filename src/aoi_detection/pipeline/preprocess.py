"""Normalising images before detection."""

from __future__ import annotations

import cv2
import numpy as np

from aoi_detection.config import Config


def preprocess(image: np.ndarray, cfg: Config) -> np.ndarray:
    """Resize to the configured input size and optionally drop colour."""
    width, height = cfg.input_size
    if image.shape[1] != width or image.shape[0] != height:
        interpolation = cv2.INTER_AREA if image.shape[0] > height else cv2.INTER_LINEAR
        image = cv2.resize(image, (width, height), interpolation=interpolation)

    if cfg.grayscale and image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return image
