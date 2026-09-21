"""Detector lookup by config name."""

from __future__ import annotations

from aoi_detection.config import DetectorConfig
from aoi_detection.models.base import Detector
from aoi_detection.models.threshold import ThresholdDetector


def build_detector(cfg: DetectorConfig) -> Detector:
    """Instantiate the detector named by `cfg`."""
    if cfg.name == "threshold":
        return ThresholdDetector(sensitivity=cfg.sensitivity)
    raise ValueError(f"unknown detector: {cfg.name!r}")
