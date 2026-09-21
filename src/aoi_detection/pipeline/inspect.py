"""End-to-end inspection of a single image."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from aoi_detection.config import Config
from aoi_detection.models import build_detector
from aoi_detection.pipeline.preprocess import preprocess


@dataclass(frozen=True)
class Defect:
    """A connected defect region in preprocessed-image coordinates."""

    x: int
    y: int
    width: int
    height: int
    area: int

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.width, self.height


@dataclass(frozen=True)
class InspectionResult:
    source: Path | None
    defects: tuple[Defect, ...]
    mask: np.ndarray

    @property
    def defect_count(self) -> int:
        return len(self.defects)


@dataclass(frozen=True)
class Inspector:
    """Runs preprocess -> detect -> region extraction for a configuration."""

    config: Config

    def inspect(self, image: np.ndarray, source: Path | None = None) -> InspectionResult:
        prepared = preprocess(image, self.config)
        detector = build_detector(self.config.detector)
        mask = detector.detect(prepared)
        defects = self._extract_defects(mask)
        return InspectionResult(source=source, defects=defects, mask=mask)

    def passed(self, result: InspectionResult) -> bool:
        """Whether the image meets the configured acceptance threshold."""
        return len(result.defects) <= self.config.report.max_defects

    def _extract_defects(self, mask: np.ndarray) -> tuple[Defect, ...]:
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        min_area = self.config.detector.min_defect_area

        defects = []
        # Label 0 is the background component.
        for label in range(1, count):
            x, y, w, h, area = (int(v) for v in stats[label])
            if area < min_area:
                continue
            defects.append(Defect(x=x, y=y, width=w, height=h, area=area))

        return tuple(sorted(defects, key=lambda d: d.area, reverse=True))
