"""A classical baseline detector: deviation from the local median."""

from __future__ import annotations

import cv2
import numpy as np


class ThresholdDetector:
    """Flags pixels that differ from their local neighbourhood.

    Serviceable for high-contrast surface defects on uniform backgrounds, and a
    useful reference point to measure a learned detector against.

    Note the locality: a defect much larger than `kernel_size` becomes its own
    local background and only its edges are flagged. Size the kernel above the
    largest defect you expect, or reach for a learned detector.
    """

    def __init__(self, sensitivity: float = 0.25, kernel_size: int = 11) -> None:
        if not 0.0 < sensitivity <= 1.0:
            raise ValueError("sensitivity must be in (0, 1]")
        if kernel_size % 2 == 0 or kernel_size < 3:
            raise ValueError("kernel_size must be an odd integer >= 3")
        self.sensitivity = sensitivity
        self.kernel_size = kernel_size

    def detect(self, image: np.ndarray) -> np.ndarray:
        """Return a uint8 mask where 255 marks a suspected defect pixel."""
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.uint8, copy=False)

        background = cv2.medianBlur(gray, self.kernel_size)
        deviation = cv2.absdiff(gray, background)

        cutoff = max(1, int(round(self.sensitivity * 255)))
        _, mask = cv2.threshold(deviation, cutoff, 255, cv2.THRESH_BINARY)
        return mask.astype(np.uint8)
