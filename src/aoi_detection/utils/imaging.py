"""Image I/O and overlay rendering."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

_OVERLAY_COLOR = (0, 0, 255)  # BGR red


def read_image(path: str | Path) -> np.ndarray:
    """Load an image as BGR. Raises if the file cannot be decoded."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"could not read image: {path}")
    return np.asarray(image)


def draw_defects(image: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    """Return a copy of `image` with each (x, y, w, h) box outlined."""
    canvas: np.ndarray = image.copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    for x, y, w, h in boxes:
        cv2.rectangle(canvas, (x, y), (x + w, y + h), _OVERLAY_COLOR, 2)

    return canvas
