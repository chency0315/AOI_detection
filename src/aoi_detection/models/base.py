"""The detector interface every model implementation satisfies."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Detector(Protocol):
    """Maps a preprocessed image to a binary defect mask.

    The mask is the same height and width as the input, with non-zero pixels
    marking suspected defects. Downstream stages turn the mask into regions.
    """

    def detect(self, image: np.ndarray) -> np.ndarray: ...
