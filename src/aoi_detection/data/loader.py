"""Locating inspection images on disk."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})


def iter_images(root: str | Path, recursive: bool = True) -> Iterator[Path]:
    """Yield image paths under `root` in stable sorted order.

    A path to a single image yields just that image.
    """
    root = Path(root)
    if root.is_file():
        if root.suffix.lower() in IMAGE_SUFFIXES:
            yield root
        return

    pattern = "**/*" if recursive else "*"
    for path in sorted(root.glob(pattern)):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path
