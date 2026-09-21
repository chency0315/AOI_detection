"""Dataset discovery and loading."""

from aoi_detection.data.dataset import (
    CLASS_NAMES,
    NUM_CLASSES,
    class_counts,
    compute_class_weights,
    has_labels,
    load_labels,
    split_train_valid,
)
from aoi_detection.data.loader import IMAGE_SUFFIXES, iter_images

# Keras generators live in aoi_detection.data.generators and are imported
# explicitly so the inspection pipeline never pulls in TensorFlow.

__all__ = [
    "CLASS_NAMES",
    "IMAGE_SUFFIXES",
    "NUM_CLASSES",
    "class_counts",
    "compute_class_weights",
    "has_labels",
    "iter_images",
    "load_labels",
    "split_train_valid",
]
