"""Keras image generators with the notebook's augmentation recipe."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, Iterator

from aoi_detection.config import AugmentationConfig
from aoi_detection.data.dataset import ID_COL, LABEL_COL

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input


def make_train_generator(
    train: pd.DataFrame,
    image_dir: str | Path,
    image_size: tuple[int, int],
    batch_size: int,
    aug: AugmentationConfig,
    seed: int | None = None,
) -> Iterator:
    """Augmented, shuffled, one-hot generator over the training split."""
    datagen = ImageDataGenerator(
        rotation_range=aug.rotation_range,
        horizontal_flip=aug.horizontal_flip,
        vertical_flip=aug.vertical_flip,
        zoom_range=aug.zoom_range,
        brightness_range=list(aug.brightness_range),
        shear_range=aug.shear_range,
        width_shift_range=aug.width_shift_range,
        height_shift_range=aug.height_shift_range,
        preprocessing_function=preprocess_input,
    )
    return datagen.flow_from_dataframe(
        dataframe=train,
        directory=str(image_dir),
        x_col=ID_COL,
        y_col=LABEL_COL,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        seed=seed,
    )


def make_valid_generator(
    valid: pd.DataFrame,
    image_dir: str | Path,
    image_size: tuple[int, int],
    batch_size: int,
) -> Iterator:
    """Un-augmented one-hot generator over the validation split."""
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    return datagen.flow_from_dataframe(
        dataframe=valid,
        directory=str(image_dir),
        x_col=ID_COL,
        y_col=LABEL_COL,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )


def make_predict_generator(
    df: pd.DataFrame,
    image_dir: str | Path,
    image_size: tuple[int, int],
    batch_size: int,
) -> Iterator:
    """Ordered, label-free generator for inference (row order == prediction order)."""
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    return datagen.flow_from_dataframe(
        dataframe=df,
        directory=str(image_dir),
        x_col=ID_COL,
        y_col=None,
        target_size=image_size,
        batch_size=batch_size,
        class_mode=None,
        shuffle=False,
    )


def steps_per_epoch(generator: Iterator, batch_size: int) -> int:
    """Batches needed to see every sample once."""
    n = generator.n
    return n // batch_size + (1 if n % batch_size else 0)
