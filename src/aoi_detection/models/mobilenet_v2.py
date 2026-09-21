"""MobileNetV2 transfer-learning classifier for the six AOI classes."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Model, layers

from aoi_detection.data.dataset import NUM_CLASSES


def build_mobilenet_v2(
    num_classes: int = NUM_CLASSES,
    input_shape: tuple[int, int, int] = (224, 224, 3),
    dropout: float = 0.5,
    learning_rate: float = 1e-4,
    weights: str | None = "imagenet",
) -> Model:
    """ImageNet MobileNetV2 backbone + GAP + Dropout + softmax head, compiled.

    The whole network is trainable (the notebook does not freeze the backbone).
    Pass `weights=None` for a randomly initialised model (e.g. in tests).
    """
    backbone = tf.keras.applications.MobileNetV2(
        weights=weights, input_shape=input_shape, include_top=False
    )
    x = layers.GlobalAveragePooling2D()(backbone.output)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=backbone.inputs, outputs=outputs, name="aoi_mobilenet_v2")
    model.compile(
        loss="categorical_crossentropy",
        metrics=["accuracy"],
        optimizer=tf.keras.optimizers.Adam(learning_rate),
    )
    return model


def load_model(path: str) -> Model:
    """Load a checkpoint saved by the trainer."""
    return tf.keras.models.load_model(path)
