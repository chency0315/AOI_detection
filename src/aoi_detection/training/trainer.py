"""Train the MobileNetV2 classifier and record everything into a run directory."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import tensorflow as tf
from tensorflow.keras import callbacks

from aoi_detection.config import TrainConfig
from aoi_detection.data.dataset import (
    class_counts,
    compute_class_weights,
    load_labels,
    split_train_valid,
)
from aoi_detection.data.generators import (
    make_train_generator,
    make_valid_generator,
    steps_per_epoch,
)
from aoi_detection.models.mobilenet_v2 import build_mobilenet_v2
from aoi_detection.utils.logging import get_logger
from aoi_detection.utils.visualize import plot_history

log = get_logger(__name__)

MODEL_FILENAME = "best_model.keras"
HISTORY_FILENAME = "history.json"
CONFIG_FILENAME = "train_config.json"


@dataclass(frozen=True)
class TrainResult:
    run_dir: Path
    model_path: Path
    history: dict[str, list[float]]

    @property
    def best_val_accuracy(self) -> float:
        return max(self.history["val_accuracy"])

    @property
    def best_epoch(self) -> int:
        """1-based epoch with the highest validation accuracy."""
        return self.history["val_accuracy"].index(self.best_val_accuracy) + 1


def make_run_dir(output_dir: str | Path, name: str | None = None) -> Path:
    run_dir = Path(output_dir) / (name or datetime.now().strftime("%Y%m%d-%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _config_to_json(cfg: TrainConfig) -> dict[str, Any]:
    raw = dataclasses.asdict(cfg)
    raw["data_dir"] = str(cfg.data_dir)
    raw["output_dir"] = str(cfg.output_dir)
    return raw


def train(cfg: TrainConfig, run_name: str | None = None) -> TrainResult:
    """Full training loop: split -> generators -> model -> fit -> plots.

    Writes to `<output_dir>/<run_name>/`:
      best_model.keras   checkpoint with the highest val_accuracy
      history.json       per-epoch metrics
      training_log.csv   same, as CSV
      accuracy.png / loss.png
      train_config.json  the exact config used
    """
    tf.keras.utils.set_random_seed(cfg.seed)
    run_dir = make_run_dir(cfg.output_dir, run_name)
    (run_dir / CONFIG_FILENAME).write_text(json.dumps(_config_to_json(cfg), indent=2))

    if not cfg.train_csv_path.exists():
        raise FileNotFoundError(
            f"{cfg.train_csv_path} not found - place the labelled train.csv in {cfg.data_dir}"
        )
    labels = load_labels(cfg.train_csv_path)
    train_df, valid_df = split_train_valid(labels, cfg.valid_split, cfg.seed)
    log.info("train/valid split: %d / %d images", len(train_df), len(valid_df))
    log.info("train class counts: %s", class_counts(train_df).to_dict())

    class_weights = compute_class_weights(train_df) if cfg.use_class_weights else None
    if class_weights:
        log.info("class weights: %s", {k: round(v, 3) for k, v in class_weights.items()})

    train_gen = make_train_generator(
        train_df, cfg.train_images_dir, cfg.image_size, cfg.batch_size, cfg.augmentation, cfg.seed
    )
    valid_gen = make_valid_generator(valid_df, cfg.train_images_dir, cfg.image_size, cfg.batch_size)
    # Keras sorts string labels lexicographically, which for "0".."5" matches the ints.
    assert list(train_gen.class_indices.values()) == sorted(train_gen.class_indices.values())

    model = build_mobilenet_v2(
        input_shape=(*cfg.image_size, 3),
        dropout=cfg.dropout,
        learning_rate=cfg.learning_rate,
    )

    model_path = run_dir / MODEL_FILENAME
    callback_list: list[callbacks.Callback] = [
        callbacks.ModelCheckpoint(
            str(model_path), monitor="val_accuracy", save_best_only=True, verbose=1
        ),
        callbacks.CSVLogger(str(run_dir / "training_log.csv")),
    ]
    if cfg.early_stopping_patience > 0:
        callback_list.append(
            callbacks.EarlyStopping(
                monitor="val_loss", patience=cfg.early_stopping_patience, verbose=1
            )
        )

    log.info("training for %d epochs, run dir %s", cfg.epochs, run_dir)
    fit = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch(train_gen, cfg.batch_size),
        epochs=cfg.epochs,
        validation_data=valid_gen,
        validation_steps=steps_per_epoch(valid_gen, cfg.batch_size),
        class_weight=class_weights,
        callbacks=callback_list,
    )

    history = {k: [float(v) for v in vals] for k, vals in fit.history.items()}
    (run_dir / HISTORY_FILENAME).write_text(json.dumps(history, indent=2))
    plot_history(history, run_dir)

    result = TrainResult(run_dir=run_dir, model_path=model_path, history=history)
    log.info(
        "best val_accuracy %.4f at epoch %d -> %s",
        result.best_val_accuracy,
        result.best_epoch,
        model_path,
    )
    return result
