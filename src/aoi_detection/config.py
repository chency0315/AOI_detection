"""Configuration loading for the inspection pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DetectorConfig:
    name: str = "threshold"
    sensitivity: float = 0.25
    min_defect_area: int = 24


@dataclass(frozen=True)
class ReportConfig:
    max_defects: int = 0
    save_overlays: bool = True


@dataclass(frozen=True)
class Config:
    input_size: tuple[int, int] = (512, 512)
    grayscale: bool = False
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    report: ReportConfig = field(default_factory=ReportConfig)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Config:
        size = raw.get("input_size", (512, 512))
        return cls(
            input_size=(int(size[0]), int(size[1])),
            grayscale=bool(raw.get("grayscale", False)),
            detector=DetectorConfig(**raw.get("detector", {})),
            report=ReportConfig(**raw.get("report", {})),
        )


def load_config(path: str | Path) -> Config:
    """Read a YAML config file into a `Config`."""
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level")
    return Config.from_dict(raw)


# --- Training -------------------------------------------------------------


@dataclass(frozen=True)
class AugmentationConfig:
    """ImageDataGenerator arguments used for the training split only."""

    rotation_range: float = 20.0
    horizontal_flip: bool = True
    vertical_flip: bool = False
    zoom_range: float = 0.1
    brightness_range: tuple[float, float] = (0.9, 1.1)
    shear_range: float = 0.1
    width_shift_range: float = 0.1
    height_shift_range: float = 0.1

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> AugmentationConfig:
        raw = dict(raw)
        if "brightness_range" in raw:
            lo, hi = raw["brightness_range"]
            raw["brightness_range"] = (float(lo), float(hi))
        return cls(**raw)


@dataclass(frozen=True)
class TrainConfig:
    """Settings for training and evaluating the MobileNetV2 classifier."""

    data_dir: Path = Path("data/raw")
    train_csv: str = "train.csv"
    train_images: str = "train_images"
    test_csv: str = "test.csv"
    test_images: str = "test_images"
    output_dir: Path = Path("runs")

    image_size: tuple[int, int] = (224, 224)
    batch_size: int = 25
    epochs: int = 30
    learning_rate: float = 1e-4
    dropout: float = 0.5
    valid_split: float = 0.2
    seed: int = 666
    use_class_weights: bool = True
    # Minimum softmax confidence required to accept a "normal" prediction at
    # evaluation time. Below it the image is re-labelled with its strongest
    # defect class, trading overkill for fewer escapes. 0 disables the rule.
    # Tune it on the validation split, never on the test set.
    normal_confidence_threshold: float = 0.0
    # 0 disables early stopping (the notebook trained a fixed 30 epochs).
    early_stopping_patience: int = 0
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)

    @property
    def train_csv_path(self) -> Path:
        return self.data_dir / self.train_csv

    @property
    def train_images_dir(self) -> Path:
        return self.data_dir / self.train_images

    @property
    def test_csv_path(self) -> Path:
        return self.data_dir / self.test_csv

    @property
    def test_images_dir(self) -> Path:
        return self.data_dir / self.test_images

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TrainConfig:
        raw = dict(raw)
        for key in ("data_dir", "output_dir"):
            if key in raw:
                raw[key] = Path(raw[key])
        if "image_size" in raw:
            size = raw["image_size"]
            raw["image_size"] = (int(size[0]), int(size[1]))
        raw["augmentation"] = AugmentationConfig.from_dict(raw.get("augmentation", {}))
        return cls(**raw)


def load_train_config(path: str | Path) -> TrainConfig:
    """Read a YAML training config file into a `TrainConfig`."""
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level")
    return TrainConfig.from_dict(raw)
