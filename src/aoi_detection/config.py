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
