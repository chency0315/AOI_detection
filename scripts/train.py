"""Train the MobileNetV2 defect classifier.

Usage:
    python scripts/train.py --config configs/train_mobilenet_v2.yaml [--epochs N] [--run-name NAME]

Outputs land in <output_dir>/<run-name>/ (default: runs/<timestamp>/):
    best_model.keras, history.json, training_log.csv, accuracy.png, loss.png
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from aoi_detection.config import load_train_config
from aoi_detection.training.trainer import train
from aoi_detection.utils.logging import get_logger

log = get_logger("train")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/train_mobilenet_v2.yaml"))
    parser.add_argument("--epochs", type=int, help="override epochs from the config")
    parser.add_argument("--batch-size", type=int, help="override batch_size from the config")
    parser.add_argument("--run-name", help="folder name under output_dir (default: timestamp)")
    args = parser.parse_args(argv)

    cfg = load_train_config(args.config)
    overrides = {"epochs": args.epochs, "batch_size": args.batch_size}
    overrides = {k: v for k, v in overrides.items() if v}
    if overrides:
        cfg = dataclasses.replace(cfg, **overrides)

    result = train(cfg, run_name=args.run_name)

    log.info("done. best val_accuracy=%.4f (epoch %d)", result.best_val_accuracy, result.best_epoch)
    log.info("model:  %s", result.model_path)
    log.info("charts: %s", result.run_dir / "accuracy.png")
    log.info("        %s", result.run_dir / "loss.png")
    log.info("next:   python scripts/evaluate.py --run %s", result.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
