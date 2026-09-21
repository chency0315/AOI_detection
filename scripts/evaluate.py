"""Evaluate a trained run on test.csv: submission CSV, accuracy, confusion matrix.

Usage:
    python scripts/evaluate.py [--config CFG.yaml] [--run runs/<name>] [--show]

Without --run the most recent run containing best_model.keras is used.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from aoi_detection.config import load_train_config
from aoi_detection.training.evaluate import evaluate, latest_run_dir
from aoi_detection.utils.logging import get_logger

log = get_logger("evaluate")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/train_mobilenet_v2.yaml"))
    parser.add_argument("--run", type=Path, help="run directory (default: latest under output_dir)")
    parser.add_argument("--show", action="store_true", help="open the confusion matrix in a window")
    args = parser.parse_args(argv)

    cfg = load_train_config(args.config)
    run_dir = args.run or latest_run_dir(cfg.output_dir)
    log.info("evaluating %s", run_dir)

    result = evaluate(cfg, run_dir, show=args.show)

    log.info("submission: %s", result.submission_path)
    if result.accuracy is not None:
        log.info("accuracy:   %.3f", result.accuracy)
        log.info("confusion:  %s", result.run_dir / "confusion_matrix.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
