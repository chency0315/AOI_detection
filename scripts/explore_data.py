"""Data exploration: image info, one sample per class, class distribution.

Usage:
    python scripts/explore_data.py [--config configs/train_mobilenet_v2.yaml] [--show]

Plots are written to <output_dir>/eda/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from aoi_detection.config import TrainConfig, load_train_config
from aoi_detection.data.dataset import CLASS_NAMES, ID_COL, class_counts, load_labels
from aoi_detection.utils.logging import get_logger
from aoi_detection.utils.visualize import (
    plot_class_distribution,
    plot_class_samples,
    plot_image_info,
)

log = get_logger("explore_data")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/train_mobilenet_v2.yaml"))
    parser.add_argument("--show", action="store_true", help="also open the plots in a window")
    args = parser.parse_args(argv)

    cfg = load_train_config(args.config) if args.config.exists() else TrainConfig()
    out_dir = cfg.output_dir / "eda"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = cfg.train_csv_path if cfg.train_csv_path.exists() else cfg.test_csv_path
    image_dir = cfg.train_images_dir if cfg.train_csv_path.exists() else cfg.test_images_dir
    if not csv_path.exists():
        log.error("no labels CSV found at %s or %s", cfg.train_csv_path, cfg.test_csv_path)
        return 1
    if csv_path != cfg.train_csv_path:
        log.warning("train.csv not found - exploring %s instead", csv_path)

    df = load_labels(csv_path)
    log.info("%d labelled images in %s", len(df), csv_path)

    info = plot_image_info(image_dir / df.loc[0, ID_COL], out_dir / "image_info.png", args.show)
    log.info(
        "first image: shape=%s dtype=%s range=[%d, %d]",
        info["shape"],
        info["dtype"],
        info["min"],
        info["max"],
    )

    counts = class_counts(df)
    for cls, n in counts.items():
        log.info("class %d %-18s %5d", cls, CLASS_NAMES[cls], n)
    plot_class_distribution(df, out_dir / "class_distribution.png", args.show)
    plot_class_samples(df, image_dir, out_dir / "class_samples.png", args.show)

    log.info("plots written to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
