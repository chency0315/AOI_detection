"""Command-line entry point for batch inspection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from aoi_detection.config import Config, load_config
from aoi_detection.data import iter_images
from aoi_detection.pipeline.inspect import Inspector
from aoi_detection.utils.imaging import draw_defects, read_image
from aoi_detection.utils.logging import get_logger

log = get_logger()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aoi-detect", description="Run AOI defect detection.")
    parser.add_argument("--input", required=True, type=Path, help="image file or directory")
    parser.add_argument("--output", type=Path, help="directory for overlays and report.json")
    parser.add_argument("--config", type=Path, help="YAML config (defaults are used if omitted)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    config = load_config(args.config) if args.config else Config()
    inspector = Inspector(config)

    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)

    report = []
    failures = 0

    for path in iter_images(args.input):
        result = inspector.inspect(read_image(path), source=path)
        passed = inspector.passed(result)
        failures += not passed

        log.info("%s: %d defect(s) - %s", path, result.defect_count, "PASS" if passed else "FAIL")
        report.append(
            {
                "source": str(path),
                "passed": passed,
                "defects": [d.bbox for d in result.defects],
            }
        )

        if args.output and config.report.save_overlays and result.defects:
            overlay = draw_defects(read_image(path), [d.bbox for d in result.defects])
            _write_overlay(overlay, args.output / f"{path.stem}_overlay.png")

    if not report:
        log.warning("no images found under %s", args.input)
        return 1

    if args.output:
        (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    log.info("inspected %d image(s), %d failed", len(report), failures)
    return 1 if failures else 0


def _write_overlay(image: np.ndarray, path: Path) -> None:
    cv2.imwrite(str(path), image)


if __name__ == "__main__":
    sys.exit(main())
