#!/usr/bin/env bash
# Batch-inspect the raw image folder and write overlays plus report.json.
set -euo pipefail

CONFIG="${1:-configs/default.yaml}"
INPUT="${2:-data/raw}"
OUTPUT="${3:-outputs}"

python -m aoi_detection.cli --config "$CONFIG" --input "$INPUT" --output "$OUTPUT"
