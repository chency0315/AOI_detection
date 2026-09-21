# AOI_detection

AOI detection using a CNN deep learning model.

Automated Optical Inspection (AOI) — finding surface and assembly defects in
product images. A classical threshold baseline ships today; the CNN detector
plugs into the same `Detector` interface and model registry.

## Layout

```
src/aoi_detection/
  config.py        # typed settings loaded from YAML + environment
  data/            # dataset loading, splits, augmentation
  models/          # detector definitions and checkpoint loading
  pipeline/        # preprocess -> infer -> postprocess -> report
  utils/           # imaging helpers, logging, metrics
configs/           # experiment configs (default.yaml)
data/raw/          # source images (git-ignored)
data/processed/    # derived tensors/crops (git-ignored)
notebooks/         # exploration
scripts/           # entry points for training / batch inference
tests/             # pytest suite
```

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Run inference over a folder of images:

```bash
python -m aoi_detection.cli --config configs/default.yaml --input data/raw --output outputs/
```

This writes `report.json` plus a `*_overlay.png` per failing image, and exits
non-zero when any image fails inspection — usable as a CI gate.

## Adding the CNN detector

`models/base.py` defines the contract: image in, binary defect mask out.
Implement it, register the name in `models/registry.py`, and select it from
config — the pipeline, CLI, and reporting need no changes.

```yaml
detector:
  name: cnn
  weights: checkpoints/best.pt
```

## Development

```bash
ruff check src tests      # lint
ruff format src tests     # format
mypy src                  # type-check
pytest --cov=aoi_detection
```

## Status

Scaffold. The threshold baseline works end to end and is covered by tests; the
CNN model is not implemented yet.
