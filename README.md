# AOI defects detection

English | [繁體中文](README.zh-TW.md)

AOI detection using a CNN deep learning model.

Automated Optical Inspection (AOI) — finding surface and assembly defects in
product images. A classical threshold baseline ships today; the CNN detector
plugs into the same `Detector` interface and model registry.

## Layout

```
src/aoi_detection/
  config.py            # Config (inspection) + TrainConfig (classifier training)
  data/
    loader.py          # image discovery for the inspection CLI
    dataset.py         # train.csv loading, 8:2 split, class weights, CLASS_NAMES
    generators.py      # Keras ImageDataGenerators (augmentation, preprocess_input)
  models/
    threshold.py       # classical baseline detector
    mobilenet_v2.py    # MobileNetV2 (ImageNet) + GAP + Dropout + softmax(6)
  training/
    trainer.py         # fit loop, best-checkpoint, history.json, accuracy/loss plots
    evaluate.py        # predict test.csv, submission.csv, confusion matrix, metrics
  pipeline/            # preprocess -> infer -> postprocess -> report
  utils/
    visualize.py       # EDA plots, history curves, confusion matrix
configs/
  default.yaml         # inspection pipeline
  train_mobilenet_v2.yaml  # training hyper-parameters (mirrors the Colab notebook)
data/raw/              # train.csv, train_images/, test.csv, test_images/ (git-ignored)
notebooks/             # train_mobilenet_v2_walkthrough.ipynb (calls the modules above)
scripts/               # explore_data.py, train.py, evaluate.py, run_inspection.sh
runs/                  # training outputs, one folder per run (git-ignored)
tests/                 # pytest suite
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

## Training the MobileNetV2 defect classifier

The Colab notebook `AOI_detection_Colaboratory_(mobilenetV2).ipynb` is split into
the modules above. Six classes: `0 normal, 1 void, 2 horizontal_defect,
3 vertical_defect, 4 edge_defect, 5 particle`.

### 1. Data layout

```
data/raw/
  train.csv            # ID,Label   (2528 rows)
  train_images/        # train_00000.png ...
  test.csv             # ID[,Label] (Label optional - scored when present)
  test_images/
```

### 2. Install the training stack

```bash
pip install -e ".[train]"
```

TensorFlow 2.15 (CPU on Windows). Python 3.9-3.11.

### 3. Explore the data

```bash
python scripts/explore_data.py
```

Writes `runs/eda/image_info.png`, `class_samples.png`, `class_distribution.png`
and logs the per-class counts. Add `--show` to open the windows.

### 4. Train

```bash
python scripts/train.py --config configs/train_mobilenet_v2.yaml
```

Quick smoke test first: `python scripts/train.py --epochs 2 --run-name smoke`.

Each run writes `runs/<timestamp>/`:

| file | what |
|---|---|
| `best_model.keras` | checkpoint from the epoch with the best `val_accuracy` |
| `history.json`, `training_log.csv` | per-epoch `accuracy / loss / val_accuracy / val_loss` |
| `accuracy.png`, `loss.png` | train-vs-validation curves |
| `train_config.json` | exact hyper-parameters used |

Hyper-parameters (image size, batch, epochs, lr, dropout, augmentation, class
weights, early stopping) live in `configs/train_mobilenet_v2.yaml`.

### 5. Evaluate

```bash
python scripts/evaluate.py
```

Uses the latest run (or `--run runs/<name>`), predicts every row of `test.csv`
and writes `submission.csv`, `probabilities.csv`, and - when `test.csv` has a
`Label` column - `confusion_matrix.png`, `metrics.json` and a per-class
precision/recall report in the log. `--show` opens the confusion matrix.

### 6. Notebook

`notebooks/train_mobilenet_v2_walkthrough.ipynb` runs the same steps with the
plots inline. The original Colab notebook is kept unchanged.

### Trading overkill for escapes

Two mistakes cost differently on an inspection line: an **escape** (a defective
part passed as good) usually costs far more than **overkill** (a good part
rejected). A plain `argmax` treats them the same.

`normal_confidence_threshold` makes the decision one-sided — passing an image
needs confidence, rejecting one does not:

```yaml
normal_confidence_threshold: 0.8   # 0 disables the rule
```

```bash
python scripts/evaluate.py --normal-threshold 0.8
```

When `normal` wins with a probability below the threshold, the image is
re-labelled with its strongest defect class instead. On the 224px run this
moved accuracy 95.3% -> 96.0%, escapes 5 -> 4, overkill 0 -> 0.

**Pick the value on the validation split, not on the test set.** Sweeping
thresholds against test and then reporting that test number makes the score
optimistic. The rule also cannot fix confident errors: on this model 4 of the 5
escapes sat at P(normal) > 0.97, inside the range of genuinely normal images,
so no threshold separates them - those need more data of that defect sub-type.

### Differences from the Colab notebook

- Checkpoints use the native `.keras` format instead of legacy `.h5`.
- Test accuracy compares predictions with `test.csv`'s `Label` column (the
  notebook compared against the `ID` column).
- The confusion matrix is 6x6 with class names (the notebook hard-coded 2 ticks).
- Validation generator is not shuffled, so its metrics are deterministic.

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

The threshold baseline works end to end and is covered by tests. The MobileNetV2
classifier trains and evaluates via `scripts/train.py` / `scripts/evaluate.py`;
wiring it into the `Detector` interface for the inspection CLI is still open.

## results

<img width="1673" height="998" alt="aoi_detection_final" src="https://github.com/user-attachments/assets/eef45267-7196-4cdd-85b2-a5579c4090b8" />

