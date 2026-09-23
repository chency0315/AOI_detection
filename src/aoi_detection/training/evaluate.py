"""Predict on the test set, write the submission CSV and score it when labels exist."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from aoi_detection.config import TrainConfig
from aoi_detection.data.dataset import (
    CLASS_NAMES,
    ID_COL,
    LABEL_COL,
    NORMAL_CLASS,
    has_labels,
    load_labels,
)
from aoi_detection.data.generators import make_predict_generator, steps_per_epoch
from aoi_detection.models.mobilenet_v2 import load_model
from aoi_detection.utils.logging import get_logger
from aoi_detection.utils.visualize import plot_confusion_matrix

log = get_logger(__name__)

MODEL_FILENAME = "best_model.keras"


@dataclass(frozen=True)
class EvalResult:
    run_dir: Path
    submission_path: Path
    predictions: pd.DataFrame
    accuracy: float | None
    confusion: np.ndarray | None
    report: str | None
    overridden: int = 0
    """How many predictions the normal-confidence rule flipped to a defect."""


def latest_run_dir(output_dir: str | Path) -> Path:
    """Most recently modified run folder that contains a model checkpoint."""
    candidates = [p for p in Path(output_dir).glob("*/") if (p / MODEL_FILENAME).exists()]
    if not candidates:
        raise FileNotFoundError(f"no run with {MODEL_FILENAME} under {output_dir}")
    return max(candidates, key=lambda p: (p / MODEL_FILENAME).stat().st_mtime)


def predict_dataframe(
    model_path: str | Path,
    df: pd.DataFrame,
    image_dir: str | Path,
    image_size: tuple[int, int],
    batch_size: int,
) -> np.ndarray:
    """Class probabilities, one row per `df` row, in the same order."""
    model = load_model(str(model_path))
    gen = make_predict_generator(df, image_dir, image_size, batch_size)
    probs = model.predict(gen, steps=steps_per_epoch(gen, batch_size))
    return np.asarray(probs)


def apply_normal_threshold(probs: np.ndarray, threshold: float) -> np.ndarray:
    """Argmax, except a `normal` win below `threshold` becomes a defect call.

    The rule is deliberately one-sided: passing an image needs confidence,
    rejecting one does not. That trades overkill (good parts rejected) for
    fewer escapes (defective parts passed), which is the cheaper mistake on
    most inspection lines. `threshold <= 0` is a plain argmax.
    """
    y_pred = probs.argmax(axis=-1)
    if threshold <= 0:
        return y_pred

    unsure = (y_pred == NORMAL_CLASS) & (probs[:, NORMAL_CLASS] < threshold)
    if unsure.any():
        defects = np.delete(probs[unsure], NORMAL_CLASS, axis=1)
        best = defects.argmax(axis=-1)
        # Undo the column deletion: indices at or above NORMAL_CLASS shift by one.
        y_pred[unsure] = np.where(best >= NORMAL_CLASS, best + 1, best)
    return y_pred


def evaluate(cfg: TrainConfig, run_dir: str | Path, show: bool = False) -> EvalResult:
    """Run inference over `test.csv` and write results next to the model.

    Writes into `run_dir`:
      submission.csv        ID,Label with predicted classes
      probabilities.csv     ID plus one column per class
      confusion_matrix.png  and metrics.json - only when test.csv has labels
    """
    run_dir = Path(run_dir)
    model_path = run_dir / MODEL_FILENAME
    test_df = load_labels(cfg.test_csv_path, require_label=False)

    log.info("predicting %d images from %s", len(test_df), cfg.test_images_dir)
    probs = predict_dataframe(
        model_path, test_df, cfg.test_images_dir, cfg.image_size, cfg.batch_size
    )
    threshold = cfg.normal_confidence_threshold
    y_pred = apply_normal_threshold(probs, threshold)
    overridden = int((probs.argmax(axis=-1) != y_pred).sum())
    if threshold > 0:
        log.info(
            "normal-confidence rule at %.2f: %d image(s) re-labelled as defective",
            threshold,
            overridden,
        )

    predictions = pd.DataFrame({ID_COL: test_df[ID_COL].to_numpy(), LABEL_COL: y_pred})
    submission_path = run_dir / "submission.csv"
    predictions.to_csv(submission_path, index=False)

    prob_df = pd.DataFrame(probs, columns=list(CLASS_NAMES))
    prob_df.insert(0, ID_COL, test_df[ID_COL].to_numpy())
    prob_df.to_csv(run_dir / "probabilities.csv", index=False)

    accuracy = confusion = report = None
    if has_labels(test_df):
        y_true = test_df[LABEL_COL].astype(int).to_numpy()
        accuracy = float(accuracy_score(y_true, y_pred))
        confusion = plot_confusion_matrix(
            y_true, y_pred, run_dir / "confusion_matrix.png", show=show
        )
        report = classification_report(
            y_true,
            y_pred,
            labels=list(range(len(CLASS_NAMES))),
            target_names=list(CLASS_NAMES),
            zero_division=0,
        )
        (run_dir / "metrics.json").write_text(
            json.dumps({"accuracy": accuracy, "confusion_matrix": confusion.tolist()}, indent=2)
        )
        log.info("test accuracy: %.3f", accuracy)
        log.info("\n%s", report)
    else:
        log.info("test.csv has no labels - wrote predictions only")

    return EvalResult(
        run_dir=run_dir,
        submission_path=submission_path,
        predictions=predictions,
        accuracy=accuracy,
        confusion=confusion,
        report=report,
        overridden=overridden,
    )
