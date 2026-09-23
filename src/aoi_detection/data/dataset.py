"""Label CSV loading, train/valid splitting and class balancing.

The AOI dataset ships as a folder of PNGs plus a CSV with two columns:
`ID` (file name) and `Label` (integer class 0-5).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

CLASS_NAMES: tuple[str, ...] = (
    "normal",
    "void",
    "horizontal_defect",
    "vertical_defect",
    "edge_defect",
    "particle",
)
NUM_CLASSES = len(CLASS_NAMES)
# Index of the "pass" class. Every other class is a defect of some kind.
NORMAL_CLASS = 0

ID_COL = "ID"
LABEL_COL = "Label"


def load_labels(csv_path: str | Path, require_label: bool = True) -> pd.DataFrame:
    """Read an `ID,Label` CSV. Raises if the expected columns are missing."""
    df = pd.read_csv(csv_path, index_col=False)
    if ID_COL not in df.columns:
        raise ValueError(f"{csv_path}: missing {ID_COL!r} column")
    if require_label and LABEL_COL not in df.columns:
        raise ValueError(f"{csv_path}: missing {LABEL_COL!r} column")
    return df


def has_labels(df: pd.DataFrame) -> bool:
    return LABEL_COL in df.columns and df[LABEL_COL].notna().all()


def class_counts(df: pd.DataFrame) -> pd.Series:
    """Number of images per class, indexed 0..NUM_CLASSES-1 (missing classes are 0)."""
    counts = df[LABEL_COL].astype(int).value_counts()
    return counts.reindex(range(NUM_CLASSES), fill_value=0).sort_index()


def split_train_valid(
    df: pd.DataFrame, valid_split: float = 0.2, seed: int = 666
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Random train/valid split with labels cast to `str` for Keras generators."""
    train, valid = train_test_split(df, test_size=valid_split, random_state=seed)
    train = train.reset_index(drop=True).copy()
    valid = valid.reset_index(drop=True).copy()
    train[LABEL_COL] = train[LABEL_COL].astype(str)
    valid[LABEL_COL] = valid[LABEL_COL].astype(str)
    return train, valid


def compute_class_weights(train: pd.DataFrame) -> dict[int, float]:
    """sklearn 'balanced' weights keyed by integer class index."""
    labels = train[LABEL_COL].astype(int).to_numpy()
    classes = np.unique(labels)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    return {int(c): float(w) for c, w in zip(classes, weights)}
