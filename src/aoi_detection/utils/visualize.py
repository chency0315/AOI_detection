"""Plots for data exploration and training results.

Every function saves to `save_path` when given and shows the figure when
`show=True`, so the same helpers work from scripts and notebooks.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import confusion_matrix

from aoi_detection.data.dataset import CLASS_NAMES, ID_COL, LABEL_COL


def _finish(fig: plt.Figure, save_path: str | Path | None, show: bool) -> None:
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def describe_image(path: str | Path) -> dict[str, Any]:
    """Shape, dtype and intensity range of one image (the notebook's first check)."""
    img = cv2.imread(str(path))
    if img is None:
        raise OSError(f"could not read image: {path}")
    return {
        "path": str(path),
        "shape": tuple(img.shape),
        "dtype": str(img.dtype),
        "min": int(img.min()),
        "max": int(img.max()),
    }


def plot_image_info(
    path: str | Path, save_path: str | Path | None = None, show: bool = False
) -> dict[str, Any]:
    """Show one image with its shape/dtype/range in the title."""
    info = describe_image(path)
    img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img)
    ax.set_title(
        f"{Path(path).name}\nshape={info['shape']} dtype={info['dtype']} "
        f"range=[{info['min']}, {info['max']}]",
        fontsize=9,
    )
    ax.axis("off")
    _finish(fig, save_path, show)
    return info


def plot_class_samples(
    df: pd.DataFrame,
    image_dir: str | Path,
    save_path: str | Path | None = None,
    show: bool = False,
) -> None:
    """One grayscale example per class in a 2x3 grid."""
    image_dir = Path(image_dir)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for cls, ax in enumerate(axes.flat):
        ids = df.loc[df[LABEL_COL].astype(int) == cls, ID_COL].to_numpy()
        if len(ids) == 0:
            ax.set_title(f"{cls}: {CLASS_NAMES[cls]} (no samples)")
            ax.axis("off")
            continue
        img = cv2.imread(str(image_dir / ids[0]), cv2.IMREAD_GRAYSCALE)
        ax.imshow(img, cmap="gray")
        ax.set_title(f"{cls}: {CLASS_NAMES[cls]}  {img.shape}", fontsize=10)
        ax.set_xlabel(ids[0], fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("One sample per class", fontsize=12)
    fig.tight_layout()
    _finish(fig, save_path, show)


def plot_class_distribution(
    df: pd.DataFrame, save_path: str | Path | None = None, show: bool = False
) -> None:
    """Bar chart of images per class."""
    labels = df[LABEL_COL].astype(int)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.countplot(x=labels, ax=ax, order=list(range(len(CLASS_NAMES))))
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels([f"{i}\n{n}" for i, n in enumerate(CLASS_NAMES)], fontsize=8)
    ax.set_xlabel("class")
    ax.set_ylabel("count")
    ax.set_title("Images per class")
    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2, p.get_height()),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    _finish(fig, save_path, show)


def plot_history(
    history: Mapping[str, Sequence[float]],
    out_dir: str | Path | None = None,
    show: bool = False,
) -> None:
    """Accuracy and loss curves (train vs validation), as two PNGs."""
    epochs = range(1, len(history["loss"]) + 1)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, history["accuracy"], marker="o", ms=3, label="train")
    if "val_accuracy" in history:
        ax.plot(epochs, history["val_accuracy"], marker="o", ms=3, label="validation")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title("model accuracy")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    _finish(fig, Path(out_dir) / "accuracy.png" if out_dir else None, show)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, history["loss"], marker="o", ms=3, label="train")
    if "val_loss" in history:
        ax.plot(epochs, history["val_loss"], marker="o", ms=3, label="validation")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title("model loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    _finish(fig, Path(out_dir) / "loss.png" if out_dir else None, show)


def plot_confusion_matrix(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    save_path: str | Path | None = None,
    show: bool = False,
) -> np.ndarray:
    """6x6 confusion matrix labelled with class names. Returns the matrix."""
    labels = list(range(len(CLASS_NAMES)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        cmap="Oranges",
        annot=True,
        fmt="d",
        cbar=False,
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
    )
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Confusion matrix")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)
    fig.tight_layout()
    _finish(fig, save_path, show)
    return cm
