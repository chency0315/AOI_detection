from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aoi_detection.data.dataset import (
    NUM_CLASSES,
    class_counts,
    compute_class_weights,
    has_labels,
    load_labels,
    split_train_valid,
)


@pytest.fixture
def labels() -> pd.DataFrame:
    # 100 rows, deliberately imbalanced: class 2 has only 4 images.
    counts = {0: 30, 1: 20, 2: 4, 3: 16, 4: 10, 5: 20}
    labels = [cls for cls, n in counts.items() for _ in range(n)]
    ids = [f"train_{i:05d}.png" for i in range(len(labels))]
    return pd.DataFrame({"ID": ids, "Label": labels})


def test_load_labels_validates_columns(tmp_path):
    good = tmp_path / "train.csv"
    good.write_text("ID,Label\na.png,0\nb.png,5\n", encoding="utf-8")
    df = load_labels(good)
    assert list(df.columns) == ["ID", "Label"]
    assert len(df) == 2

    bad = tmp_path / "bad.csv"
    bad.write_text("file,cls\na.png,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ID"):
        load_labels(bad)


def test_load_labels_without_label_column(tmp_path):
    path = tmp_path / "test.csv"
    path.write_text("ID\na.png\n", encoding="utf-8")
    df = load_labels(path, require_label=False)
    assert not has_labels(df)


def test_class_counts_fills_missing_classes(labels):
    counts = class_counts(labels[labels["Label"] != 2])
    assert list(counts.index) == list(range(NUM_CLASSES))
    assert counts[2] == 0
    assert counts.sum() == 96


def test_split_is_reproducible_and_stringifies_labels(labels):
    train_a, valid_a = split_train_valid(labels, valid_split=0.2, seed=666)
    train_b, valid_b = split_train_valid(labels, valid_split=0.2, seed=666)

    assert len(train_a) == 80 and len(valid_a) == 20
    assert train_a.equals(train_b) and valid_a.equals(valid_b)
    assert train_a["Label"].dtype == object  # Keras generators need str labels
    assert set(train_a["ID"]).isdisjoint(valid_a["ID"])
    assert list(train_a.index) == list(range(80))


def test_class_weights_favour_rare_classes(labels):
    weights = compute_class_weights(labels)
    assert set(weights) == set(range(NUM_CLASSES))
    # balanced weights: n_samples / (n_classes * count)
    assert weights[2] == pytest.approx(100 / (6 * 4))
    assert weights[0] == pytest.approx(100 / (6 * 30))
    assert weights[2] > weights[0]
    assert np.isclose(sum(w * c for w, c in zip(weights.values(), class_counts(labels))), 100)
