from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from aoi_detection.utils.visualize import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_history,
)

tf = pytest.importorskip("tensorflow", reason="TensorFlow not installed (pip install -e .[train])")


def test_mobilenet_v2_output_shape():
    from aoi_detection.models.mobilenet_v2 import build_mobilenet_v2

    model = build_mobilenet_v2(input_shape=(32, 32, 3), weights=None)
    out = model.predict(np.zeros((2, 32, 32, 3), dtype=np.float32), verbose=0)
    assert out.shape == (2, 6)
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-5)  # softmax


def test_predict_generator_keeps_row_order(tmp_path):
    from aoi_detection.data.generators import make_predict_generator, steps_per_epoch

    ids = []
    for i in range(3):
        name = f"img_{i}.png"
        tf.keras.utils.save_img(tmp_path / name, np.full((8, 8, 3), i * 40, dtype=np.uint8))
        ids.append(name)
    df = pd.DataFrame({"ID": ids})

    gen = make_predict_generator(df, tmp_path, (8, 8), batch_size=2)
    assert gen.filenames == ids
    assert steps_per_epoch(gen, 2) == 2
    batch = next(gen)
    assert batch.shape == (2, 8, 8, 3)
    assert batch.min() >= -1.0 and batch.max() <= 1.0  # mobilenet_v2 preprocess_input


def test_plots_write_files(tmp_path):
    history = {
        "accuracy": [0.5, 0.7],
        "val_accuracy": [0.4, 0.6],
        "loss": [1.2, 0.8],
        "val_loss": [1.5, 1.0],
    }
    plot_history(history, tmp_path)
    assert (tmp_path / "accuracy.png").exists() and (tmp_path / "loss.png").exists()

    cm = plot_confusion_matrix([0, 1, 2, 2], [0, 1, 1, 2], tmp_path / "cm.png")
    assert cm.shape == (6, 6) and cm[2, 1] == 1
    assert (tmp_path / "cm.png").exists()

    df = pd.DataFrame({"ID": ["a", "b", "c"], "Label": [0, 0, 5]})
    plot_class_distribution(df, tmp_path / "dist.png")
    assert (tmp_path / "dist.png").exists()
    json.dumps(history)  # history must stay JSON-serialisable
