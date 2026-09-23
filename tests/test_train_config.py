from __future__ import annotations

from pathlib import Path

from aoi_detection.config import TrainConfig, load_train_config


def test_defaults_match_notebook():
    cfg = TrainConfig()
    assert cfg.image_size == (224, 224)
    assert cfg.batch_size == 25
    assert cfg.epochs == 30
    assert cfg.learning_rate == 1e-4
    assert cfg.dropout == 0.5
    assert cfg.valid_split == 0.2
    assert cfg.seed == 666
    assert cfg.augmentation.rotation_range == 20
    assert cfg.augmentation.brightness_range == (0.9, 1.1)


def test_paths_are_derived_from_data_dir():
    cfg = TrainConfig(data_dir=Path("x"), train_csv="labels.csv", train_images="imgs")
    assert cfg.train_csv_path == Path("x/labels.csv")
    assert cfg.train_images_dir == Path("x/imgs")
    assert cfg.test_csv_path == Path("x/test.csv")


def test_from_dict_coerces_types():
    cfg = TrainConfig.from_dict(
        {
            "data_dir": "data/raw",
            "image_size": ["128", "128"],
            "augmentation": {"brightness_range": [0.8, 1.2], "zoom_range": 0.2},
        }
    )
    assert isinstance(cfg.data_dir, Path)
    assert cfg.image_size == (128, 128)
    assert cfg.augmentation.brightness_range == (0.8, 1.2)
    assert cfg.augmentation.zoom_range == 0.2
    assert cfg.augmentation.rotation_range == 20  # untouched default


def test_repo_config_loads():
    cfg = load_train_config(Path(__file__).parents[1] / "configs" / "train_mobilenet_v2.yaml")
    assert cfg.epochs == 30
    assert cfg.use_class_weights is True
