"""Training and evaluation of the MobileNetV2 classifier (requires TensorFlow)."""

from aoi_detection.training.evaluate import EvalResult, evaluate, latest_run_dir
from aoi_detection.training.trainer import TrainResult, train

__all__ = ["EvalResult", "TrainResult", "evaluate", "latest_run_dir", "train"]
