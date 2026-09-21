"""Automated Optical Inspection defect detection."""

from aoi_detection.config import Config, load_config
from aoi_detection.pipeline.inspect import Defect, InspectionResult, Inspector

__version__ = "0.1.0"

__all__ = ["Config", "Defect", "InspectionResult", "Inspector", "load_config", "__version__"]
