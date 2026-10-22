"""Binary road perception: swappable segmenter + single-ROI offset (no ROS imports)."""
from rover_roadkeeper.perception.base_model import BaseRoadSegmenter, preprocess_ir
from rover_roadkeeper.perception.offset_calculator import OffsetCalculator, OffsetSmoother
from rover_roadkeeper.perception.pipeline import (
    PerceptionResult,
    RoadPerceptionCore,
    build_segmenter,
    load_segmenter,
    save_checkpoint,
)

__all__ = [
    "BaseRoadSegmenter",
    "preprocess_ir",
    "OffsetCalculator",
    "OffsetSmoother",
    "PerceptionResult",
    "RoadPerceptionCore",
    "build_segmenter",
    "load_segmenter",
    "save_checkpoint",
]
