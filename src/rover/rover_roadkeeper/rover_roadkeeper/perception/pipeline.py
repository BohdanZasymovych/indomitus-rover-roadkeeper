"""
RoadPerceptionCore: the ONE object both test_video.py and the ROS 2 node use.

    core = RoadPerceptionCore("weights/road_segmenter_best.pt")
    result = core.process_frame(frame_bgr)
    result.offset, result.is_valid, result.road_confidence, result.debug_frame ...

Pure Python / PyTorch / OpenCV - no ROS imports.
"""
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch

from rover_roadkeeper import config as C
from rover_roadkeeper.perception.base_model import BaseRoadSegmenter
from rover_roadkeeper.perception.offset_calculator import OffsetCalculator


# ----------------------------------------------------------------------------
# Model factory / checkpoint I/O (swap architectures by name)
# ----------------------------------------------------------------------------
def build_segmenter(arch: str = "dinov2") -> BaseRoadSegmenter:
    if arch == "dinov2":
        from rover_roadkeeper.perception.dinov2_model import DINOv2RoadSegmenter
        return DINOv2RoadSegmenter()
    if arch == "unet":
        from rover_roadkeeper.perception.unet_model import UNetRoadSegmenter
        return UNetRoadSegmenter()
    raise ValueError(f"Unknown segmenter arch '{arch}' (choose 'dinov2' or 'unet')")


def save_checkpoint(model: BaseRoadSegmenter, path: Union[str, Path], **extra) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"arch": model.ARCH, "state_dict": model.trainable_state_dict(), **extra}, str(path))


def load_segmenter(path: Union[str, Path], device: Union[str, torch.device] = "cpu") -> BaseRoadSegmenter:
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    if "arch" not in ckpt:
        raise RuntimeError(f"{path} is not a binary road-segmenter checkpoint (old multi-class format?). "
                           f"Retrain with train.py.")
    model = build_segmenter(ckpt["arch"])
    model.load_trainable_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


# ----------------------------------------------------------------------------
# Result
# ----------------------------------------------------------------------------
@dataclass
class PerceptionResult:
    offset: float                                # smoothed lateral offset [-1, 1]
    is_valid: bool                               # valid road found in ROI this frame
    road_confidence: float                       # [0, 1]
    road_center_xy: Optional[Tuple[int, int]]    # raw road centroid (pixels, 720x364 frame)
    clean_mask: np.ndarray                       # uint8 {0,1} (FRAME_HEIGHT, FRAME_WIDTH)
    raw_offset: Optional[float] = None
    prob_map: Optional[np.ndarray] = field(default=None, repr=False)
    debug_frame: Optional[np.ndarray] = field(default=None, repr=False)
    latency_ms: float = 0.0


# ----------------------------------------------------------------------------
# Core
# ----------------------------------------------------------------------------
class RoadPerceptionCore:
    def __init__(
        self,
        checkpoint_path: Union[str, Path] = C.DEFAULT_CHECKPOINT,
        device: Optional[str] = None,
        prob_threshold: float = C.ROAD_PROB_THRESHOLD,
        model: Optional[BaseRoadSegmenter] = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = (model.to(self.device).eval() if model is not None
                      else load_segmenter(checkpoint_path, self.device))
        self.offset_calc = OffsetCalculator(prob_threshold=prob_threshold)

    def reset(self) -> None:
        """Reset temporal state (call when switching videos / after long link loss)."""
        self.offset_calc.reset()

    def process_frame(self, frame_bgr: np.ndarray, render_debug: bool = True) -> PerceptionResult:
        t0 = time.perf_counter()
        prob = self.model.predict_mask(frame_bgr, self.device)
        mask, m, offset = self.offset_calc.update(prob)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        result = PerceptionResult(
            offset=offset,
            is_valid=m.is_valid,
            road_confidence=m.road_confidence,
            road_center_xy=m.road_center_xy,
            clean_mask=mask,
            raw_offset=m.raw_offset,
            prob_map=prob,
            latency_ms=latency_ms,
        )
        if render_debug:
            result.debug_frame = self.render_debug(frame_bgr, result)
        return result

    # ------------------------------------------------------------------
    def render_debug(self, frame_bgr: np.ndarray, r: PerceptionResult) -> np.ndarray:
        """Minimal overlay: road mask, ROI box, rover centre line, raw + smoothed centre, readout."""
        W, H = C.FRAME_WIDTH, C.FRAME_HEIGHT
        img = frame_bgr if frame_bgr.shape[:2] == (H, W) else cv2.resize(frame_bgr, (W, H), interpolation=cv2.INTER_AREA)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        out = img.copy()

        # Green semi-transparent road mask
        road = r.clean_mask.astype(bool)
        if road.any():
            out[road] = (0.55 * out[road] + 0.45 * np.array([0, 220, 0])).astype(np.uint8)

        # ROI rectangle
        x0, y0, x1, y1 = self.offset_calc.roi_box
        cv2.rectangle(out, (x0 + 1, y0), (x1 - 2, y1), (255, 200, 0), 1, cv2.LINE_AA)

        # Rover centre line
        cx = W // 2
        cv2.line(out, (cx, y0 - 10), (cx, H), (255, 255, 255), 1, cv2.LINE_AA)

        # Raw detected road centre (red) and smoothed target (yellow)
        y_dot = r.road_center_xy[1] if r.road_center_xy else (y0 + y1) // 2
        if r.road_center_xy:
            cv2.circle(out, r.road_center_xy, 5, (0, 0, 255), -1, cv2.LINE_AA)
        x_s = int(round(cx + r.offset * cx))
        cv2.circle(out, (x_s, y_dot), 8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.line(out, (cx, y_dot), (x_s, y_dot), (0, 255, 255), 2, cv2.LINE_AA)

        # Readout
        color = (0, 255, 0) if r.is_valid else (0, 0, 255)
        text = f"Offset: {r.offset:+.2f} | Valid: {r.is_valid} | Conf: {r.road_confidence:.2f}"
        cv2.rectangle(out, (0, 0), (W, 26), (0, 0, 0), -1)
        cv2.putText(out, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        cv2.putText(out, f"{r.latency_ms:5.1f} ms", (W - 85, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200), 1, cv2.LINE_AA)
        return out
