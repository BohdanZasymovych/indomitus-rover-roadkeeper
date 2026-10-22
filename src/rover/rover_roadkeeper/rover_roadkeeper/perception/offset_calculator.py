"""
Road probability map -> clean mask -> single stable lateral offset.

  1. clean_mask():  p(road) > threshold, HUD box removed, morphological open+close,
                    keep ONLY the largest connected component (measured in the lower half).
  2. measure():     one ROI box (y in [ROI_Y_MIN, ROI_Y_MAX]), one bottom-weighted centroid,
                    offset = (x_road - W/2) / (W/2) in [-1, 1]  (<0: road is left, >0: right).
  3. OffsetSmoother: EMA (alpha) + max-delta rate limiter; on invalid frames the last value
                    is held and decays smoothly toward 0.
Pure numpy/OpenCV - no torch, no ROS.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from rover_roadkeeper import config as C


@dataclass
class OffsetMeasurement:
    raw_offset: Optional[float]                 # None when no valid road in ROI
    is_valid: bool
    road_confidence: float                      # [0, 1]
    road_center_xy: Optional[Tuple[int, int]]   # raw (unsmoothed) centroid in pixels
    area_ratio: float                           # road pixels / ROI pixels


class OffsetSmoother:
    """EMA + max-delta rate limiter with hold-and-decay when measurements are missing."""

    def __init__(self, alpha: float = C.EMA_ALPHA, max_delta: float = C.MAX_OFFSET_DELTA,
                 decay: float = C.INVALID_DECAY):
        self.alpha, self.max_delta, self.decay = alpha, max_delta, decay
        self.value: Optional[float] = None

    def reset(self) -> None:
        self.value = None

    def update(self, raw: Optional[float]) -> float:
        if self.value is None:
            # Initialise directly from the first valid measurement (no slow ramp-up).
            self.value = float(raw) if raw is not None else None
            return self.value if self.value is not None else 0.0

        prev = self.value
        target = prev + self.alpha * (raw - prev) if raw is not None else prev * self.decay
        step = float(np.clip(target - prev, -self.max_delta, self.max_delta))
        self.value = float(np.clip(prev + step, -1.0, 1.0))
        return self.value


class OffsetCalculator:
    def __init__(
        self,
        width: int = C.FRAME_WIDTH,
        height: int = C.FRAME_HEIGHT,
        prob_threshold: float = C.ROAD_PROB_THRESHOLD,
        morph_kernel_size: int = C.MORPH_KERNEL_SIZE,
        roi_y_min_ratio: float = C.ROI_Y_MIN_RATIO,
        roi_y_max_ratio: float = C.ROI_Y_MAX_RATIO,
        min_area_ratio: float = C.MIN_ROAD_AREA_RATIO,
        smoother: Optional[OffsetSmoother] = None,
    ):
        self.width, self.height = width, height
        self.prob_threshold = prob_threshold
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
        self.roi_y0 = int(height * roi_y_min_ratio)
        self.roi_y1 = int(height * roi_y_max_ratio)
        self.min_area_ratio = min_area_ratio
        self.lower_y0 = int(height * C.LOWER_HALF_START_RATIO)
        # Linear row weights: rows near the rover count more than far rows.
        self.row_weights = np.linspace(1.0, C.ROI_BOTTOM_WEIGHT, self.roi_y1 - self.roi_y0, dtype=np.float32)
        self.smoother = smoother or OffsetSmoother()

    @property
    def roi_box(self) -> Tuple[int, int, int, int]:
        """(x0, y0, x1, y1) of the single measurement ROI."""
        return 0, self.roi_y0, self.width, self.roi_y1

    def reset(self) -> None:
        self.smoother.reset()

    # ------------------------------------------------------------------
    def clean_mask(self, prob: np.ndarray) -> np.ndarray:
        """float prob map -> uint8 {0,1} mask containing only the main road component."""
        binary = (prob > self.prob_threshold).astype(np.uint8)
        x0, y0, x1, y1 = C.HUD_BOX
        binary[y0:y1, x0:x1] = 0

        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self.kernel)

        n, labels, _, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if n <= 1:
            return np.zeros_like(binary)

        # Pick the component with the most pixels in the LOWER half (the road we stand on).
        lower_counts = np.bincount(labels[self.lower_y0:].ravel(), minlength=n)
        lower_counts[0] = 0
        best = int(np.argmax(lower_counts))
        if lower_counts[best] < C.MIN_COMPONENT_AREA_PX:
            return np.zeros_like(binary)
        return (labels == best).astype(np.uint8)

    def measure(self, mask: np.ndarray, prob: np.ndarray) -> OffsetMeasurement:
        roi = mask[self.roi_y0:self.roi_y1]
        ys, xs = np.nonzero(roi)
        area_ratio = float(len(xs)) / roi.size

        if area_ratio < self.min_area_ratio:
            return OffsetMeasurement(None, False, 0.0, None, area_ratio)

        w = self.row_weights[ys]
        x_road = float(np.dot(w, xs) / w.sum())
        y_road = float(np.dot(w, ys) / w.sum()) + self.roi_y0

        half_w = self.width / 2.0
        raw_offset = float(np.clip((x_road - half_w) / half_w, -1.0, 1.0))

        mean_prob = float(prob[self.roi_y0:self.roi_y1][ys, xs].mean())
        area_term = min(1.0, area_ratio / C.FULL_CONFIDENCE_AREA_RATIO)
        confidence = float(np.clip(mean_prob * area_term, 0.0, 1.0))

        return OffsetMeasurement(raw_offset, True, confidence, (int(round(x_road)), int(round(y_road))), area_ratio)

    def update(self, prob: np.ndarray) -> Tuple[np.ndarray, OffsetMeasurement, float]:
        """Full step: returns (clean_mask, raw measurement, smoothed offset)."""
        mask = self.clean_mask(prob)
        m = self.measure(mask, prob)
        smoothed = self.smoother.update(m.raw_offset)
        return mask, m, smoothed
