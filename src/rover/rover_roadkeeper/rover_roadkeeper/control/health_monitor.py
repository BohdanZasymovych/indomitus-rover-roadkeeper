import time
import numpy as np
import cv2
from enum import Enum
from typing import Dict, Any, Tuple

from rover_roadkeeper.config import (
    LINK_LOSS_MEAN_GRAY_TH,
    LINK_LOSS_MIN_DURATION_SEC,
    GLARE_PIXEL_INTENSITY_TH,
    GLARE_AREA_RATIO_TH,
    BUSH_ENTRAPMENT_VEG_RATIO,
    HUD_MASK_BOX,
    ORIG_WIDTH,
    ORIG_HEIGHT,
)


class SystemHealthState(Enum):
    HEALTHY = "HEALTHY"
    LINK_LOST = "LINK_LOST"
    CAMERA_GLARE_WARNING = "CAMERA_GLARE_WARNING"
    OFF_ROAD_RECOVERY = "OFF_ROAD_RECOVERY"


class HealthMonitor:
    """
    Monitors camera stream & link health:
      1. Link Loss (Starlink dropouts, black screen with reconnect text)
      2. IR Glare / Lens saturation (e.g. dev_0012)
      3. Off-Road Bush Entrapment (e.g. dev_0011)
    """

    def __init__(
        self,
        link_loss_gray_th: float = LINK_LOSS_MEAN_GRAY_TH,
        link_loss_duration_sec: float = LINK_LOSS_MIN_DURATION_SEC,
        glare_pixel_th: int = GLARE_PIXEL_INTENSITY_TH,
        glare_ratio_th: float = GLARE_AREA_RATIO_TH,
        veg_entrapment_th: float = BUSH_ENTRAPMENT_VEG_RATIO,
    ):
        self.link_loss_gray_th = link_loss_gray_th
        self.link_loss_duration_sec = link_loss_duration_sec
        self.glare_pixel_th = glare_pixel_th
        self.glare_ratio_th = glare_ratio_th
        self.veg_entrapment_th = veg_entrapment_th

        self.black_screen_start_time: float = 0.0
        self.is_link_lost: bool = False

    def update(
        self,
        frame_bgr: np.ndarray,
        class_mask: np.ndarray,
        current_time_sec: float,
    ) -> Tuple[SystemHealthState, Dict[str, Any]]:
        """
        Evaluates frame brightness, glare blobs, and semantic vegetation/road ratio.

        Returns:
            state: Active SystemHealthState
            diagnostics: Dictionary with telemetry measurements
        """
        h, w = frame_bgr.shape[:2]

        # 1. Grayscale intensity excluding bottom-left HUD box
        if len(frame_bgr.shape) == 3:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame_bgr

        # Exclude HUD box from mean brightness calculation
        y_hud_min, y_hud_max = HUD_MASK_BOX["y_min"], HUD_MASK_BOX["y_max"]
        x_hud_min, x_hud_max = HUD_MASK_BOX["x_min"], HUD_MASK_BOX["x_max"]

        valid_pixels = np.ones((h, w), dtype=bool)
        valid_pixels[y_hud_min:y_hud_max, x_hud_min:x_hud_max] = False
        mean_gray = float(np.mean(gray[valid_pixels]))

        # Link-Loss Detection
        if mean_gray < self.link_loss_gray_th:
            if self.black_screen_start_time == 0.0:
                self.black_screen_start_time = current_time_sec
            elif (current_time_sec - self.black_screen_start_time) >= self.link_loss_duration_sec:
                self.is_link_lost = True
        else:
            self.black_screen_start_time = 0.0
            self.is_link_lost = False

        if self.is_link_lost:
            diagnostics = {
                "mean_gray": mean_gray,
                "glare_ratio": 0.0,
                "veg_ratio": 0.0,
                "road_ratio": 0.0,
                "alert": "LINK LOST: Black screen / Reconnecting text detected",
            }
            return SystemHealthState.LINK_LOST, diagnostics

        # 2. Central IR Glare Detection
        # Check central 40% ROI: x in [0.3w, 0.7w], y in [0.2h, 0.8h]
        y1, y2 = int(0.2 * h), int(0.8 * h)
        x1, x2 = int(0.3 * w), int(0.7 * w)
        center_roi = gray[y1:y2, x1:x2]
        glare_pixels = np.count_nonzero(center_roi >= self.glare_pixel_th)
        glare_ratio = float(glare_pixels / center_roi.size)

        if glare_ratio >= self.glare_ratio_th:
            diagnostics = {
                "mean_gray": mean_gray,
                "glare_ratio": glare_ratio,
                "veg_ratio": 0.0,
                "road_ratio": 0.0,
                "alert": f"IR GLARE: Saturated blob covering {glare_ratio * 100:.1f}% of central ROI",
            }
            return SystemHealthState.CAMERA_GLARE_WARNING, diagnostics

        # 3. Off-Road / Bush Entrapment Detection
        # Check lower half of frame (y from 0.5h to 1.0h)
        lower_mask = class_mask[int(0.5 * h):, :]
        total_lower_px = lower_mask.size
        veg_px = np.count_nonzero(lower_mask == 3)        # class 3 = vegetation
        road_px = np.count_nonzero(lower_mask == 1)       # class 1 = road

        veg_ratio = float(veg_px / total_lower_px)
        road_ratio = float(road_px / total_lower_px)

        if veg_ratio >= self.veg_entrapment_th and road_ratio < 0.10:
            diagnostics = {
                "mean_gray": mean_gray,
                "glare_ratio": glare_ratio,
                "veg_ratio": veg_ratio,
                "road_ratio": road_ratio,
                "alert": f"OFF-ROAD ENTRAPMENT: Bush covers {veg_ratio * 100:.1f}%, road {road_ratio * 100:.1f}%",
            }
            return SystemHealthState.OFF_ROAD_RECOVERY, diagnostics

        # Default healthy state
        diagnostics = {
            "mean_gray": mean_gray,
            "glare_ratio": glare_ratio,
            "veg_ratio": veg_ratio,
            "road_ratio": road_ratio,
            "alert": "Nominal",
        }
        return SystemHealthState.HEALTHY, diagnostics
