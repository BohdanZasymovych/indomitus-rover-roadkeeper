from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import numpy as np
import cv2

from rover_roadkeeper.config import (
    ORIG_WIDTH,
    ORIG_HEIGHT,
    ROI_Y_MIN_RATIO,
    ROI_Y_MAX_RATIO,
    NUM_LOOKAHEAD_BANDS,
    BAND_WEIGHTS,
    OBSTACLE_REPULSION_GAIN,
    MIN_ROAD_PIXELS_PER_BAND,
)


@dataclass
class WaypointBand:
    band_idx: int
    y_center: int
    y_min: int
    y_max: int
    x_road: Optional[float]
    x_target: Optional[float]
    x_obstacle: Optional[float]
    has_obstacle: bool
    road_pixel_count: int
    weight: float


@dataclass
class TrajectoryPlan:
    waypoints: List[WaypointBand]
    target_x: float
    normalized_error: float        # e in [-1, 1], negative = road is left, positive = road is right
    road_confidence: float         # 0.0 to 1.0
    poly_coeffs: Optional[np.ndarray]  # 2nd-degree polynomial [a, b, c] for x = a*y^2 + b*y + c
    is_road_visible: bool


class RowWiseCentroidPathPlanner:
    """
    Slices the drivable camera ROI into horizontal lookahead bands,
    computes road centers-of-mass, applies repulsive potential fields
    away from obstacles, and fits a smooth trajectory curve.
    """

    def __init__(
        self,
        image_width: int = ORIG_WIDTH,
        image_height: int = ORIG_HEIGHT,
        roi_y_min_ratio: float = ROI_Y_MIN_RATIO,
        roi_y_max_ratio: float = ROI_Y_MAX_RATIO,
        num_bands: int = NUM_LOOKAHEAD_BANDS,
        band_weights: List[float] = BAND_WEIGHTS,
        repulsion_gain: float = OBSTACLE_REPULSION_GAIN,
        min_road_pixels: int = MIN_ROAD_PIXELS_PER_BAND,
    ):
        self.image_width = image_width
        self.image_height = image_height
        self.roi_y_min = int(image_height * roi_y_min_ratio)
        self.roi_y_max = int(image_height * roi_y_max_ratio)
        self.num_bands = num_bands
        self.band_weights = band_weights
        self.repulsion_gain = repulsion_gain
        self.min_road_pixels = min_road_pixels

        self.last_valid_target_x = image_width / 2.0

    def plan(
        self,
        class_mask: np.ndarray,
        road_conf_map: Optional[np.ndarray] = None,
    ) -> TrajectoryPlan:
        """
        Plans a trajectory and computes target lateral offset from semantic mask.

        Args:
            class_mask: uint8 (H, W) array with class indices:
                        0: ignore, 1: road, 2: sky, 3: vegetation, 4: obstacle
            road_conf_map: Optional float32 (H, W) road probability map.
        """
        roi_height = self.roi_y_max - self.roi_y_min
        band_h = roi_height / self.num_bands

        waypoints: List[WaypointBand] = []
        band_x_targets: List[float] = []
        band_weights_used: List[float] = []

        # Bands are indexed from Near (bottom of ROI) to Far (top of ROI)
        for i in range(self.num_bands):
            # Band 0 is nearest: [roi_y_max - band_h, roi_y_max]
            # Band num_bands - 1 is farthest: [roi_y_min, roi_y_min + band_h]
            y_end = int(self.roi_y_max - i * band_h)
            y_start = int(self.roi_y_max - (i + 1) * band_h)
            y_center = (y_start + y_end) // 2
            weight = self.band_weights[i]

            band_mask = class_mask[y_start:y_end, :]

            # Road pixels
            road_pts = np.argwhere(band_mask == 1)
            road_pixel_count = len(road_pts)

            x_road: Optional[float] = None
            if road_pixel_count >= self.min_road_pixels:
                x_road = float(np.mean(road_pts[:, 1]))

            # Obstacle & Vegetation hazard pixels
            obs_pts = np.argwhere((band_mask == 4) | (band_mask == 3))
            has_obstacle = False
            x_obstacle: Optional[float] = None
            x_target: Optional[float] = x_road

            if x_road is not None and len(obs_pts) > 20:
                obs_xs = obs_pts[:, 1]
                # Filter obstacles near the road corridor (within +/- 150px)
                near_road_obs = obs_xs[np.abs(obs_xs - x_road) < 160]
                if len(near_road_obs) > 15:
                    has_obstacle = True
                    x_obstacle = float(np.mean(near_road_obs))

                    # Repulsive potential field away from obstacle
                    diff = x_road - x_obstacle
                    direction = 1.0 if diff >= 0 else -1.0
                    dist = max(abs(diff), 20.0)
                    repulsion = direction * (self.repulsion_gain * 2500.0 / dist)
                    # Shift target away from hazard, clamp to valid frame coordinates
                    x_target = np.clip(x_road + repulsion, 50.0, self.image_width - 50.0)

            waypoint = WaypointBand(
                band_idx=i,
                y_center=y_center,
                y_min=y_start,
                y_max=y_end,
                x_road=x_road,
                x_target=x_target,
                x_obstacle=x_obstacle,
                has_obstacle=has_obstacle,
                road_pixel_count=road_pixel_count,
                weight=weight,
            )
            waypoints.append(waypoint)

            if x_target is not None:
                band_x_targets.append(x_target)
                band_weights_used.append(weight)

        # Compute overall target X
        if band_x_targets:
            weights_norm = np.array(band_weights_used) / np.sum(band_weights_used)
            target_x = float(np.sum(np.array(band_x_targets) * weights_norm))
            self.last_valid_target_x = target_x
            is_road_visible = True
        else:
            # Fallback to last known centroid or center
            target_x = self.last_valid_target_x
            is_road_visible = False

        # Normalized cross-track error: [-1.0, 1.0]
        # Negative = turn left, Positive = turn right
        center_x = self.image_width / 2.0
        normalized_error = float((target_x - center_x) / center_x)
        normalized_error = float(np.clip(normalized_error, -1.0, 1.0))

        # Fit 2nd-degree polynomial curve x = a*y^2 + b*y + c for valid waypoints
        curve_ys = [wp.y_center for wp in waypoints if wp.x_target is not None]
        curve_xs = [wp.x_target for wp in waypoints if wp.x_target is not None]
        poly_coeffs = None
        if len(curve_xs) >= 3:
            try:
                poly_coeffs = np.polyfit(curve_ys, curve_xs, deg=2)
            except Exception:
                poly_coeffs = None
        elif len(curve_xs) == 2:
            try:
                poly_coeffs = np.polyfit(curve_ys, curve_xs, deg=1)
            except Exception:
                poly_coeffs = None

        # Road confidence calculation
        roi_mask = class_mask[self.roi_y_min:self.roi_y_max, :]
        road_area_ratio = float(np.count_nonzero(roi_mask == 1) / roi_mask.size)

        if road_conf_map is not None:
            roi_conf = road_conf_map[self.roi_y_min:self.roi_y_max, :]
            mean_prob = float(np.mean(roi_conf[roi_mask == 1])) if np.any(roi_mask == 1) else 0.0
            road_confidence = float(np.clip(0.6 * (road_area_ratio / 0.40) + 0.4 * mean_prob, 0.0, 1.0))
        else:
            road_confidence = float(np.clip(road_area_ratio / 0.40, 0.0, 1.0))

        return TrajectoryPlan(
            waypoints=waypoints,
            target_x=target_x,
            normalized_error=normalized_error,
            road_confidence=road_confidence,
            poly_coeffs=poly_coeffs,
            is_road_visible=is_road_visible,
        )
