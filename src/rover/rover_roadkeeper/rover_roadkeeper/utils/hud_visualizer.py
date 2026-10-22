import cv2
import numpy as np
from typing import Optional

from rover_roadkeeper.config import (
    ORIG_WIDTH,
    ORIG_HEIGHT,
    CLASS_COLORS_BGR,
    HUD_MASK_BOX,
)
from rover_roadkeeper.control.path_planner import TrajectoryPlan
from rover_roadkeeper.control.pid_controller import ControlCommand
from rover_roadkeeper.control.health_monitor import SystemHealthState


class HUDVisualizer:
    """
    Renders a tactical military-grade ADAS HUD overlay:
      - Alpha-blended semantic mask
      - Trajectory polynomial path & lookahead waypoints
      - Steering angle gauge bar & Recommended speed gauge
      - Road confidence indicator & Alert banners
    """

    def __init__(
        self,
        width: int = ORIG_WIDTH,
        height: int = ORIG_HEIGHT,
        mask_alpha: float = 0.45,
    ):
        self.width = width
        self.height = height
        self.mask_alpha = mask_alpha

    def render(
        self,
        frame_bgr: np.ndarray,
        class_mask: np.ndarray,
        plan: TrajectoryPlan,
        command: ControlCommand,
        health_state: SystemHealthState,
        fps: float = 0.0,
    ) -> np.ndarray:
        """
        Renders HUD overlay onto frame_bgr.
        """
        # Ensure frame is standard (height, width)
        if frame_bgr.shape[:2] != (self.height, self.width):
            canvas = cv2.resize(frame_bgr, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        else:
            canvas = frame_bgr.copy()

        # 1. Semi-transparent semantic mask overlay
        color_mask = CLASS_COLORS_BGR[class_mask]
        # Only blend where class > 0 (ignore background remains clean)
        foreground = class_mask > 0
        if np.any(foreground):
            canvas[foreground] = cv2.addWeighted(
                canvas[foreground],
                1.0 - self.mask_alpha,
                color_mask[foreground],
                self.mask_alpha,
                0,
            )

        # 2. Draw Lookahead Bands & Waypoints
        for wp in plan.waypoints:
            # Draw subtle dashed band boundary
            cv2.line(canvas, (0, wp.y_center), (self.width, wp.y_center), (70, 70, 70), 1, cv2.LINE_AA)

            # Draw obstacle marker if present in band
            if wp.has_obstacle and wp.x_obstacle is not None:
                cv2.circle(canvas, (int(wp.x_obstacle), wp.y_center), 8, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.putText(canvas, "OBS", (int(wp.x_obstacle) - 15, wp.y_center - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA)

            # Draw road centroid (green)
            if wp.x_road is not None:
                cv2.circle(canvas, (int(wp.x_road), wp.y_center), 5, (0, 255, 0), -1, cv2.LINE_AA)

            # Draw target waypoint after obstacle repulsion (cyan)
            if wp.x_target is not None:
                cv2.circle(canvas, (int(wp.x_target), wp.y_center), 7, (255, 255, 0), 2, cv2.LINE_AA)

        # 3. Draw Planned Trajectory Curve
        if plan.poly_coeffs is not None:
            curve_pts = []
            y_eval = np.linspace(plan.waypoints[0].y_max, plan.waypoints[-1].y_min, 25)
            for y in y_eval:
                x = np.polyval(plan.poly_coeffs, y)
                if 0 <= x < self.width:
                    curve_pts.append([int(x), int(y)])
            if len(curve_pts) > 1:
                pts_arr = np.array(curve_pts, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(canvas, [pts_arr], isClosed=False, color=(0, 255, 255), thickness=3, lineType=cv2.LINE_AA)

        # 4. Vehicle Center Reference (Vertical Heading Line)
        cx = self.width // 2
        cv2.line(canvas, (cx, self.height - 10), (cx, self.height - 40), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(canvas, (cx, self.height - 10), 4, (255, 255, 255), -1)

        # 5. Burned-in HUD Box Outline (Ignored Region)
        cv2.rectangle(
            canvas,
            (HUD_MASK_BOX["x_min"], HUD_MASK_BOX["y_min"]),
            (HUD_MASK_BOX["x_max"], HUD_MASK_BOX["y_max"]),
            (50, 50, 50),
            1,
        )

        # 6. Top Status Banner (Dark Glassmorphism Bar)
        cv2.rectangle(canvas, (0, 0), (self.width, 36), (15, 15, 15), -1)
        cv2.line(canvas, (0, 36), (self.width, 36), (60, 60, 60), 1)

        # Status text with color coding
        if health_state == SystemHealthState.LINK_LOST:
            status_color = (0, 0, 255)  # Red
        elif health_state in [SystemHealthState.CAMERA_GLARE_WARNING, SystemHealthState.OFF_ROAD_RECOVERY]:
            status_color = (0, 140, 255)  # Orange
        elif command.is_ldw_active:
            status_color = (0, 220, 255)  # Yellow
        else:
            status_color = (0, 255, 100)  # Tactical green

        cv2.putText(canvas, f"STATUS: {command.status_label}", (10, 23),
                    cv2.FONT_HERSHEY_DUPLEX, 0.52, status_color, 1, cv2.LINE_AA)

        # FPS indicator
        cv2.putText(canvas, f"FPS: {fps:4.1f}", (self.width - 95, 23),
                    cv2.FONT_HERSHEY_DUPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA)

        # 7. Bottom Telemetry Cockpit (Gauges: Steer, Speed, Confidence)
        panel_y = self.height - 45
        # Background bar for gauges
        overlay = canvas.copy()
        cv2.rectangle(overlay, (105, panel_y - 5), (self.width - 10, self.height - 5), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

        # Gauge 1: Steering Bar
        steer_bar_w = 140
        steer_bar_x = 115
        steer_bar_y = panel_y + 12
        # Center marker & border
        cv2.rectangle(canvas, (steer_bar_x, steer_bar_y), (steer_bar_x + steer_bar_w, steer_bar_y + 12), (100, 100, 100), 1)
        center_bar_x = steer_bar_x + steer_bar_w // 2
        cv2.line(canvas, (center_bar_x, steer_bar_y - 2), (center_bar_x, steer_bar_y + 14), (180, 180, 180), 1)
        # Steer fill
        steer_fill_w = int((command.steer / 2.0) * steer_bar_w)
        fill_color = (0, 220, 255) if command.is_ldw_active else (0, 255, 100)
        if steer_fill_w > 0:
            cv2.rectangle(canvas, (center_bar_x, steer_bar_y + 2), (center_bar_x + steer_fill_w, steer_bar_y + 10), fill_color, -1)
        elif steer_fill_w < 0:
            cv2.rectangle(canvas, (center_bar_x + steer_fill_w, steer_bar_y + 2), (center_bar_x, steer_bar_y + 10), fill_color, -1)
        cv2.putText(canvas, f"STEER: {command.steer:+.2f}", (steer_bar_x, steer_bar_y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)

        # Gauge 2: Speed Bar
        speed_bar_w = 120
        speed_bar_x = steer_bar_x + steer_bar_w + 35
        speed_bar_y = steer_bar_y
        cv2.rectangle(canvas, (speed_bar_x, speed_bar_y), (speed_bar_x + speed_bar_w, speed_bar_y + 12), (100, 100, 100), 1)
        speed_fill_w = int((command.target_speed_pct / 100.0) * speed_bar_w)
        cv2.rectangle(canvas, (speed_bar_x + 1, speed_bar_y + 2), (speed_bar_x + speed_fill_w, speed_bar_y + 10), (0, 200, 255), -1)
        cv2.putText(canvas, f"SPEED: {command.target_speed_pct:3.0f}%", (speed_bar_x, speed_bar_y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)

        # Gauge 3: Road Confidence Bar
        conf_bar_w = 120
        conf_bar_x = speed_bar_x + speed_bar_w + 35
        conf_bar_y = steer_bar_y
        cv2.rectangle(canvas, (conf_bar_x, conf_bar_y), (conf_bar_x + conf_bar_w, conf_bar_y + 12), (100, 100, 100), 1)
        conf_fill_w = int(plan.road_confidence * conf_bar_w)
        conf_color = (0, 255, 100) if plan.road_confidence >= 0.5 else (0, 140, 255)
        cv2.rectangle(canvas, (conf_bar_x + 1, conf_bar_y + 2), (conf_bar_x + conf_fill_w, conf_bar_y + 10), conf_color, -1)
        cv2.putText(canvas, f"CONF: {plan.road_confidence * 100:3.0f}%", (conf_bar_x, conf_bar_y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)

        return canvas
