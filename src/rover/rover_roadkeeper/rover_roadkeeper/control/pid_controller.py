from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from rover_roadkeeper.config import (
    KP_STEER,
    KD_STEER,
    EMA_ALPHA,
    LANE_DEPARTURE_TH,
    LOW_ROAD_CONFIDENCE_TH,
)
from rover_roadkeeper.control.health_monitor import SystemHealthState


@dataclass
class ControlCommand:
    steer: float             # Normalized steering [-1.0 (left) .. +1.0 (right)]
    target_speed_pct: float  # Recommended speed [0.0 .. 100.0 %]
    smoothed_error: float    # EMA filtered lateral error [-1.0 .. 1.0]
    raw_error: float         # Instantaneous lateral error [-1.0 .. 1.0]
    is_ldw_active: bool      # Lane Departure Warning triggered
    status_label: str        # Human-readable state for HUD telemetry


class PDSteeringController:
    """
    PD Steering Controller with EMA temporal filter,
    adaptive cruise speed scaling, and Lane-Departure Warning (LDW).
    """

    def __init__(
        self,
        kp: float = KP_STEER,
        kd: float = KD_STEER,
        ema_alpha: float = EMA_ALPHA,
        ldw_threshold: float = LANE_DEPARTURE_TH,
        low_conf_threshold: float = LOW_ROAD_CONFIDENCE_TH,
    ):
        self.kp = kp
        self.kd = kd
        self.ema_alpha = ema_alpha
        self.ldw_threshold = ldw_threshold
        self.low_conf_threshold = low_conf_threshold

        self.prev_smoothed_error: Optional[float] = None
        self.last_time_sec: Optional[float] = None

    def reset(self):
        self.prev_smoothed_error = None
        self.last_time_sec = None

    def compute(
        self,
        raw_error: float,
        road_confidence: float,
        is_road_visible: bool,
        health_state: SystemHealthState,
        current_time_sec: float,
        dt_fallback: float = 0.05,
    ) -> ControlCommand:
        """
        Computes PD steering and adaptive cruise speed.

        Args:
            raw_error: Normalized lateral error [-1.0 .. 1.0].
            road_confidence: Road detection confidence [0.0 .. 1.0].
            is_road_visible: Whether road is detected in ROI.
            health_state: SystemHealthState from HealthMonitor.
            current_time_sec: Monotonic timestamp in seconds.
            dt_fallback: Fallback time step in seconds.
        """
        # Time delta
        if self.last_time_sec is not None:
            dt = max(current_time_sec - self.last_time_sec, 1e-4)
        else:
            dt = dt_fallback
        self.last_time_sec = current_time_sec

        # Handle anomalous system health states first
        if health_state == SystemHealthState.LINK_LOST:
            # Memory Hold: Keep steering centered, drop speed to cautious crawl forward
            return ControlCommand(
                steer=0.0,
                target_speed_pct=20.0,
                smoothed_error=0.0,
                raw_error=0.0,
                is_ldw_active=False,
                status_label="LINK LOST - AUTONOMOUS RECOVERY HOLD",
            )
        elif health_state == SystemHealthState.CAMERA_GLARE_WARNING:
            # Glare alert: slow down to 25%, gently hold heading
            return ControlCommand(
                steer=0.0,
                target_speed_pct=25.0,
                smoothed_error=0.0,
                raw_error=0.0,
                is_ldw_active=False,
                status_label="CAMERA GLARE - CAUTIOUS SLOWDOWN",
            )
        elif health_state == SystemHealthState.OFF_ROAD_RECOVERY:
            # Off-road / dense bush: STOP to prevent mechanical damage
            return ControlCommand(
                steer=0.0,
                target_speed_pct=0.0,
                smoothed_error=0.0,
                raw_error=0.0,
                is_ldw_active=True,
                status_label="OFF-ROAD ENTRAPMENT - FULL STOP",
            )

        # Exponential Moving Average (EMA) smoothing for bumpy camera motion
        if self.prev_smoothed_error is None:
            smoothed_e = raw_error
        else:
            smoothed_e = self.ema_alpha * raw_error + (1.0 - self.ema_alpha) * self.prev_smoothed_error

        # Derivative calculation
        if self.prev_smoothed_error is not None:
            de_dt = (smoothed_e - self.prev_smoothed_error) / dt
        else:
            de_dt = 0.0

        self.prev_smoothed_error = smoothed_e

        # PD steering output
        steer = self.kp * smoothed_e + self.kd * de_dt
        steer = float(np.clip(steer, -1.0, 1.0))

        # Lane Departure Warning (LDW)
        is_ldw = abs(smoothed_e) >= self.ldw_threshold

        # Adaptive Speed Control (0% .. 100%)
        # Base cruising speed is 85%
        base_speed = 85.0

        # Reduce speed on curvature / high lateral offset
        turn_penalty = 45.0 * min(abs(smoothed_e), 1.0)

        # Reduce speed on low road confidence
        if road_confidence < self.low_conf_threshold or not is_road_visible:
            conf_scale = max(road_confidence / self.low_conf_threshold, 0.2)
        else:
            conf_scale = 1.0

        target_speed = (base_speed - turn_penalty) * conf_scale
        target_speed = float(np.clip(target_speed, 0.0, 100.0))

        # Determine human-readable status
        if not is_road_visible:
            status = "NO ROAD DETECTED - SEARCHING"
            target_speed = 15.0
        elif is_ldw:
            status = "LANE DEPARTURE WARNING"
        else:
            status = "CRUISE CONTROL ACTIVE"

        return ControlCommand(
            steer=steer,
            target_speed_pct=target_speed,
            smoothed_error=smoothed_e,
            raw_error=raw_error,
            is_ldw_active=is_ldw,
            status_label=status,
        )
