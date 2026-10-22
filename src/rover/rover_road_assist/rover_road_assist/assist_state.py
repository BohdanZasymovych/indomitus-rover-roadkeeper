#!/usr/bin/env python3
"""Whether, and how hard, the road correction may take over the operator's turn.

steering_law says where the rover should steer; this decides how much of that
reaches the wheels. Each operator command is blended with the correction by a
strength alpha in [0, alpha_max]:

    wz_target = (1 - alpha)·wz_operator + alpha·wz_correction

alpha follows a target that is zero unless the assist is on, the road
estimate is fresh, valid and trusted, the rover is moving forward, and the
operator has not steered hard recently. It ramps at alpha_rate either way,
so the assist never grabs or lets go of the wheel in a single step.

No ROS import anywhere in here — assist_state is deliberately standalone.
"""

import math
from dataclasses import dataclass
from enum import Enum

from rover_road_assist.steering_law import SteeringParams, clamp, correction_yaw_rate, smoothstep


class Mode(str, Enum):
    #: Pass the operator's command through.
    OFF = 'off'
    #: Pass it through too; the status still says what the assist would do.
    WARN = 'warn'
    #: Correct the steering.
    ASSIST = 'assist'


class Status(str, Enum):
    OFF = 'OFF'
    WARN = 'WARN'
    #: Armed and trusting the road. alpha may still be zero below min_speed.
    ASSIST_ACTIVE = 'ASSIST_ACTIVE'
    #: No usable road in view, or not confident enough in it.
    STANDBY_LOW_CONF = 'STANDBY_LOW_CONF'
    #: No road estimate yet, or the last one is too old.
    STALE = 'STALE'
    #: The operator steered hard within the last override_hold.
    OVERRIDE = 'OVERRIDE'


@dataclass(frozen=True)
class AssistParams:
    #: Strongest the correction may ever be, out of 1.
    alpha_max: float = 0.5
    #: [1/s] How fast alpha may move toward its target.
    alpha_rate: float = 2.0
    #: [m/s] No correction at or below this forward speed; full strength
    #: from twice it.
    min_speed: float = 0.3
    #: Confidence must stay above this for confidence_on_time to engage.
    confidence_on: float = 0.6
    #: Disengages below this. Strength ramps up from here...
    confidence_off: float = 0.4
    #: ...to full here.
    confidence_full: float = 0.8
    #: [s]
    confidence_on_time: float = 0.3
    #: [rad/s] An operator yaw rate above this is a deliberate turn.
    override_yaw_rate: float = 0.4
    #: [s] How long the assist stays out of the way after one.
    override_hold: float = 1.5
    #: [rad/s²] How fast the yaw rate the assist adds may change.
    max_yaw_accel: float = 1.0
    #: [s] Road estimates older than this are ignored.
    max_road_age: float = 0.3
    #: [s] A pause in operator commands longer than this starts over. Matches
    #: the cmd_vel_remote timeout in twist_mux.yaml.
    command_gap: float = 0.5


@dataclass(frozen=True)
class RoadEstimate:
    """One road estimate, in steering_law's convention (see road_error())."""

    #: [m] + = road centre to the left.
    offset: float
    #: [rad] + = road heads to the left.
    heading: float
    #: [0..1]
    confidence: float
    #: False = no usable road in view.
    valid: bool
    #: [s] When the image it came from was taken.
    stamp: float


class ConfidenceGate:
    """Confidence with hysteresis: opens once it has stayed above `on` for
    `on_time`, closes as soon as it drops below `off`.

    Without the hold, a single good frame in a run of bad ones would let the
    assist start pulling toward a road the camera has not really found.
    """

    def __init__(self, on: float, off: float, on_time: float):
        self._on = on
        self._off = off
        self._on_time = on_time
        self._open = False
        self._above_since = None

    @property
    def open(self) -> bool:
        return self._open

    def update(self, confidence: float, now: float) -> bool:
        if confidence < self._off:
            self._open = False
            self._above_since = None
        elif confidence > self._on:
            if self._above_since is None:
                self._above_since = now
            if now - self._above_since >= self._on_time:
                self._open = True
        else:
            # In the hysteresis band: stay as we are, but an engage has to
            # start its hold over.
            self._above_since = None
        return self._open


class RoadAssist:
    """Blends each operator command with the road correction."""

    def __init__(self, steering: SteeringParams, params: AssistParams):
        self._steering = steering
        self._p = params
        self._gate = ConfidenceGate(
            params.confidence_on, params.confidence_off, params.confidence_on_time)
        self._road = None
        self._alpha = 0.0
        # The yaw rate the assist currently adds on top of the operator's.
        self._added = 0.0
        self._last_command = None
        self._last_override = -math.inf

    @property
    def alpha(self) -> float:
        return self._alpha

    def on_road(self, road: RoadEstimate, now: float):
        self._road = road
        self._gate.update(road.confidence if road.valid else 0.0, now)

    def on_command(self, v: float, wz: float, now: float, mode: Mode) -> float:
        """The yaw rate to publish for the operator's (v, wz) at `now`."""
        dt = self._step(now)
        if abs(wz) > self._p.override_yaw_rate:
            self._last_override = now

        if mode is not Mode.ASSIST:
            self._reset()
            return wz

        target = self._target_alpha(v, now)
        self._alpha += clamp(target - self._alpha,
                             -self._p.alpha_rate * dt, self._p.alpha_rate * dt)

        road = self._road
        if road is not None and road.valid:
            correction = correction_yaw_rate(road.offset, road.heading, v, self._steering)
        else:
            correction = wz   # nothing to pull toward while alpha fades out
        blended = (1.0 - self._alpha) * wz + self._alpha * correction

        # Only what the assist adds is rate-limited, never the operator's own
        # input: limiting the whole output would delay every stick movement
        # by up to a second, even with the assist doing nothing.
        step = self._p.max_yaw_accel * dt
        self._added += clamp(blended - wz - self._added, -step, step)
        return wz + self._added

    def status(self, now: float, mode: Mode) -> Status:
        if mode is Mode.OFF:
            return Status.OFF
        if mode is Mode.WARN:
            return Status.WARN
        if not self._fresh(now):
            return Status.STALE
        if self._overridden(now):
            return Status.OVERRIDE
        if not (self._road.valid and self._gate.open):
            return Status.STANDBY_LOW_CONF
        return Status.ASSIST_ACTIVE

    def _step(self, now: float) -> float:
        """Seconds since the previous command, or 0 after a reset."""
        last, self._last_command = self._last_command, now
        if last is None or not 0.0 <= now - last <= self._p.command_gap:
            # First command, or the operator went quiet long enough for
            # twist_mux to drop this source: start from a plain pass-through
            # rather than ramp from a state that no longer means anything.
            self._reset()
            return 0.0
        return now - last

    def _reset(self):
        self._alpha = 0.0
        self._added = 0.0

    def _fresh(self, now: float) -> bool:
        return self._road is not None and now - self._road.stamp < self._p.max_road_age

    def _overridden(self, now: float) -> bool:
        return now - self._last_override < self._p.override_hold

    def _target_alpha(self, v: float, now: float) -> float:
        p = self._p
        road = self._road
        if not (self._fresh(now) and road.valid and self._gate.open and v > p.min_speed):
            return 0.0
        if self._overridden(now):
            return 0.0
        return (p.alpha_max
                * smoothstep(road.confidence, p.confidence_off, p.confidence_full)
                * smoothstep(v, p.min_speed, 2.0 * p.min_speed))
