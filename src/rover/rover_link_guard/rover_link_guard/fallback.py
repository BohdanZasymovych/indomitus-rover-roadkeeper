#!/usr/bin/env python3
"""What the rover does by itself while the operator's link is down.

    IDLE ──link lost──▶ CONTINUE ──budget spent / road lost──▶ TURN? ──▶ RETREAT ──▶ HOLD
              │                                                                       ▲
              └──── not moving, or no odometry ───────────────────────────────────────┘

CONTINUE drives on along the road at a crawl, steering by steering_law, because
the dead zone often ends a little further on. RETREAT drives the recorded
trail back to a point `retreat_distance` before where the link was lost,
either in reverse or, with TURN first, forwards after a turn on the spot. HOLD
stops and stays stopped: the rover has done what it can, or cannot go on safely.
A restored link ends any of them at once (stop()).

Everything is bounded: speeds, how long and how far CONTINUE may go, and the
whole episode (max_duration). Doubt ends in a stop, never in more driving.
While the link is up this only records the trail; step() returns None.

The retreat is pure pursuit on the trail. Reversing is the same law in a frame
turned half a circle (the "gear" frame), where the rover is driving forward.

No ROS import anywhere in here — fallback is deliberately standalone.
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple, Optional

from rover_road_assist.assist_state import RoadEstimate
from rover_road_assist.steering_law import SteeringParams, clamp, correction_yaw_rate

from rover_link_guard.trail import Trail


class Phase(str, Enum):
    #: Link up: nothing to do but remember the way.
    IDLE = 'IDLE'
    #: Driving on along the road.
    CONTINUE = 'CONTINUE'
    #: Turning on the spot to face the way back.
    TURN = 'TURN'
    #: Driving the trail back.
    RETREAT = 'RETREAT'
    #: Stopped, until the link returns.
    HOLD = 'HOLD'


class RetreatMode(str, Enum):
    #: Back up along the trail.
    REVERSE = 'reverse'
    #: Turn around, then drive forward along it.
    TURN_AROUND = 'turn_around'


@dataclass(frozen=True)
class FallbackParams:
    # --- CONTINUE ---
    #: [m/s] Speed while continuing, capped by the speed the operator had.
    continue_speed: float = 0.5
    #: [m/s] Slower than this when the link dropped counts as standing still,
    #: and the rover stays put: the operator stopped it on purpose.
    min_continue_speed: float = 0.2
    #: [s] Longest CONTINUE.
    continue_time: float = 15.0
    #: [m] Furthest CONTINUE.
    continue_distance: float = 6.0
    #: Road estimates less confident than this are not steered by.
    road_min_confidence: float = 0.5
    #: [s] Road estimates older than this are ignored.
    road_max_age: float = 0.3
    #: [s] How long CONTINUE drives straight on without a trusted road before
    #: it gives up and retreats.
    road_lost_grace: float = 2.0

    # --- RETREAT ---
    #: [m/s]
    retreat_speed: float = 0.4
    #: [m] How far back before the point of loss the retreat goes: the link was
    #: still marginal there.
    retreat_distance: float = 5.0
    #: [m] Pure pursuit looks this far down the trail.
    lookahead: float = 1.5
    #: [m] A trail point this close to the rover has been reached.
    reach_radius: float = 0.5
    #: [rad/s] Fastest the turn on the spot goes.
    turn_rate: float = 0.5
    #: [1/s] Turn rate per radian of heading error.
    turn_gain: float = 1.5
    #: [rad] The turn is done within this of the way back.
    turn_tolerance: float = 0.2

    # --- Limits ---
    #: [rad/s] Hard cap on any yaw rate commanded.
    max_yaw_rate: float = 0.5
    #: [m/s²] How fast speed may build up.
    max_accel: float = 0.5
    #: [m/s²] How fast speed may fall, to a stop or through one.
    max_decel: float = 1.0
    #: [s] Longest the whole episode may last before HOLD.
    max_duration: float = 120.0
    #: [s] Odometry older than this stops the rover, until it is fresh again.
    pose_max_age: float = 0.5

    # --- Trail ---
    #: [m] Distance between recorded trail points.
    trail_spacing: float = 0.25
    #: Oldest points are forgotten past this many.
    trail_max_points: float = 4000


class Pose(NamedTuple):
    x: float
    y: float
    #: [rad]
    yaw: float
    #: [s] When it was measured.
    stamp: float


class Command(NamedTuple):
    #: [m/s]
    v: float
    #: [rad/s]
    wz: float


class _Goal(NamedTuple):
    """What a phase wants this step; the speed is ramped toward it afterwards."""

    v: float
    #: [rad/s] Used as is unless `kappa` is given.
    wz: float = 0.0
    #: [1/m] Path curvature in the gear frame; the yaw rate follows the speed.
    kappa: Optional[float] = None
    #: +1 driving forward, -1 in reverse.
    gear: int = 1


def wrap_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


class Fallback:

    def __init__(self, steering: SteeringParams, params: FallbackParams,
                 retreat: RetreatMode = RetreatMode.REVERSE):
        self._steering = steering
        self._p = params
        self._retreat = retreat
        self._trail = Trail(params.trail_spacing, int(params.trail_max_points))
        self._pose = None
        self._road = None
        self._reset()

    @property
    def phase(self) -> Phase:
        return self._phase

    @property
    def reason(self) -> str:
        """Why the phase last changed, for the log."""
        return self._reason

    def on_pose(self, pose: Pose):
        self._pose = pose
        if self._phase in (Phase.IDLE, Phase.CONTINUE):
            self._trail.add(pose.x, pose.y)

    def on_road(self, road: RoadEstimate):
        self._road = road

    def start(self, now: float, v: float):
        """The link was just lost, with the rover driving at `v` [m/s]."""
        self._reset()
        self._started = now
        self._v = v
        self._loss_s = self._trail.length
        self._start_pose = self._pose
        if self._pose is None:
            self._enter(Phase.HOLD, 'no odometry')
        elif v < self._p.min_continue_speed:
            self._enter(Phase.HOLD, 'not moving when the link dropped')
        else:
            self._enter(Phase.CONTINUE, 'link lost')

    def stop(self):
        """The link is back: hand over, forgetting the episode (not the trail)."""
        self._reset()

    def step(self, now: float) -> Optional[Command]:
        """The command to publish at `now`, or None while the link is up."""
        if self._phase is Phase.IDLE:
            return None
        dt = clamp(now - self._last_step, 0.0, 0.5) if self._last_step is not None else 0.0
        self._last_step = now

        goal = self._decide(now)
        self._v = self._ramp(goal.v, dt)
        if goal.kappa is None:
            wz = goal.wz
        elif self._v * goal.gear > 0.0:
            wz = abs(self._v) * goal.kappa
        else:
            wz = 0.0   # not driving the way the path goes: nothing to follow yet
        return Command(self._v, clamp(wz, -self._p.max_yaw_rate, self._p.max_yaw_rate))

    # ── phases ────────────────────────────────────────────────────────────

    def _decide(self, now: float) -> _Goal:
        pose = self._pose
        if self._phase is not Phase.HOLD:
            if now - self._started >= self._p.max_duration:
                self._enter(Phase.HOLD, 'time budget spent')
            elif pose is None or now - pose.stamp > self._p.pose_max_age:
                # Stopped for now, not for good: odometry may come back.
                return _Goal(0.0)
        if self._phase is Phase.CONTINUE:
            return self._continue(now, pose)
        if self._phase is Phase.TURN:
            return self._turn(pose)
        if self._phase is Phase.RETREAT:
            return self._follow_trail(pose)
        return _Goal(0.0)

    def _continue(self, now: float, pose: Pose) -> _Goal:
        p = self._p
        travelled = math.hypot(pose.x - self._start_pose.x, pose.y - self._start_pose.y)
        if now - self._started >= p.continue_time or travelled >= p.continue_distance:
            return self._begin_retreat('continue budget spent')

        v = min(self._last_v(), p.continue_speed)
        road = self._road
        if (road is not None and road.valid and road.confidence >= p.road_min_confidence
                and now - road.stamp <= p.road_max_age):
            self._untrusted_since = None
            return _Goal(v, wz=correction_yaw_rate(road.offset, road.heading, v, self._steering))

        if self._untrusted_since is None:
            self._untrusted_since = now
        if now - self._untrusted_since >= p.road_lost_grace:
            return self._begin_retreat('road lost')
        return _Goal(v)   # straight on for a moment: a dip in confidence is not the end of the road

    def _begin_retreat(self, reason: str) -> _Goal:
        self._goal_s = max(0.0, self._loss_s - self._p.retreat_distance)
        if self._retreat is RetreatMode.TURN_AROUND:
            self._enter(Phase.TURN, reason)
            return self._turn(self._pose)
        self._gear = -1
        self._enter(Phase.RETREAT, reason)
        return self._follow_trail(self._pose)

    def _turn(self, pose: Pose) -> _Goal:
        p = self._p
        self._trail.pop_reached(pose.x, pose.y, p.reach_radius)
        target = self._trail.target(pose.x, pose.y, p.lookahead)
        if target is None:
            self._enter(Phase.HOLD, 'no trail to follow')
            return _Goal(0.0)
        error = wrap_angle(math.atan2(target.y - pose.y, target.x - pose.x) - pose.yaw)
        if abs(error) < p.turn_tolerance:
            self._gear = 1
            self._enter(Phase.RETREAT, 'turned around')
            return self._follow_trail(pose)
        if abs(self._v) > 0.05:
            return _Goal(0.0)   # let it stop before turning
        return _Goal(0.0, wz=clamp(p.turn_gain * error, -p.turn_rate, p.turn_rate))

    def _follow_trail(self, pose: Pose) -> _Goal:
        p = self._p
        self._trail.pop_reached(pose.x, pose.y, p.reach_radius)
        if len(self._trail) == 0 or self._trail.length <= self._goal_s:
            self._enter(Phase.HOLD, 'back at the retreat point')
            return _Goal(0.0)
        target = self._trail.target(pose.x, pose.y, p.lookahead)

        gear = self._gear
        yaw = pose.yaw + (math.pi if gear < 0 else 0.0)
        dx, dy = target.x - pose.x, target.y - pose.y
        ahead = math.cos(yaw) * dx + math.sin(yaw) * dy
        left = -math.sin(yaw) * dx + math.cos(yaw) * dy
        if ahead <= 0.0:
            # Pure pursuit cannot reach a point beside or behind it.
            self._enter(Phase.HOLD, 'off the trail')
            return _Goal(0.0)
        kappa = 2.0 * left / (ahead * ahead + left * left)

        # Braking out of the opposite direction first, never straight through.
        v = gear * p.retreat_speed if self._v * gear >= 0.0 else 0.0
        return _Goal(v, kappa=kappa, gear=gear)

    # ── helpers ───────────────────────────────────────────────────────────

    def _ramp(self, target: float, dt: float) -> float:
        """self._v moved toward `target` within the acceleration limits."""
        p = self._p
        slowing = abs(target) < abs(self._v) or target * self._v < 0.0
        rate = (p.max_decel if slowing else p.max_accel) * dt
        return self._v + clamp(target - self._v, -rate, rate)

    def _last_v(self) -> float:
        return self._entry_v

    def _enter(self, phase: Phase, reason: str):
        if phase is Phase.CONTINUE:
            self._entry_v = self._v
        self._phase = phase
        self._reason = reason

    def _reset(self):
        self._phase = Phase.IDLE
        self._reason = ''
        self._started = 0.0
        self._last_step = None
        self._v = 0.0
        self._entry_v = 0.0
        self._loss_s = 0.0
        self._goal_s = 0.0
        self._gear = 1
        self._start_pose = None
        self._untrusted_since = None
