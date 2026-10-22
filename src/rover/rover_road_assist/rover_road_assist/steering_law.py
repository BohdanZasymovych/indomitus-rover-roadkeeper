#!/usr/bin/env python3
"""The yaw rate that brings the rover back to the road centre, from one estimate.

Stateless: it says where the rover should steer, not whether or how much the
operator's command should give way to it — that is assist_state's job.

Conventions are REP-103 throughout: a positive offset or heading error means
the road is to the left, so it asks for a positive (left) yaw rate. The camera
publishes the opposite sign; road_error() is the one place that converts.

The law is a PD controller on the offset. The offset changes as
de/dt = v·sin(psi), so the heading error already is its measured derivative
and no numerical differentiation is needed. For small angles the closed loop is

    e'' + alpha·K_h·e' + alpha·K_h·k_e·v/(v + v_0)·e = 0

which does not overshoot while alpha·K_h is about 4·k_e or more.

No ROS import anywhere in here — steering_law is deliberately standalone.
"""

import math
from dataclasses import dataclass


#: Floor on the speed the turn-rate limit divides by, so it stays finite when
#: the rover is crawling.
MIN_LIMIT_SPEED = 0.1


@dataclass(frozen=True)
class SteeringParams:
    #: [m] Turns the camera's normalised offset into metres.
    road_width: float = 3.0
    #: [1/s] How steeply an offset turns into an approach angle.
    k_e: float = 0.3
    #: [m/s] Keeps the approach angle sane near standstill.
    v_0: float = 0.5
    #: [rad] Steepest approach toward the centre line.
    theta_max: float = 0.35
    #: [1/s] Yaw rate per radian of (heading error + approach angle).
    k_h: float = 1.2
    #: Offsets inside this fraction of the half-width are ignored.
    offset_deadband: float = 0.1
    #: [rad] Heading errors inside this are ignored, while the offset is
    #: inside its own deadband.
    heading_deadband: float = 0.05
    #: [rad/s] Hard cap on the correction's yaw rate.
    max_yaw_rate: float = 0.5
    #: [m/s²] Caps the correction's yaw rate further as speed rises.
    max_lateral_accel: float = 1.0


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def smoothstep(x: float, a: float, b: float) -> float:
    """0 at or below a, 1 at or above b, a smooth S-curve in between."""
    if b <= a:
        return 1.0 if x >= b else 0.0
    t = clamp((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def deadband(x: float, d: float) -> float:
    """Soft deadband: zero inside ±d, and shifted toward zero by d outside it,
    so the output has no step at the edge."""
    return math.copysign(max(0.0, abs(x) - d), x)


def road_error(lateral_offset: float, heading_error: float,
               road_width: float) -> tuple[float, float]:
    """RoadDeviation's (offset, heading) in this module's convention.

    The message gives the offset as a fraction of the half-width and both
    errors with + = right; here the offset is in metres and + = left.
    """
    return -lateral_offset * road_width / 2.0, -heading_error


def yaw_rate_limit(v: float, p: SteeringParams) -> float:
    """Largest correction allowed at speed v: the same lateral acceleration
    is a much smaller yaw rate at speed."""
    return min(p.max_yaw_rate, p.max_lateral_accel / max(v, MIN_LIMIT_SPEED))


def correction_yaw_rate(offset: float, heading: float, v: float,
                        p: SteeringParams) -> float:
    """Yaw rate that turns the rover onto the road direction plus an approach
    angle toward the centre line, within yaw_rate_limit().

    offset [m] and heading [rad] are REP-103, see road_error(); v [m/s] is
    the forward speed. The law is forward-only: a reverse speed counts as
    standing still, which also keeps v + v_0 away from zero.
    """
    v = max(v, 0.0)
    e = deadband(offset, p.offset_deadband * p.road_width / 2.0)
    # The heading error is the damping term, so it is only deadbanded while
    # the offset is too: ignoring a small heading error during a correction
    # leaves the rover free to drift across the road at v·sin(heading_deadband)
    # and weave from edge to edge of the offset deadband indefinitely.
    psi = deadband(heading, p.heading_deadband) if e == 0.0 else heading

    approach = clamp(math.atan(p.k_e * e / (v + p.v_0)), -p.theta_max, p.theta_max)
    limit = yaw_rate_limit(v, p)
    return clamp(p.k_h * (psi + approach), -limit, limit)
