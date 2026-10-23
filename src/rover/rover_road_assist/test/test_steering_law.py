"""The correction yaw rate, from one road estimate.

Every sign here is a chance to steer the rover off the road instead of onto it,
and every limit a chance to yank the wheel at speed. Both are cheap to check
here and expensive to find on the rover.

No ROS import anywhere in here — steering_law is deliberately standalone.
"""

import math

import pytest

from rover_road_assist.steering_law import (
    SteeringParams,
    correction_yaw_rate,
    deadband,
    road_error,
    smoothstep,
    yaw_rate_limit,
)


PARAMS = SteeringParams()


# ── signs ────────────────────────────────────────────────────────────────────

def test_a_road_to_the_left_turns_the_rover_left():
    assert correction_yaw_rate(0.5, 0.0, 1.0, PARAMS) > 0.0
    assert correction_yaw_rate(-0.5, 0.0, 1.0, PARAMS) < 0.0


def test_a_road_heading_left_turns_the_rover_left():
    assert correction_yaw_rate(0.0, 0.2, 1.0, PARAMS) > 0.0
    assert correction_yaw_rate(0.0, -0.2, 1.0, PARAMS) < 0.0


def test_the_camera_convention_is_flipped_to_rep103():
    # RoadDeviation says + = right and normalises to the half-width; a road
    # centre at the right edge is half a road width to the rover's right.
    offset, heading = road_error(1.0, 0.1, road_width=3.0)

    assert offset == -1.5
    assert heading == -0.1


def test_a_road_the_camera_sees_on_the_right_turns_the_rover_right():
    offset, heading = road_error(0.5, 0.1, PARAMS.nominal_road_width)

    assert correction_yaw_rate(offset, heading, 1.0, PARAMS) < 0.0


# ── zero and the deadbands ───────────────────────────────────────────────────

def test_no_error_asks_for_no_correction():
    assert correction_yaw_rate(0.0, 0.0, 1.0, PARAMS) == 0.0


def test_errors_inside_both_deadbands_are_ignored():
    offset_db = PARAMS.offset_deadband * PARAMS.nominal_road_width / 2.0

    assert correction_yaw_rate(0.9 * offset_db, 0.9 * PARAMS.heading_deadband,
                               1.0, PARAMS) == 0.0


def test_the_heading_is_not_deadbanded_while_correcting_an_offset():
    # It is the damping term: dropping it mid-correction lets the rover weave.
    offset = PARAMS.nominal_road_width   # far outside the offset deadband
    small = 0.5 * PARAMS.heading_deadband

    assert (correction_yaw_rate(offset, small, 1.0, PARAMS)
            > correction_yaw_rate(offset, 0.0, 1.0, PARAMS))


def test_the_deadband_has_no_step_at_its_edge():
    assert deadband(0.1, 0.1) == 0.0
    assert math.isclose(deadband(0.1001, 0.1), 0.0001)
    assert math.isclose(deadband(-0.3, 0.1), -0.2)


# ── limits ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('v', [0.0, 0.3, 1.0, 2.0, 5.0])
def test_the_correction_never_exceeds_the_yaw_rate_limit(v):
    limit = yaw_rate_limit(v, PARAMS)
    for offset in (-10.0, -1.0, 1.0, 10.0):
        for heading in (-1.5, 0.0, 1.5):
            assert abs(correction_yaw_rate(offset, heading, v, PARAMS)) <= limit + 1e-12


def test_the_limit_tightens_with_speed():
    # 1 m/s² of lateral acceleration is 0.5 rad/s at 2 m/s, 0.2 rad/s at 5.
    assert yaw_rate_limit(0.5, PARAMS) == PARAMS.max_yaw_rate
    assert math.isclose(yaw_rate_limit(2.0, PARAMS), 0.5)
    assert math.isclose(yaw_rate_limit(5.0, PARAMS), 0.2)


def test_the_approach_angle_is_capped():
    # With no heading error the whole correction is the approach angle, so a
    # huge offset must not ask for more than theta_max of it.
    p = SteeringParams(max_yaw_rate=10.0, max_lateral_accel=100.0)

    assert math.isclose(correction_yaw_rate(100.0, 0.0, 1.0, p), p.k_h * p.theta_max)


def test_reversing_does_not_divide_by_zero():
    assert math.isfinite(correction_yaw_rate(1.0, 0.0, -PARAMS.v_0, PARAMS))


# ── smoothstep ───────────────────────────────────────────────────────────────

def test_smoothstep_ramps_from_zero_to_one():
    assert smoothstep(0.3, 0.4, 0.8) == 0.0
    assert math.isclose(smoothstep(0.6, 0.4, 0.8), 0.5)
    assert smoothstep(0.9, 0.4, 0.8) == 1.0
