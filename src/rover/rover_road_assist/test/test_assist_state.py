"""Gating, blending and smoothing of the road correction.

The rules that matter most here are the ones that keep the assist out of the
way: off means off, an operator who steers hard gets the rover to themselves,
and a stale or doubtful road estimate never pulls on the wheel.

No ROS import anywhere in here — assist_state is deliberately standalone.
"""

import math
import random

from rover_road_assist.assist_state import (
    AssistParams,
    ConfidenceGate,
    Mode,
    RoadAssist,
    RoadEstimate,
    Status,
)
from rover_road_assist.steering_law import SteeringParams


STEERING = SteeringParams()
PARAMS = AssistParams()
DT = 0.05   # a 20 Hz operator


def road(offset=0.5, heading=0.0, confidence=1.0, valid=True, stamp=0.0):
    return RoadEstimate(offset, heading, confidence, valid, stamp)


def drive(assist, t0, duration, v=1.0, wz=0.0, mode=Mode.ASSIST, offset=0.5,
          confidence=1.0, valid=True):
    """Fresh road estimates and operator commands every DT; returns the time
    after the last step and the yaw rates published."""
    out = []
    t = t0
    for _ in range(round(duration / DT)):
        assist.on_road(road(offset, 0.0, confidence, valid, stamp=t), t)
        out.append(assist.on_command(v, wz, t, mode))
        t += DT
    return t, out


def engaged():
    """An assist that has been at full strength for a while."""
    assist = RoadAssist(STEERING, PARAMS)
    t, _ = drive(assist, 0.0, 2.0)
    assert math.isclose(assist.alpha, PARAMS.alpha_max)
    return assist, t


# ── pass-through ─────────────────────────────────────────────────────────────

def test_off_and_warn_pass_the_command_through_untouched():
    for mode in (Mode.OFF, Mode.WARN):
        assist = RoadAssist(STEERING, PARAMS)
        _, out = drive(assist, 0.0, 2.0, wz=0.123, mode=mode)

        assert all(wz == 0.123 for wz in out)
        assert assist.alpha == 0.0


def test_zero_alpha_publishes_exactly_the_operators_command():
    # Below min_speed the target strength is zero from the start.
    assist = RoadAssist(STEERING, PARAMS)
    _, out = drive(assist, 0.0, 2.0, v=PARAMS.min_speed, wz=0.2)

    assert all(wz == 0.2 for wz in out)


def test_no_road_estimate_at_all_is_a_pass_through():
    assist = RoadAssist(STEERING, PARAMS)

    assert assist.on_command(1.0, 0.2, 0.0, Mode.ASSIST) == 0.2
    assert assist.on_command(1.0, 0.2, DT, Mode.ASSIST) == 0.2
    assert assist.status(DT, Mode.ASSIST) is Status.STALE


def test_switching_off_lets_go_at_once():
    assist, t = engaged()

    assert assist.on_command(1.0, 0.1, t, Mode.OFF) == 0.1
    assert assist.alpha == 0.0


# ── engaging ─────────────────────────────────────────────────────────────────

def test_a_road_to_the_left_pulls_the_operators_command_left():
    assist, t = engaged()

    assert assist.on_command(1.0, 0.0, t, Mode.ASSIST) > 0.0


def test_alpha_ramps_rather_than_steps():
    assist = RoadAssist(STEERING, PARAMS)
    drive(assist, 0.0, 1.0)
    alphas = []
    t = 1.0
    for _ in range(20):
        alphas.append(assist.alpha)
        t, _ = drive(assist, t, DT)

    steps = [b - a for a, b in zip(alphas, alphas[1:])]
    assert all(abs(s) <= PARAMS.alpha_rate * DT + 1e-12 for s in steps)


def test_a_stale_road_estimate_fades_the_assist_out():
    assist, t = engaged()
    for _ in range(20):   # road/state stops; the operator keeps driving
        assist.on_command(1.0, 0.0, t, Mode.ASSIST)
        t += DT

    assert assist.alpha == 0.0
    assert assist.status(t, Mode.ASSIST) is Status.STALE


def test_an_invalid_road_fades_the_assist_out():
    assist, t = engaged()
    t, _ = drive(assist, t, 1.0, valid=False)

    assert assist.alpha == 0.0
    assert assist.status(t, Mode.ASSIST) is Status.STANDBY_LOW_CONF


def test_a_pause_longer_than_the_twist_mux_timeout_starts_over():
    assist, t = engaged()

    assist.on_road(road(stamp=t + 1.0), t + 1.0)
    assert assist.on_command(1.0, 0.2, t + 1.0, Mode.ASSIST) == 0.2
    assert assist.alpha == 0.0


# ── override ─────────────────────────────────────────────────────────────────

def test_hard_steering_keeps_the_assist_out_for_override_hold():
    assist, t = engaged()
    start = t
    t, _ = drive(assist, t, DT, wz=PARAMS.override_yaw_rate + 0.1)

    fade = PARAMS.alpha_max / PARAMS.alpha_rate
    while t < start + PARAMS.override_hold:
        if t >= start + fade + DT:
            assert assist.alpha == 0.0
            assert assist.status(t, Mode.ASSIST) is Status.OVERRIDE
        t, _ = drive(assist, t, DT)

    t, _ = drive(assist, t, 1.0)
    assert math.isclose(assist.alpha, PARAMS.alpha_max)
    assert assist.status(t, Mode.ASSIST) is Status.ASSIST_ACTIVE


def test_gentle_steering_does_not_count_as_an_override():
    assist, t = engaged()
    t, _ = drive(assist, t, 1.0, wz=PARAMS.override_yaw_rate - 0.05)

    assert math.isclose(assist.alpha, PARAMS.alpha_max)


# ── confidence ───────────────────────────────────────────────────────────────

def test_the_confidence_gate_opens_only_after_holding_above_on():
    gate = ConfidenceGate(on=0.6, off=0.4, on_time=0.3)

    assert not gate.update(0.7, 0.0)
    assert not gate.update(0.7, 0.2)
    assert gate.update(0.7, 0.3)


def test_dipping_into_the_band_restarts_the_hold():
    gate = ConfidenceGate(on=0.6, off=0.4, on_time=0.3)
    gate.update(0.7, 0.0)
    gate.update(0.5, 0.2)

    assert not gate.update(0.7, 0.4)
    assert gate.update(0.7, 0.8)


def test_an_open_gate_stays_open_in_the_band_and_closes_below_off():
    gate = ConfidenceGate(on=0.6, off=0.4, on_time=0.0)
    gate.update(0.7, 0.0)

    assert gate.update(0.5, 1.0)
    assert not gate.update(0.39, 2.0)


def test_low_confidence_keeps_the_assist_on_standby():
    assist = RoadAssist(STEERING, PARAMS)
    t, out = drive(assist, 0.0, 2.0, confidence=0.5)

    assert all(wz == 0.0 for wz in out)
    assert assist.status(t, Mode.ASSIST) is Status.STANDBY_LOW_CONF


# ── limits ───────────────────────────────────────────────────────────────────

def test_the_limits_hold_under_arbitrary_input():
    rng = random.Random(0)
    assist = RoadAssist(STEERING, PARAMS)
    added = 0.0
    t = 0.0
    assist.on_command(1.0, 0.0, t, Mode.ASSIST)
    for _ in range(5000):
        dt = rng.uniform(0.01, 0.1)
        t += dt
        # Confidence mostly high, so the assist spends time engaged.
        assist.on_road(road(rng.uniform(-3.0, 3.0), rng.uniform(-1.0, 1.0),
                            rng.uniform(0.3, 1.0), rng.random() > 0.05, stamp=t), t)
        v, wz = rng.uniform(-1.0, 3.0), rng.uniform(-0.6, 0.6)
        alpha = assist.alpha

        new_added = assist.on_command(v, wz, t, Mode.ASSIST) - wz

        assert 0.0 <= assist.alpha <= PARAMS.alpha_max
        assert abs(assist.alpha - alpha) <= PARAMS.alpha_rate * dt + 1e-12
        assert abs(new_added - added) <= PARAMS.max_yaw_accel * dt + 1e-12
        # Never more than alpha_max of the way from the operator to the
        # largest correction there is.
        assert abs(new_added) <= PARAMS.alpha_max * (abs(wz) + STEERING.max_yaw_rate) + 1e-12
        added = new_added


# ── closed loop ──────────────────────────────────────────────────────────────

def simulate(steering, e0=0.6, v=1.0, duration=30.0):
    """The rover on a straight road, hands off: de/dt = v·sin(psi), dpsi/dt = -wz.
    Returns (t, offset) samples."""
    assist = RoadAssist(steering, PARAMS)
    e, psi = e0, 0.0
    samples = []
    for i in range(round(duration / DT)):
        t = i * DT
        assist.on_road(road(e, psi, stamp=t), t)
        wz = assist.on_command(v, 0.0, t, Mode.ASSIST)
        samples.append((t, e))
        e += v * math.sin(psi) * DT
        psi -= wz * DT
    return samples


def test_the_closed_loop_centres_the_rover():
    # Deadbands off, so the law is free to centre the rover completely.
    # The bound is 10.5 s rather than 10: with the default gains the offset
    # crosses 0.05 m at about 10.15 s, ~0.55 s of which is the confidence
    # hold and alpha ramp before the correction is at full strength.
    samples = simulate(SteeringParams(offset_deadband=0.0, heading_deadband=0.0))

    first = next(t for t, e in samples if abs(e) < 0.05)
    assert first <= 10.5
    assert all(abs(e) < 0.05 for t, e in samples if t >= first)


def test_with_the_default_deadbands_the_rover_settles_without_weaving():
    # It stops at the edge of the offset deadband — the point of a deadband —
    # and must stay there rather than drift from edge to edge.
    edge = STEERING.offset_deadband * STEERING.road_width / 2.0
    samples = simulate(STEERING, duration=60.0)

    assert all(abs(e) <= edge + 0.01 for t, e in samples if t >= 20.0)
    assert all(e >= 0.0 for _, e in samples)   # no overshoot to the far side
