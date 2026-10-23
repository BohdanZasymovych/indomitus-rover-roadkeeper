"""What the rover does on its own when the link drops.

A small unicycle simulation closes the loop: the fallback's commands move the
pose that is fed back to it, so a retreat has to actually follow the trail.
What matters most is that every way out ends in a stop, and that a restored
link ends the fallback at once.

No ROS import anywhere in here — fallback is deliberately standalone.
"""

import math
from dataclasses import replace

from rover_road_assist.assist_state import RoadEstimate
from rover_road_assist.steering_law import SteeringParams

from rover_link_guard.fallback import (
    Fallback, FallbackParams, Phase, Pose, RetreatMode, wrap_angle)

STEERING = SteeringParams()
PARAMS = FallbackParams()
DT = 0.05
CRUISE = 0.5   # [m/s] the operator's speed in the lead-up


class Sim:
    """The fallback driving a unicycle, one DT per tick."""

    def __init__(self, fallback, x=0.0, y=0.0, yaw=0.0, t=0.0):
        self.fb = fallback
        self.x, self.y, self.yaw, self.t = x, y, yaw, t
        self.v = 0.0
        self.road = True   # whether a trusted road estimate is published
        self.odometry = True

    def tick(self):
        if self.odometry:
            self.fb.on_pose(Pose(self.x, self.y, self.yaw, self.t))
        if self.road:
            self.fb.on_road(RoadEstimate(0.0, 0.0, 1.0, True, self.t))
        cmd = self.fb.step(self.t)
        if cmd is not None:
            self.v = cmd.v
            self.x += cmd.v * math.cos(self.yaw) * DT
            self.y += cmd.v * math.sin(self.yaw) * DT
            self.yaw += cmd.wz * DT
        self.t += DT
        return cmd

    def run(self, seconds):
        for _ in range(round(seconds / DT)):
            self.tick()

    def run_until(self, phase, limit=200.0):
        for _ in range(round(limit / DT)):
            self.tick()
            if self.fb.phase is phase:
                return
        raise AssertionError(f'never reached {phase}, stuck in {self.fb.phase}')

    def drive_straight(self, distance, speed=CRUISE):
        """The operator driving forward with the link up."""
        for _ in range(round(distance / speed / DT)):
            self.x += speed * math.cos(self.yaw) * DT
            self.y += speed * math.sin(self.yaw) * DT
            self.tick()


def fallback(retreat=RetreatMode.REVERSE, **overrides):
    return Fallback(STEERING, replace(PARAMS, **overrides), retreat)


def lost_after_driving(retreat=RetreatMode.REVERSE, distance=10.0, **overrides):
    """A rover 10 m down a straight road when the link drops."""
    sim = Sim(fallback(retreat, **overrides))
    sim.drive_straight(distance)
    sim.fb.start(sim.t, CRUISE)
    return sim


# ── link up ──────────────────────────────────────────────────────────────────

def test_idle_publishes_nothing_but_records_the_trail():
    sim = Sim(fallback())
    sim.drive_straight(5.0)
    assert sim.fb.phase is Phase.IDLE
    assert sim.tick() is None
    assert sim.fb._trail.length > 4.0


# ── when it does not move ────────────────────────────────────────────────────

def test_a_rover_that_was_standing_still_stays_put():
    sim = Sim(fallback())
    sim.drive_straight(2.0)
    sim.fb.start(sim.t, 0.0)
    assert sim.fb.phase is Phase.HOLD
    x = sim.x
    sim.run(5.0)
    assert sim.x == x


def test_no_odometry_means_hold():
    fb = fallback()
    fb.start(0.0, CRUISE)
    assert fb.phase is Phase.HOLD
    cmd = fb.step(0.0)
    for i in range(1, 41):
        cmd = fb.step(i * DT)
    assert cmd == (0.0, 0.0)


# ── CONTINUE ─────────────────────────────────────────────────────────────────

def test_it_continues_along_the_road_at_the_capped_speed():
    sim = lost_after_driving(continue_speed=0.3)
    sim.run(2.0)
    assert sim.fb.phase is Phase.CONTINUE
    assert math.isclose(sim.v, 0.3, abs_tol=1e-6)
    assert abs(sim.yaw) < 0.01   # road dead ahead: no steering


def test_it_never_continues_faster_than_the_operator_was_going():
    sim = Sim(fallback(continue_speed=2.0))
    sim.drive_straight(3.0, speed=0.4)
    sim.fb.start(sim.t, 0.4)
    sim.run(3.0)
    assert math.isclose(sim.v, 0.4, abs_tol=1e-6)


def test_it_steers_toward_the_road_while_continuing():
    sim = lost_after_driving()
    sim.fb.on_road(RoadEstimate(offset=1.0, heading=0.0, confidence=1.0, valid=True,
                                stamp=sim.t))
    cmd = sim.fb.step(sim.t)
    assert cmd.wz > 0.0   # road to the left, steer left


def test_continue_ends_at_the_distance_budget_and_retreats():
    sim = lost_after_driving(continue_distance=3.0)
    sim.run_until(Phase.RETREAT)
    assert 2.9 < sim.x - 10.0 < 3.5


def test_continue_ends_at_the_time_budget():
    sim = lost_after_driving(continue_time=2.0, continue_distance=100.0)
    sim.run_until(Phase.RETREAT)
    assert 2.0 <= sim.t - sim.fb._started < 2.2


def test_a_road_lost_for_longer_than_the_grace_ends_continue():
    sim = lost_after_driving()
    sim.road = False
    sim.run(PARAMS.road_lost_grace - 0.5)
    assert sim.fb.phase is Phase.CONTINUE
    sim.run(1.0)
    assert sim.fb.phase is Phase.RETREAT
    assert sim.fb.reason == 'road lost'


def test_a_doubtful_road_does_not_steer():
    sim = lost_after_driving()
    sim.road = False
    sim.fb.on_road(RoadEstimate(offset=2.0, heading=0.5, confidence=0.2, valid=True,
                                stamp=sim.t))
    assert sim.fb.step(sim.t).wz == 0.0


# ── RETREAT ──────────────────────────────────────────────────────────────────

def test_it_reverses_to_the_retreat_point_and_holds():
    sim = lost_after_driving(continue_distance=4.0, retreat_distance=5.0)
    sim.run_until(Phase.HOLD)
    sim.run(2.0)
    # Lost at x = 10; 5 m before it is x = 5. It is 0.5 m short of the last
    # trail point it was within reach of.
    assert 4.5 < sim.x < 6.2
    assert abs(wrap_angle(sim.yaw)) < 0.05   # still facing the way it came
    assert sim.v == 0.0
    assert sim.fb.reason == 'back at the retreat point'


def test_it_brakes_before_reversing_never_straight_through():
    sim = lost_after_driving(continue_time=0.0)
    speeds = []
    for _ in range(80):
        sim.tick()
        speeds.append(sim.v)
    i = next(i for i, v in enumerate(speeds) if v < 0.0)
    assert all(v >= 0.0 for v in speeds[:i])
    # A fall to zero within max_decel, then a rise in reverse within max_accel.
    assert all(abs(b - a) <= PARAMS.max_decel * DT + 1e-9 for a, b in zip(speeds, speeds[1:]))


def test_reversing_follows_a_curved_trail():
    radius = 10.0
    sim = Sim(fallback(continue_time=0.0, retreat_distance=6.0))
    theta = 0.0
    while theta < 0.8:
        theta += CRUISE * DT / radius
        sim.x, sim.y, sim.yaw = radius * math.sin(theta), radius * (1 - math.cos(theta)), theta
        sim.tick()
    sim.fb.start(sim.t, CRUISE)

    worst = 0.0
    for _ in range(round(60.0 / DT)):
        sim.tick()
        worst = max(worst, abs(math.hypot(sim.x, sim.y - radius) - radius))
        if sim.fb.phase is Phase.HOLD:
            break
    assert sim.fb.phase is Phase.HOLD
    assert worst < 0.4
    travelled_back = math.atan2(sim.x, radius - sim.y)
    assert 0.8 - 6.0 / radius - 0.1 < travelled_back < 0.8 - 6.0 / radius + 0.1


def test_turn_around_faces_the_trail_and_drives_it_forward():
    sim = lost_after_driving(RetreatMode.TURN_AROUND, continue_distance=3.0)
    sim.run_until(Phase.TURN)
    sim.run_until(Phase.RETREAT)
    assert abs(wrap_angle(sim.yaw - math.pi)) < PARAMS.turn_tolerance + 0.05
    sim.run_until(Phase.HOLD)
    assert 4.5 < sim.x < 6.2
    assert sim.fb.reason == 'back at the retreat point'


def test_the_turn_waits_for_the_rover_to_stop():
    sim = lost_after_driving(RetreatMode.TURN_AROUND, continue_time=0.0)
    cmds = [sim.tick() for _ in range(40)]
    assert all(c.wz == 0.0 for c in cmds if abs(c.v) > 0.05)


def test_a_trail_the_rover_has_left_ends_in_hold():
    sim = lost_after_driving(continue_time=0.0)
    sim.run(1.0)
    sim.y += 5.0   # thrown sideways off the trail
    sim.run_until(Phase.HOLD)
    assert sim.fb.reason == 'off the trail'


# ── limits ───────────────────────────────────────────────────────────────────

def test_the_time_budget_holds_from_any_phase():
    sim = lost_after_driving(max_duration=3.0, continue_time=100.0, continue_distance=100.0)
    sim.run_until(Phase.HOLD)
    assert sim.fb.reason == 'time budget spent'
    assert 3.0 <= sim.t - sim.fb._started < 3.2
    sim.run(2.0)
    assert sim.v == 0.0


def test_stale_odometry_stops_the_rover_without_giving_up():
    sim = lost_after_driving()
    sim.run(1.0)
    sim.odometry = False
    sim.run(PARAMS.pose_max_age + 1.5)
    assert sim.fb.phase is Phase.CONTINUE
    assert sim.v == 0.0
    sim.odometry = True
    sim.run(2.0)
    assert sim.v > 0.0


def test_speeds_and_yaw_rates_stay_within_the_limits():
    sim = lost_after_driving(RetreatMode.TURN_AROUND, continue_distance=3.0)
    for _ in range(round(120.0 / DT)):
        cmd = sim.tick()
        assert abs(cmd.v) <= max(PARAMS.continue_speed, PARAMS.retreat_speed) + 1e-9
        assert abs(cmd.wz) <= PARAMS.max_yaw_rate + 1e-9


# ── handing back ─────────────────────────────────────────────────────────────

def test_a_restored_link_ends_the_fallback_at_once():
    sim = lost_after_driving()
    sim.run(2.0)
    sim.fb.stop()
    assert sim.fb.phase is Phase.IDLE
    assert sim.fb.step(sim.t) is None


def test_a_second_outage_starts_clean():
    sim = lost_after_driving(max_duration=3.0)
    sim.run_until(Phase.HOLD)
    sim.fb.stop()
    sim.drive_straight(3.0)
    sim.fb.start(sim.t, CRUISE)
    assert sim.fb.phase is Phase.CONTINUE
    sim.run(1.0)
    assert sim.v > 0.0
