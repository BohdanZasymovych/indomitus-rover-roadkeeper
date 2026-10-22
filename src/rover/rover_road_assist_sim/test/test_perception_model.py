"""The simulated camera: RoadDeviation's convention, confidence, latency.

The real camera publishes + = right and an offset in half-widths; the assist
flips both. If the simulation got the convention wrong in the same way the
assist did, the two bugs would cancel and every test would pass — so the
convention is pinned here against the message definition itself.

No ROS import anywhere in here — perception_model is deliberately standalone.
"""

import math

from rover_road_assist_sim.perception_model import (
    CameraParams, ConfidenceParams, DelayLine, NoiseParams, RoadCamera, model_confidence,
)
from rover_road_assist_sim.track import Pose, Track


STRAIGHT = Track.from_spec(Pose(-10.0, 0.0, 0.0), ['straight 100'])
FORK = Track.from_spec(Pose(-10.0, 0.0, 0.0), ['straight 10', 'straight 50'], 1,
                       ['right 12 40', 'straight 30'])
CAMERA = CameraParams(road_width=3.0)
NO_NOISE = NoiseParams(enabled=False)
FIXED = ConfidenceParams(mode='fixed', fixed_value=0.9)
MODEL = ConfidenceParams(mode='model')


def camera(track=STRAIGHT, confidence=FIXED, noise=NO_NOISE, params=CAMERA):
    return RoadCamera(track, params, noise, confidence)


# ── RoadDeviation's convention ───────────────────────────────────────────────

def test_rover_left_of_the_road_sees_the_centre_to_its_right():
    # "+ = road centre is to the right of the vehicle"
    seen = camera().observe(Pose(10.0, 0.75, 0.0))

    assert seen.valid
    assert math.isclose(seen.lateral_offset, 0.5)   # 0.75 m of a 1.5 m half-width


def test_rover_pointing_left_of_the_road_sees_it_bend_right():
    # "+ = road bends to the right"
    seen = camera().observe(Pose(10.0, 0.0, 0.2))

    assert math.isclose(seen.heading_error, 0.2)


def test_lookahead_measures_the_road_ahead():
    track = Track.from_spec(Pose(-10.0, 0.0, 0.0), ['straight 10', 'left 10 90'])
    near = camera(track).observe(Pose(-1.0, 0.0, 0.0))
    ahead = camera(track, params=CameraParams(lookahead=5.0)).observe(Pose(-1.0, 0.0, 0.0))

    assert math.isclose(near.heading_error, 0.0, abs_tol=1e-9)
    assert ahead.heading_error < 0.0   # the road is starting to bend left


# ── in view or not ───────────────────────────────────────────────────────────

def test_far_off_the_road_nothing_is_in_view():
    seen = camera().observe(Pose(10.0, 10.0, 0.0))

    assert not seen.valid and seen.confidence == 0.0


def test_facing_away_from_the_road_nothing_is_in_view():
    assert not camera().observe(Pose(10.0, 0.0, math.pi / 2)).valid


# ── confidence ───────────────────────────────────────────────────────────────

def test_fixed_confidence_is_fixed():
    cam = camera(noise=NoiseParams(enabled=True))
    for y in (0.0, 1.2, -2.0):
        assert cam.observe(Pose(10.0, y, 0.0)).confidence == 0.9


def test_model_confidence_is_full_centred_on_a_straight():
    assert camera(confidence=MODEL).observe(Pose(10.0, 0.0, 0.0)).confidence == 1.0


def test_model_confidence_drops_toward_the_edge_and_past_it():
    cam = camera(confidence=MODEL)
    near_edge = cam.observe(Pose(10.0, 1.5, 0.0)).confidence
    past_edge = cam.observe(Pose(10.0, 2.0, 0.0)).confidence

    assert 0.0 < near_edge < 1.0
    assert past_edge < near_edge


def test_model_confidence_drops_ahead_of_a_tight_curve():
    p = ConfidenceParams(mode='model')

    assert model_confidence(0.0, 0.0, 1.0 / 20.0, False, p) == 1.0
    assert model_confidence(0.0, 0.0, 1.0 / 5.0, False, p) < 0.5


def test_each_factor_can_be_switched_off():
    p = ConfidenceParams(mode='model', lateral=False, heading=False, curvature=False)

    assert model_confidence(1.5, 0.6, 0.3, False, p) == 1.0


def test_the_fork_lowers_confidence_while_both_roads_are_in_view():
    cam = camera(FORK, MODEL)

    assert cam.observe(Pose(-5.0, 0.0, 0.0)).confidence == 1.0
    assert math.isclose(cam.observe(Pose(1.0, 0.0, 0.0)).confidence, MODEL.fork_confidence)
    assert cam.observe(Pose(30.0, 0.0, 0.0)).confidence == 1.0


# ── following one road through the fork ──────────────────────────────────────

def test_the_camera_follows_the_road_the_rover_takes():
    cam = camera(FORK)
    branch = FORK.branch
    path = [cam.observe(branch.pose_at(s)).path for s in (0.5, 3.0, 8.0, 20.0)]

    assert path[-1] == 'branch'
    # Once on the branch it stays there rather than flicking back to main.
    first = path.index('branch')
    assert all(p == 'branch' for p in path[first:])


def test_staying_on_main_keeps_main():
    cam = camera(FORK)

    assert all(cam.observe(Pose(x, 0.0, 0.0)).path == 'main' for x in (-5.0, 1.0, 5.0, 30.0))


# ── noise ────────────────────────────────────────────────────────────────────

def test_noise_is_repeatable_for_a_seed_and_absent_when_off():
    pose = Pose(10.0, 0.3, 0.05)
    a = camera(noise=NoiseParams(enabled=True, seed=7)).observe(pose)
    b = camera(noise=NoiseParams(enabled=True, seed=7)).observe(pose)
    clean = camera().observe(pose)

    assert a == b
    assert a.lateral_offset != clean.lateral_offset
    assert math.isclose(clean.lateral_offset, 0.2)


# ── latency ──────────────────────────────────────────────────────────────────

def test_the_delay_line_returns_the_sample_from_delay_ago():
    line = DelayLine(0.1)
    for i in range(10):
        line.push(i * 0.02, i)

    assert line.at(0.19) == (0.08, 4)
    assert line.at(0.05) is None


def test_zero_delay_returns_the_newest():
    line = DelayLine(0.0)
    line.push(1.0, 'a')
    line.push(1.1, 'b')

    assert line.at(1.1) == (1.1, 'b')


def test_a_sim_reset_starts_the_delay_line_over():
    line = DelayLine(0.0)
    line.push(50.0, 'old')
    line.push(0.5, 'new')

    assert line.at(0.5) == (0.5, 'new')
