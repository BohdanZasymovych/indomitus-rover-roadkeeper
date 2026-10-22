"""Track geometry: where a pose is relative to the centre line.

The simulated camera is only as good as this projection. A sign slip turns
'road to the left' into 'road to the right' and the assist steers the rover
off the very road it is meant to keep it on.

No ROS import anywhere in here — track is deliberately standalone.
"""

import math

import pytest

from rover_road_assist_sim.track import Path, Pose, Track, build_track, parse_segment


ORIGIN = Pose(0.0, 0.0, 0.0)


def close(a, b, tol=1e-9):
    return math.isclose(a, b, abs_tol=tol)


# ── building ─────────────────────────────────────────────────────────────────

def test_segments_chain_end_to_start():
    path = Path('main', ORIGIN, ['straight 10', 'left 5 90', 'right 5 90'])
    for a, b in zip(path.pieces, path.pieces[1:]):
        assert close(a.end.x, b.start.x) and close(a.end.y, b.start.y)
        assert close(a.end.yaw, b.start.yaw)


def test_a_left_quarter_turn_ends_where_it_should():
    path = Path('main', ORIGIN, ['left 5 90'])
    end = path.pieces[-1].end

    assert close(end.x, 5.0) and close(end.y, 5.0) and close(end.yaw, math.pi / 2)
    assert close(path.length, 5.0 * math.pi / 2)


def test_a_right_turn_goes_right():
    end = Path('main', ORIGIN, ['right 5 90']).pieces[-1].end

    assert close(end.x, 5.0) and close(end.y, -5.0) and close(end.yaw, -math.pi / 2)


def test_scale_multiplies_lengths_and_radii():
    assert close(Path('main', ORIGIN, ['straight 10', 'left 5 90'], scale=2.0).length,
                 20.0 + 10.0 * math.pi / 2)


def test_bad_segments_are_rejected():
    for text in ('straight', 'left 5', 'up 5 90', 'straight ten'):
        with pytest.raises(ValueError):
            parse_segment(text)


def test_the_start_yaw_is_in_degrees():
    track = build_track([1.0, 2.0, 90.0], 1.0, ['straight 10'], -1, ['straight 1'])
    end = track.main.pieces[-1].end

    assert close(end.x, 1.0) and close(end.y, 12.0)


# ── projecting ───────────────────────────────────────────────────────────────

def test_left_of_a_straight_is_positive_lateral():
    p = Path('main', ORIGIN, ['straight 10']).project(4.0, 1.5)

    assert close(p.s, 4.0) and close(p.lateral, 1.5) and close(p.heading, 0.0)
    assert p.beside


@pytest.mark.parametrize('turn', ['left', 'right'])
@pytest.mark.parametrize('side', [1.0, -1.0])
def test_projection_onto_an_arc_recovers_the_offset(turn, side):
    path = Path('main', ORIGIN, [f'{turn} 10 120'])
    on = path.pose_at(7.0)
    x, y = on.left(side * 0.8)

    p = path.project(x, y)

    assert close(p.s, 7.0, 1e-6)
    assert close(p.lateral, side * 0.8, 1e-6)
    assert close(p.heading, on.yaw, 1e-9)
    assert close(p.curvature, (1.0 if turn == 'left' else -1.0) / 10.0)


def test_before_the_start_and_past_the_end_are_not_beside_the_road():
    path = Path('main', ORIGIN, ['straight 10'])

    assert not path.project(-1.0, 0.0).beside
    assert not path.project(11.0, 0.0).beside
    assert path.project(10.0, 0.5).beside


def test_the_closest_piece_wins():
    # Inside the bend of an L: closer to the second leg than the first.
    path = Path('main', ORIGIN, ['straight 10', 'left 0.001 90', 'straight 10'])
    p = path.project(9.0, 5.0)

    assert close(p.heading, math.pi / 2, 1e-3)
    assert close(p.lateral, 1.0, 1e-3)


def test_max_curvature_sees_the_tightest_piece_in_range():
    path = Path('main', ORIGIN, ['straight 10', 'right 5 90', 'left 20 90'])

    assert path.max_curvature(0.0, 5.0) == 0.0
    assert close(path.max_curvature(5.0, 12.0), 0.2)


# ── forks ────────────────────────────────────────────────────────────────────

FORK = Track.from_spec(ORIGIN, ['straight 10', 'straight 30'], 1, ['right 10 60', 'straight 20'])


def test_the_branch_starts_at_the_fork():
    assert close(FORK.branch.start.x, 10.0) and close(FORK.branch.start.y, 0.0)


def test_just_past_the_fork_both_roads_are_candidates():
    found = FORK.candidates(11.0, 0.0, 0.0, max_lateral=3.0, max_heading=1.0)

    assert {p.path for p in found} == {'main', 'branch'}


def test_before_the_fork_only_main_is():
    found = FORK.candidates(5.0, 0.0, 0.0, max_lateral=3.0, max_heading=1.0)

    assert [p.path for p in found] == ['main']


def test_a_road_pointing_the_other_way_is_not_a_candidate():
    assert FORK.candidates(5.0, 0.0, math.pi, max_lateral=3.0, max_heading=1.0) == []


def test_branch_from_minus_one_means_no_branch():
    assert Track.from_spec(ORIGIN, ['straight 10'], -1, ['straight 5']).branch is None


def test_a_branch_past_the_end_of_main_is_rejected():
    with pytest.raises(ValueError):
        Track.from_spec(ORIGIN, ['straight 10'], 3, ['straight 5'])
