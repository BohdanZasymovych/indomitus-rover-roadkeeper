"""The recorded path: spacing, length, and the points still to go.

No ROS import anywhere in here — trail is deliberately standalone.
"""

import math

from rover_link_guard.trail import Trail


def straight(n, spacing=0.25, max_points=1000):
    trail = Trail(spacing, max_points)
    for i in range(n):
        trail.add(i * spacing, 0.0)
    return trail


def test_points_closer_than_the_spacing_are_not_kept():
    trail = Trail(0.25, 100)
    trail.add(0.0, 0.0)
    trail.add(0.1, 0.0)
    trail.add(0.2, 0.0)
    assert len(trail) == 1
    trail.add(0.3, 0.0)
    assert len(trail) == 2


def test_length_is_the_path_length_not_the_distance_from_the_start():
    trail = Trail(0.5, 100)
    for x, y in [(0, 0), (1, 0), (1, 1), (0, 1)]:
        trail.add(x, y)
    assert math.isclose(trail.length, 3.0)


def test_empty_trail():
    trail = Trail(0.25, 100)
    assert trail.length == 0.0
    assert trail.target(0.0, 0.0, 1.0) is None


def test_pop_reached_drops_only_the_newest_points_near_the_rover():
    trail = straight(41)   # x = 0 … 10
    trail.pop_reached(10.0, 0.0, 0.6)
    assert math.isclose(trail.length, 9.25)   # 10.0, 9.75, 9.5 gone; 9.25 is 0.75 away


def test_target_is_the_newest_point_at_least_lookahead_away():
    trail = straight(41)
    target = trail.target(10.0, 0.0, 1.5)
    assert math.isclose(target.x, 8.5)


def test_target_is_the_oldest_point_when_all_are_closer_than_lookahead():
    trail = straight(5)   # x = 0 … 1
    target = trail.target(0.5, 0.0, 3.0)
    assert target.x == 0.0


def test_the_oldest_points_are_forgotten_past_max_points():
    trail = straight(100, max_points=10)
    assert len(trail) == 10
    assert math.isclose(trail.length, 99 * 0.25)
