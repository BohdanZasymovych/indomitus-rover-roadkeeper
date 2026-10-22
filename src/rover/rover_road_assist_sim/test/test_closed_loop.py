"""The default track, driven hands-off by the real assist through the fake camera.

The one place both sign conventions meet: the camera's (+ = right, in
half-widths) and the assist's (+ = left, in metres). A slip on either side
sends the rover off the first bend. A unicycle stands in for Gazebo.

Hands-off is not how the assist is meant to be used — alpha_max = 0.5 gives
it only half the turn — so it is only expected to manage at a gentle speed.

No ROS import anywhere in here; rover_road_assist is a test dependency.
"""

import math
import os

import pytest
import yaml

from rover_road_assist_sim.perception_model import (
    CameraParams, ConfidenceParams, NoiseParams, RoadCamera,
)
from rover_road_assist_sim.track import Pose, build_track

assist_state = pytest.importorskip('rover_road_assist.assist_state')
steering_law = pytest.importorskip('rover_road_assist.steering_law')

CONFIG = os.path.join(os.path.dirname(__file__), '..', 'config', 'road_sim.yaml')
DT = 0.05


def default_track():
    with open(CONFIG) as f:
        t = yaml.safe_load(f)['/**/road_sim_node']['ros__parameters']['track']
    return build_track(t['start'], t['scale'], t['main'], t['branch']['from'],
                       t['branch']['segments'])


def drive_hands_off(confidence_mode: str, v: float):
    """Returns (furthest s reached along main, worst |offset| in half-widths)."""
    track = default_track()
    cam = RoadCamera(track, CameraParams(road_width=3.0), NoiseParams(),
                     ConfidenceParams(mode=confidence_mode))
    assist = assist_state.RoadAssist(steering_law.SteeringParams(nominal_road_width=3.0),
                                     assist_state.AssistParams())
    x, y, yaw, t = 0.0, 0.3, 0.0, 0.0
    furthest, worst = 0.0, 0.0
    while t < 2.0 * track.main.length / v:
        seen = cam.observe(Pose(x, y, yaw))
        if not seen.valid:
            break
        offset, heading = steering_law.road_error(seen.lateral_offset, seen.heading_error, 3.0)
        assist.on_road(assist_state.RoadEstimate(offset, heading, seen.confidence, True, t), t)
        wz = assist.on_command(v, 0.0, t, assist_state.Mode.ASSIST)
        x += v * math.cos(yaw) * DT
        y += v * math.sin(yaw) * DT
        yaw += wz * DT
        t += DT
        worst = max(worst, abs(seen.lateral_offset))
        if seen.path == 'main':
            furthest = max(furthest, track.main.project(x, y).s)
    return furthest, worst, track.main.length


@pytest.mark.parametrize('mode', ['fixed', 'model'])
def test_the_assist_keeps_the_rover_on_the_whole_track_at_walking_pace(mode):
    furthest, worst, length = drive_hands_off(mode, v=0.5)

    assert furthest > length - 1.0
    assert worst < 1.0   # never past the road edge


def test_with_full_confidence_it_manages_at_1_m_s():
    furthest, worst, length = drive_hands_off('fixed', v=1.0)

    assert furthest > length - 1.0
    assert worst < 1.0
