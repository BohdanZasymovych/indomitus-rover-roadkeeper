"""The road as Gazebo draws it: well-formed SDF, from the same track.

No ROS import anywhere in here — road_sdf is deliberately standalone.
"""

import xml.etree.ElementTree as ET

from rover_road_assist_sim.road_sdf import road_sdf
from rover_road_assist_sim.track import Pose, Track


TRACK = Track.from_spec(Pose(0.0, 0.0, 0.0), ['straight 20', 'left 8 90', 'straight 10'],
                        1, ['right 8 45', 'straight 10'])


def parse():
    return ET.fromstring(road_sdf(TRACK, road_width=3.0))


def test_it_is_one_static_model_of_visuals_only():
    model = parse().find('model')

    assert model.find('static').text == 'true'
    assert model.find('link/collision') is None
    assert len(model.findall('link/visual')) > 0


def test_visual_names_are_unique():
    names = [v.get('name') for v in parse().iter('visual')]

    assert len(names) == len(set(names))


def test_both_paths_are_drawn():
    names = [v.get('name') for v in parse().iter('visual')]

    assert any(n.startswith('main_surface') for n in names)
    assert any(n.startswith('branch_surface') for n in names)
    assert any(n.startswith('main_centre') for n in names)


def test_the_road_is_as_wide_as_configured():
    surface = next(v for v in parse().iter('visual') if v.get('name') == 'main_surface_0')
    _, width, _ = map(float, surface.find('geometry/box/size').text.split())

    assert width == 3.0
