import math

import numpy as np
import pytest

from rover_drone_detection import geometry
from rover_drone_detection.geometry import Intrinsics


def test_intrinsics_from_k():
    k = [500.0, 0, 320, 0, 510.0, 180, 0, 0, 1]
    assert Intrinsics.from_camera_info(k, 640, 360, 90) == Intrinsics(500, 510, 320, 180)


def test_intrinsics_fallback_hfov():
    intr = Intrinsics.from_camera_info([0.0] * 9, 640, 360, 90)
    assert intr.fx == pytest.approx(320)
    assert (intr.cx, intr.cy) == (320, 180)


def test_bearing_sign_left_positive():
    intr = Intrinsics(320, 320, 320, 180)
    assert geometry.bearing(320, intr) == pytest.approx(0)
    assert geometry.bearing(0, intr) == pytest.approx(math.pi / 4)      # left edge
    assert geometry.bearing(640, intr) == pytest.approx(-math.pi / 4)


def test_depth_from_size():
    intr = Intrinsics(500, 500, 320, 180)
    assert geometry.depth_from_size(50, intr, 0.3) == pytest.approx(3.0)
    assert math.isnan(geometry.depth_from_size(0, intr, 0.3))


def test_camera_point_to_world():
    # Optical frame of a camera at (1, 2, 0.5) looking along world +x:
    # optical z -> world x, optical x -> world -y, optical y -> world -z.
    q = (-0.5, 0.5, -0.5, 0.5)
    intr = Intrinsics(500, 500, 320, 180)
    p_cam = geometry.point_in_camera(320, 180, 4.0, intr)
    np.testing.assert_allclose(geometry.transform_point((1, 2, 0.5), q, p_cam), [5, 2, 0.5],
                               atol=1e-9)
    # a pixel to the right ends up at smaller world y
    right = geometry.transform_point((1, 2, 0.5), q, geometry.point_in_camera(570, 180, 4.0, intr))
    np.testing.assert_allclose(right, [5, 0, 0.5], atol=1e-9)


def test_yaw_of():
    half = math.pi / 4
    assert geometry.yaw_of((0, 0, math.sin(half), math.cos(half))) == pytest.approx(math.pi / 2)
