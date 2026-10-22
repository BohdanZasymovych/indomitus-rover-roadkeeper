"""Where a detected drone is: bearing, a rough range, and a point in the world.

Camera optical frame (REP-103 _optical): x right, y down, z forward. Bearing
is in the usual ROS sense instead, + = left of the camera axis.

The range is from the apparent size, depth ~= fx * drone_size / box_width,
so it is only as good as `drone_size_m` and the calibration: fine for "which
side of the road, roughly how far", not for anything precise.

No ROS import anywhere in here; transforms come in as plain tuples.
"""

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_camera_info(cls, k, width, height, hfov_deg):
        """K of a CameraInfo (row-major 3x3), or a pinhole of `hfov_deg` when
        K is all zeros (rover_cameras without a calibration file)."""
        if k is not None and len(k) == 9 and k[0] > 0 and k[4] > 0:
            return cls(float(k[0]), float(k[4]), float(k[2]), float(k[5]))
        f = 0.5 * width / math.tan(math.radians(hfov_deg) / 2)
        return cls(f, f, 0.5 * width, 0.5 * height)


def bearing(u, intr: Intrinsics):
    """[rad] of image column `u`, + = left."""
    return -math.atan2(u - intr.cx, intr.fx)


def depth_from_size(box_width_px, intr: Intrinsics, size_m):
    """[m] along the camera axis, NaN for a degenerate box."""
    if box_width_px <= 1.0 or size_m <= 0:
        return math.nan
    return intr.fx * size_m / box_width_px


def point_in_camera(u, v, depth, intr: Intrinsics):
    """The pixel back-projected to `depth` [m], in the optical frame."""
    return np.array([(u - intr.cx) / intr.fx * depth,
                     (v - intr.cy) / intr.fy * depth,
                     depth])


def quat_rotate(q, v):
    """Rotate vector v by quaternion q = (x, y, z, w)."""
    x, y, z, w = q
    u = np.array([x, y, z])
    v = np.asarray(v, dtype=float)
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def yaw_of(q):
    x, y, z, w = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def transform_point(translation, rotation, p):
    """p from the child frame into the parent, given the child's pose in it."""
    return np.asarray(translation, dtype=float) + quat_rotate(rotation, p)
