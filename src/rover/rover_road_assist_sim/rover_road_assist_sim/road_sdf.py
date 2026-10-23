#!/usr/bin/env python3
"""The track as a Gazebo model: road surface, edge lines, dashed centre line.

Visual only — no collision — and drawn from the same Track that road_sim_node
measures against, so what Gazebo shows is exactly what the assist steers to.
Arcs are approximated by short straight boxes.

No ROS import anywhere in here — road_sdf is deliberately standalone.
"""

import math

from rover_road_assist_sim.track import Path, Track

#: [m] Longest box used along an arc.
ARC_STEP = 0.5
LINE_WIDTH = 0.12
DASH_LENGTH = 1.5
DASH_GAP = 1.5

# (rgb, height above ground [m]) — stacked so the layers never z-fight.
SURFACE = ((0.22, 0.22, 0.24), 0.004)
EDGE = ((0.95, 0.95, 0.95), 0.008)
CENTRE = ((0.95, 0.75, 0.10), 0.008)
THICKNESS = 0.004


def _box(name: str, x0, y0, x1, y1, width: float, style, extend: float = 0.0) -> str:
    (r, g, b), z = style
    length = math.hypot(x1 - x0, y1 - y0) + extend
    if length <= 1e-6:
        return ''
    yaw = math.atan2(y1 - y0, x1 - x0)
    return f"""
      <visual name="{name}">
        <pose>{(x0 + x1) / 2:.4f} {(y0 + y1) / 2:.4f} {z:.4f} 0 0 {yaw:.5f}</pose>
        <cast_shadows>false</cast_shadows>
        <geometry><box><size>{length:.4f} {width:.4f} {THICKNESS}</size></box></geometry>
        <material>
          <ambient>{r} {g} {b} 1</ambient>
          <diffuse>{r} {g} {b} 1</diffuse>
          <specular>0.05 0.05 0.05 1</specular>
        </material>
      </visual>"""


def _strip(name: str, path: Path, offset: float, width: float, style,
           overlap: float = 0.0) -> list[str]:
    """Boxes along the path, `offset` to its left. `overlap` lengthens each box
    to close the wedge-shaped gaps a wide strip leaves on the outside of a bend."""
    poses = [path.pose_at(s) for s in path.samples(ARC_STEP)]
    out = []
    for i, (a, b) in enumerate(zip(poses, poses[1:])):
        (x0, y0), (x1, y1) = a.left(offset), b.left(offset)
        turn = abs(math.remainder(b.yaw - a.yaw, 2.0 * math.pi))
        out.append(_box(f'{name}_{i}', x0, y0, x1, y1, width, style, overlap * turn))
    return out


def _dashes(name: str, path: Path) -> list[str]:
    out = []
    s, i = 0.0, 0
    while s + DASH_LENGTH <= path.length:
        a, b = path.pose_at(s), path.pose_at(s + DASH_LENGTH)
        out.append(_box(f'{name}_{i}', a.x, a.y, b.x, b.y, LINE_WIDTH, CENTRE))
        s += DASH_LENGTH + DASH_GAP
        i += 1
    return out


def road_sdf(track: Track, road_width: float, name: str = 'road') -> str:
    half = road_width / 2.0
    visuals = []
    for path in track.paths:
        p = path.name
        visuals += _strip(f'{p}_surface', path, 0.0, road_width, SURFACE, overlap=half)
        for side, sign in (('left', 1.0), ('right', -1.0)):
            visuals += _strip(f'{p}_edge_{side}', path, sign * (half - LINE_WIDTH / 2.0),
                              LINE_WIDTH, EDGE)
        visuals += _dashes(f'{p}_centre', path)

    return f"""<?xml version="1.0"?>
<sdf version="1.9">
  <model name="{name}">
    <static>true</static>
    <link name="link">{''.join(visuals)}
    </link>
  </model>
</sdf>
"""
