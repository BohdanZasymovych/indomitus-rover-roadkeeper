#!/usr/bin/env python3
"""The path the rover has driven, kept so it can be driven back.

Points are added as the rover moves, `spacing` apart, each with the path
length from where the trail began. Retreating walks the trail from its newest
end and drops the points the rover has reached, so what is left is always the
way still to go.

No ROS import anywhere in here — trail is deliberately standalone.
"""

import math
from collections import deque
from typing import NamedTuple, Optional


class Point(NamedTuple):
    x: float
    y: float
    #: [m] Path length from the first point ever added.
    s: float


class Trail:

    def __init__(self, spacing: float, max_points: int):
        self._spacing = spacing
        self._points = deque(maxlen=max_points)

    def __len__(self) -> int:
        return len(self._points)

    @property
    def length(self) -> float:
        """[m] Path length at the newest point; 0 when empty."""
        return self._points[-1].s if self._points else 0.0

    def add(self, x: float, y: float):
        """Appends (x, y) unless it is closer than `spacing` to the newest point."""
        if not self._points:
            self._points.append(Point(x, y, 0.0))
            return
        last = self._points[-1]
        d = math.hypot(x - last.x, y - last.y)
        if d >= self._spacing:
            self._points.append(Point(x, y, last.s + d))

    def pop_reached(self, x: float, y: float, radius: float):
        """Drops the newest points that (x, y) is within `radius` of."""
        while self._points and math.hypot(
                self._points[-1].x - x, self._points[-1].y - y) < radius:
            self._points.pop()

    def target(self, x: float, y: float, lookahead: float) -> Optional[Point]:
        """The newest point at least `lookahead` from (x, y), or the oldest
        point when every one is closer: the end of the way back. None when empty."""
        for point in reversed(self._points):
            if math.hypot(point.x - x, point.y - y) >= lookahead:
                return point
        return self._points[0] if self._points else None

    def clear(self):
        self._points.clear()
