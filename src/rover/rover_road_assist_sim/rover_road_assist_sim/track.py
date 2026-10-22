#!/usr/bin/env python3
"""The simulated road: centre lines built from straights and arcs, and where a
pose is relative to them.

A track is a main path plus at most one branch that forks off it. Each path is
a chain of segments written as short strings, so they fit in a ROS parameter:

    "straight 20"      20 m ahead
    "left 15 90"       radius 15 m, 90 degrees to the left
    "right 8 120"

Everything is in the world frame and REP-103: x forward at yaw 0, y left,
angles counter-clockwise. The lateral offsets here are the rover's, positive
when it is left of the centre line; the camera convention of RoadDeviation is
applied by perception_model, not here.

No ROS import anywhere in here — track is deliberately standalone.
"""

import math
from dataclasses import dataclass


def wrap(angle: float) -> float:
    """Angle in [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    yaw: float

    def ahead(self, distance: float) -> 'Pose':
        return Pose(self.x + distance * math.cos(self.yaw),
                    self.y + distance * math.sin(self.yaw), self.yaw)

    def left(self, distance: float) -> tuple[float, float]:
        """The point `distance` to the left of this pose."""
        return (self.x - distance * math.sin(self.yaw),
                self.y + distance * math.cos(self.yaw))


@dataclass(frozen=True)
class Projection:
    """The closest point of one path to a query point."""

    #: Which path: 'main' or 'branch'.
    path: str
    #: [m] Along the path, from its start.
    s: float
    #: [m] Query point relative to the centre line, + = left of it.
    lateral: float
    #: [rad] Road direction at the closest point.
    heading: float
    #: [1/m] + = road turning left.
    curvature: float
    #: False when the query point lies before the path's start or past its
    #: end, i.e. the closest point is an end and the road is not beside it.
    beside: bool


class Straight:

    def __init__(self, start: Pose, length: float):
        self.start = start
        self.length = length
        self.curvature = 0.0
        self.end = start.ahead(length)

    def pose_at(self, s: float) -> Pose:
        return self.start.ahead(s)

    def locate(self, x: float, y: float) -> tuple[float, float]:
        """(s along, lateral), s unclamped."""
        c, si = math.cos(self.start.yaw), math.sin(self.start.yaw)
        dx, dy = x - self.start.x, y - self.start.y
        return dx * c + dy * si, -dx * si + dy * c

    def interior_samples(self, _ds: float) -> list[float]:
        return []


class Arc:

    def __init__(self, start: Pose, radius: float, sweep: float):
        """sweep [rad], + = left."""
        self.start = start
        self.radius = radius
        self._sign = 1.0 if sweep > 0.0 else -1.0
        self._sweep = abs(sweep)
        self.length = radius * self._sweep
        self.curvature = self._sign / radius
        self._cx, self._cy = start.left(self._sign * radius)
        self.end = self.pose_at(self.length)

    def pose_at(self, s: float) -> Pose:
        u = s / self.radius
        yaw = self.start.yaw + self._sign * u
        # The centre is on the turning side, so the point is back across it.
        return Pose(self._cx + self._sign * self.radius * math.sin(yaw),
                    self._cy - self._sign * self.radius * math.cos(yaw), yaw)

    def locate(self, x: float, y: float) -> tuple[float, float]:
        """(s along, lateral), s unclamped within half a turn of the arc."""
        dx, dy = x - self._cx, y - self._cy
        start_angle = self.start.yaw - self._sign * math.pi / 2.0
        swept = self._sign * (math.atan2(dy, dx) - start_angle)
        # Centred on the middle of the arc, so the ambiguity of a full turn
        # falls on the far side, away from the arc itself.
        u = wrap(swept - self._sweep / 2.0) + self._sweep / 2.0
        return u * self.radius, self._sign * (self.radius - math.hypot(dx, dy))

    def interior_samples(self, ds: float) -> list[float]:
        n = max(1, math.ceil(self.length / ds))
        return [self.length * i / n for i in range(1, n)]


def parse_segment(text: str, scale: float = 1.0):
    """'straight L' | 'left R deg' | 'right R deg' -> (kind, length or radius, sweep)."""
    parts = text.split()
    try:
        if parts[0] == 'straight' and len(parts) == 2:
            return 'straight', float(parts[1]) * scale, 0.0
        if parts[0] in ('left', 'right') and len(parts) == 3:
            sweep = math.radians(float(parts[2]))
            return parts[0], float(parts[1]) * scale, sweep if parts[0] == 'left' else -sweep
    except ValueError:
        pass
    raise ValueError(f"bad track segment {text!r}: expected 'straight L', "
                     f"'left R deg' or 'right R deg'")


class Path:

    def __init__(self, name: str, start: Pose, segments: list[str], scale: float = 1.0):
        if not segments:
            raise ValueError(f'track path {name!r} has no segments')
        self.name = name
        self.pieces = []
        pose = start
        for text in segments:
            kind, size, sweep = parse_segment(text, scale)
            piece = Straight(pose, size) if kind == 'straight' else Arc(pose, size, sweep)
            self.pieces.append(piece)
            pose = piece.end
        self._offsets = []
        total = 0.0
        for piece in self.pieces:
            self._offsets.append(total)
            total += piece.length
        self.length = total

    @property
    def start(self) -> Pose:
        return self.pieces[0].start

    def piece_start(self, index: int) -> Pose:
        return self.pieces[index].start

    def pose_at(self, s: float) -> Pose:
        s = min(max(s, 0.0), self.length)
        for piece, offset in zip(reversed(self.pieces), reversed(self._offsets)):
            if s >= offset:
                return piece.pose_at(s - offset)
        return self.pieces[0].pose_at(0.0)

    def max_curvature(self, s0: float, s1: float) -> float:
        """Largest |curvature| anywhere in [s0, s1]."""
        return max((abs(piece.curvature)
                    for piece, offset in zip(self.pieces, self._offsets)
                    if offset <= s1 and offset + piece.length >= s0), default=0.0)

    def samples(self, ds: float) -> list[float]:
        """s values that trace the path: every piece boundary, plus points at
        most ds apart along arcs. Straights need no points in between."""
        out = []
        for piece, offset in zip(self.pieces, self._offsets):
            out.append(offset)
            out.extend(offset + s for s in piece.interior_samples(ds))
        out.append(self.length)
        return out

    def project(self, x: float, y: float) -> Projection:
        best = None
        for index, (piece, offset) in enumerate(zip(self.pieces, self._offsets)):
            s, lateral = piece.locate(x, y)
            clamped = min(max(s, 0.0), piece.length)
            if clamped != s:
                # Past one end of this piece: measure to that end instead.
                end = piece.pose_at(clamped)
                lateral = math.copysign(math.hypot(x - end.x, y - end.y), lateral)
            if best is None or abs(lateral) < abs(best[1]):
                before_start = index == 0 and s < 0.0
                past_end = index == len(self.pieces) - 1 and s > piece.length
                best = (offset + clamped, lateral, piece, clamped,
                        not (before_start or past_end))
        s, lateral, piece, local, beside = best
        return Projection(self.name, s, lateral, piece.pose_at(local).yaw,
                          piece.curvature, beside)


def build_track(start: list[float], scale: float, main: list[str], branch_from: int,
                branch: list[str]) -> 'Track':
    """The track from its config: start is [x, y, yaw in degrees]. Shared by
    road_sim_node and the launch file that draws it, so both build the same one."""
    x, y, yaw_deg = (list(start) + [0.0, 0.0, 0.0])[:3]
    return Track.from_spec(Pose(float(x), float(y), math.radians(float(yaw_deg))),
                           main, int(branch_from), branch, float(scale))


class Track:

    def __init__(self, main: Path, branch: Path | None = None):
        self.main = main
        self.branch = branch

    @property
    def paths(self) -> list[Path]:
        return [p for p in (self.main, self.branch) if p is not None]

    def path(self, name: str) -> Path:
        return next(p for p in self.paths if p.name == name)

    @classmethod
    def from_spec(cls, start: Pose, main: list[str], branch_from: int = -1,
                  branch: list[str] = (), scale: float = 1.0) -> 'Track':
        """branch_from: index of the main segment the branch forks at the
        start of; -1 or an empty branch means no branch."""
        main_path = Path('main', start, list(main), scale)
        if branch_from < 0 or not branch:
            return cls(main_path)
        if branch_from >= len(main_path.pieces):
            raise ValueError(f'branch.from {branch_from} is past the end of main '
                             f'({len(main_path.pieces)} segments)')
        return cls(main_path, Path('branch', main_path.piece_start(branch_from),
                                   list(branch), scale))

    def candidates(self, x: float, y: float, yaw: float, max_lateral: float,
                   max_heading: float) -> list[Projection]:
        """The paths a camera at (x, y, yaw) could plausibly be looking along:
        beside the road, within max_lateral of it, and pointing within
        max_heading of its direction. Nearest first."""
        found = []
        for path in self.paths:
            p = path.project(x, y)
            if (p.beside and abs(p.lateral) <= max_lateral
                    and abs(wrap(yaw - p.heading)) <= max_heading):
                found.append(p)
        return sorted(found, key=lambda p: abs(p.lateral))
