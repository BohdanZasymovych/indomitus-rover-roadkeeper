#!/usr/bin/env python3
"""A stand-in for the road camera: what it would report from a true pose.

Turns the rover's ground-truth pose and the track into the four numbers
RoadDeviation carries, in its convention (offset as a fraction of the
half-width, + = road to the right), with as much imperfection as configured:
noise, latency and a confidence that drops where a real camera would struggle.

No ROS import anywhere in here — perception_model is deliberately standalone.
"""

import bisect
import math
import random
from dataclasses import dataclass

from rover_road_assist_sim.track import Pose, Projection, Track, wrap


def smoothstep(x: float, a: float, b: float) -> float:
    t = min(max((x - a) / (b - a), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


@dataclass(frozen=True)
class CameraParams:
    #: [m]
    road_width: float = 3.0
    #: [m] The camera reports the road this far ahead of the rover.
    lookahead: float = 0.0
    #: Road further off than this many half-widths is out of view.
    max_visible_offset: float = 2.0
    #: [rad] A road this far off the rover's heading is out of view.
    max_visible_heading: float = math.radians(60.0)


@dataclass(frozen=True)
class NoiseParams:
    enabled: bool = False
    #: [m]
    offset_std: float = 0.05
    #: [rad]
    heading_std: float = 0.02
    #: Model confidence only; a fixed confidence stays fixed.
    confidence_std: float = 0.03
    seed: int = 0


@dataclass(frozen=True)
class ConfidenceParams:
    #: 'fixed': always fixed_value. 'model': the factors below.
    mode: str = 'fixed'
    fixed_value: float = 0.9
    #: Drops as the rover nears and crosses the road edge.
    lateral: bool = True
    #: Drops as the road turns away from the camera's view.
    heading: bool = True
    #: Drops ahead of a tight curve, which leaves the view.
    curvature: bool = True
    #: [m] How far ahead the curvature factor looks.
    curvature_preview: float = 8.0
    #: Multiplies the confidence while two roads are in view (a fork).
    fork_confidence: float = 0.3


@dataclass(frozen=True)
class Observation:
    #: [-1..1]-ish, + = road centre to the right of the rover.
    lateral_offset: float
    #: [rad] + = road bends to the right.
    heading_error: float
    confidence: float
    valid: bool
    #: Which path the camera is following, or '' when none is in view.
    path: str


NOTHING_IN_VIEW = Observation(0.0, 0.0, 0.0, False, '')


def model_confidence(offset: float, heading_error: float, curvature_ahead: float,
                     ambiguous: bool, p: ConfidenceParams) -> float:
    """offset in half-widths, heading_error [rad], curvature_ahead [1/m]."""
    c = 1.0
    if p.lateral:
        c *= 1.0 - smoothstep(abs(offset), 0.7, 1.3)
    if p.heading:
        c *= 1.0 - smoothstep(abs(heading_error), 0.35, 0.7)
    if p.curvature:
        c *= 1.0 - smoothstep(curvature_ahead, 1.0 / 10.0, 1.0 / 4.0)
    if ambiguous:
        c *= p.fork_confidence
    return c


class DelayLine:
    """Timestamped samples, read back `delay` seconds late.

    The camera reports the pose an image was taken at, stamped with that
    time, so latency shows up downstream exactly as a real one would: as the
    age of the stamp.
    """

    def __init__(self, delay: float, keep: float = 2.0):
        self._delay = max(0.0, delay)
        self._keep = max(keep, self._delay + 0.5)
        self._stamps = []
        self._values = []

    def push(self, stamp: float, value):
        if self._stamps and stamp < self._stamps[-1]:
            # Time went backwards: a sim reset. Start over.
            self._stamps.clear()
            self._values.clear()
        self._stamps.append(stamp)
        self._values.append(value)
        cut = bisect.bisect_left(self._stamps, stamp - self._keep)
        del self._stamps[:cut]
        del self._values[:cut]

    def at(self, now: float):
        """(stamp, value) of the newest sample at least `delay` old, or None."""
        i = bisect.bisect_right(self._stamps, now - self._delay)
        if i == 0:
            return None
        return self._stamps[i - 1], self._values[i - 1]


class RoadCamera:
    """Pose in, Observation out. Remembers which branch it is on."""

    def __init__(self, track: Track, camera: CameraParams, noise: NoiseParams,
                 confidence: ConfidenceParams):
        if confidence.mode not in ('fixed', 'model'):
            raise ValueError(f"confidence.mode must be 'fixed' or 'model', "
                             f"got {confidence.mode!r}")
        self._track = track
        self._camera = camera
        self._noise = noise
        self._confidence = confidence
        self._rng = random.Random(noise.seed)
        self._following = ''

    def observe(self, pose: Pose) -> Observation:
        cam = self._camera
        half = cam.road_width / 2.0
        at = pose.ahead(cam.lookahead)
        found = self._track.candidates(at.x, at.y, at.yaw,
                                       cam.max_visible_offset * half,
                                       cam.max_visible_heading)
        if not found:
            self._following = ''
            return NOTHING_IN_VIEW

        road = self._choose(found)
        self._following = road.path
        # Two roads overlapping the view: a camera cannot know which one the
        # operator means, and its fit gets worse for having both.
        ambiguous = sum(1 for p in found if abs(p.lateral) < cam.road_width) > 1

        lateral = road.lateral
        heading_error = wrap(pose.yaw - road.heading)
        if self._noise.enabled:
            lateral += self._rng.gauss(0.0, self._noise.offset_std)
            heading_error += self._rng.gauss(0.0, self._noise.heading_std)
        # Rover left of the centre line = road centre to its right = +.
        offset = lateral / half

        return Observation(offset, heading_error, self._confidence_for(
            offset, heading_error, road, ambiguous), True, road.path)

    def _choose(self, found: list[Projection]) -> Projection:
        # Keep following the same road while it is still under the rover, so
        # the choice at a fork does not flip back and forth.
        for p in found:
            if p.path == self._following and abs(p.lateral) < self._camera.road_width / 2.0:
                return p
        return found[0]

    def _confidence_for(self, offset: float, heading_error: float, road: Projection,
                        ambiguous: bool) -> float:
        p = self._confidence
        if p.mode == 'fixed':
            return p.fixed_value
        path = self._track.path(road.path)
        curvature_ahead = path.max_curvature(road.s, road.s + p.curvature_preview)
        c = model_confidence(offset, heading_error, curvature_ahead, ambiguous, p)
        if self._noise.enabled:
            c += self._rng.gauss(0.0, self._noise.confidence_std)
        return min(max(c, 0.0), 1.0)
