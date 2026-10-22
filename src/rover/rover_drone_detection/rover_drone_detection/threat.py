"""From per-frame detections to a confirmed threat level.

A single frame is never enough to raise the alarm: a detection has to repeat
`confirm_hits` times within `confirm_window`, and once raised the level holds
through short misses (motion blur, a frame the model drops) until no drone has
been seen for `clear_after`.

    CLEAR   nothing confirmed
    WARN    a confirmed drone in view
    DANGER  a confirmed drone that is big in the frame (close), or whose box
            bottom is in the path zone: the lower-centre of the image, where
            the road ahead of a forward camera is

No ROS import anywhere in here; the node only feeds it stamps and boxes.
"""

from collections import deque
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Sequence


class Level(IntEnum):
    # Same values as indomitus_interfaces/DroneThreat.
    CLEAR = 0
    WARN = 1
    DANGER = 2


@dataclass(frozen=True)
class Detection:
    """One box in pixel coordinates of the image it was detected in."""
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    cls: int = 0
    label: str = 'waiter_fpv_drone'

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1

    @property
    def cx(self):
        return 0.5 * (self.x1 + self.x2)

    @property
    def cy(self):
        return 0.5 * (self.y1 + self.y2)


@dataclass
class ThreatParams:
    confirm_hits: int = 3           # frames with a drone ...
    confirm_window: float = 1.0     # [s] ... within this long to raise the alarm
    clear_after: float = 2.0        # [s] without any drone to drop it
    danger_size: float = 0.12       # box height / image height at or above = DANGER
    path_half_width: float = 0.25   # path zone: |cx / w - 0.5| at most this ...
    horizon: float = 0.45           # ... and the box bottom y2 / h at least this (below the horizon)


def frame_level(det: Detection, width: int, height: int, p: ThreatParams) -> Level:
    """How bad this one box would be if it were confirmed."""
    if det.height / height >= p.danger_size:
        return Level.DANGER
    in_path = abs(det.cx / width - 0.5) <= p.path_half_width and det.y2 / height >= p.horizon
    return Level.DANGER if in_path else Level.WARN


@dataclass(frozen=True)
class ThreatState:
    level: Level
    count: int                      # detections in this frame
    target: Optional[Detection]     # the most threatening one this frame, if any

    @property
    def confidence(self):
        return self.target.score if self.target else 0.0


class ThreatTracker:

    def __init__(self, params: ThreatParams = ThreatParams()):
        self.p = params
        self.level = Level.CLEAR
        self._hits = deque()
        self._last_hit = None
        self._last_stamp = None

    def reset(self):
        self.level = Level.CLEAR
        self._hits.clear()
        self._last_hit = None
        self._last_stamp = None

    def update(self, stamp: float, dets: Sequence[Detection], width: int, height: int) -> ThreatState:
        """Feed one processed frame; `stamp` in seconds, monotonic per source."""
        if self._last_stamp is not None and stamp < self._last_stamp:
            self.reset()    # a bag looped or the clock jumped back
        self._last_stamp = stamp

        target = None
        if dets:
            target = max(dets, key=lambda d: (frame_level(d, width, height, self.p), d.score))
            self._hits.append(stamp)
            self._last_hit = stamp
        while self._hits and self._hits[0] < stamp - self.p.confirm_window:
            self._hits.popleft()

        if target is not None:
            if self.level != Level.CLEAR or len(self._hits) >= self.p.confirm_hits:
                self.level = frame_level(target, width, height, self.p)
        elif self._last_hit is None or stamp - self._last_hit >= self.p.clear_after:
            self.level = Level.CLEAR
        # else: a short miss, the level holds
        return ThreatState(self.level, len(dets), target)
