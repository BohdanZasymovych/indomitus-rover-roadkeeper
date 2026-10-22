"""Every confirmed drone sighting, on disk, for later work (a map of dangerous
road sections, retraining on real frames, debriefs).

    <log_dir>/<YYYY-MM-DD>/sightings.jsonl     one JSON object per line
    <log_dir>/<YYYY-MM-DD>/snapshots/*.jpg     the annotated frame of a record

A record is written on every level change (including back to CLEAR, without a
snapshot) and every `interval` seconds while the level is not CLEAR. The
fields are whatever dict the node passes; NaN becomes null so that every line
is valid JSON.

No ROS import anywhere in here.
"""

import json
import math
import os
import time

import cv2

from rover_drone_detection.threat import Level


def _clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


class SightingLog:

    def __init__(self, log_dir, interval=1.0, jpeg_quality=85):
        self.root = os.path.expanduser(log_dir)
        self.interval = interval
        self.jpeg_quality = jpeg_quality
        self._last_stamp = None
        self._last_level = Level.CLEAR

    def due(self, stamp, level):
        """Whether a frame at `stamp` with `level` should be recorded."""
        if level != self._last_level:
            return True
        if level == Level.CLEAR:
            return False
        return self._last_stamp is None or stamp - self._last_stamp >= self.interval

    def write(self, stamp, level, record, snapshot=None, wall_time=None):
        """Append `record` (+ level, stamps, snapshot path); returns the jsonl path."""
        wall_time = time.time() if wall_time is None else wall_time
        local = time.localtime(wall_time)
        day = os.path.join(self.root, time.strftime('%Y-%m-%d', local))
        os.makedirs(day, exist_ok=True)

        rec = {'stamp': stamp, 'wall_time': wall_time, 'level': Level(level).name}
        rec.update(record)
        rec['snapshot'] = None
        if snapshot is not None and level != Level.CLEAR:
            snaps = os.path.join(day, 'snapshots')
            os.makedirs(snaps, exist_ok=True)
            name = '%s_%03d_%s.jpg' % (time.strftime('%H%M%S', local),
                                       int(wall_time * 1000) % 1000, Level(level).name)
            if cv2.imwrite(os.path.join(snaps, name), snapshot,
                           [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]):
                rec['snapshot'] = os.path.join('snapshots', name)

        path = os.path.join(day, 'sightings.jsonl')
        with open(path, 'a') as f:
            f.write(json.dumps(_clean(rec)) + '\n')
        self._last_stamp = stamp
        self._last_level = Level(level)
        return path
