import json
import math
import os

import numpy as np

from rover_drone_detection.sighting_log import SightingLog
from rover_drone_detection.threat import Level

WALL = 1_800_000_000.0


def lines(path):
    with open(path) as f:
        return [json.loads(x) for x in f]


def test_due_on_change_and_interval():
    log = SightingLog('/nonexistent', interval=1.0)
    assert not log.due(0.0, Level.CLEAR)
    assert log.due(0.0, Level.WARN)


def test_write_rate_limit_and_snapshot(tmp_path):
    log = SightingLog(str(tmp_path), interval=1.0)
    img = np.zeros((36, 64, 3), np.uint8)
    path = log.write(10.0, Level.WARN, {'range_est': math.nan, 'rover': {'x': 1.0}}, img, WALL)
    assert not log.due(10.5, Level.WARN)
    assert log.due(11.0, Level.WARN)
    assert log.due(10.5, Level.DANGER)
    log.write(12.0, Level.CLEAR, {}, img, WALL + 2)
    assert not log.due(13.0, Level.CLEAR)

    recs = lines(path)
    assert [r['level'] for r in recs] == ['WARN', 'CLEAR']
    assert recs[0]['range_est'] is None             # NaN -> null
    assert recs[0]['rover'] == {'x': 1.0}
    assert os.path.isfile(os.path.join(os.path.dirname(path), recs[0]['snapshot']))
    assert recs[1]['snapshot'] is None              # no picture of nothing
