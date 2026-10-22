"""Confirmation, hold and clear of the drone threat level.

The rules that matter most: one stray frame never raises the alarm, a short
miss never drops it, and a drone on the path ahead is DANGER.
"""

from rover_drone_detection.threat import Detection, Level, ThreatParams, ThreatTracker, frame_level

W, H = 640, 360
P = ThreatParams()
DT = 0.1    # a 10 Hz detector


def far_side():
    """Small, on the right edge, above the horizon: WARN."""
    return Detection(600, 100, 620, 115, 0.6)


def on_path():
    """Small, centred, below the horizon: DANGER."""
    return Detection(310, 250, 330, 265, 0.5)


def run(tracker, frames, t0=0.0):
    states = []
    for i, dets in enumerate(frames):
        states.append(tracker.update(t0 + i * DT, dets, W, H))
    return states


def test_frame_level():
    assert frame_level(far_side(), W, H, P) == Level.WARN
    assert frame_level(on_path(), W, H, P) == Level.DANGER
    big = Detection(550, 10, 630, 10 + 0.2 * H, 0.9)
    assert frame_level(big, W, H, P) == Level.DANGER


def test_single_frame_does_not_alarm():
    states = run(ThreatTracker(P), [[far_side()]] + [[]] * 5)
    assert all(s.level == Level.CLEAR for s in states)
    assert states[0].count == 1 and states[0].target is not None


def test_confirmed_after_hits_in_window():
    states = run(ThreatTracker(P), [[far_side()]] * 3)
    assert [s.level for s in states] == [Level.CLEAR, Level.CLEAR, Level.WARN]


def test_hits_spread_wider_than_window_do_not_confirm():
    frames = [[far_side()]] + [[]] * 6 + [[far_side()]] + [[]] * 6 + [[far_side()]]
    assert all(s.level == Level.CLEAR for s in run(ThreatTracker(P), frames))


def test_holds_through_short_miss_then_clears():
    tracker = ThreatTracker(P)
    run(tracker, [[on_path()]] * 3)
    assert tracker.level == Level.DANGER
    misses = run(tracker, [[]] * 25, t0=0.3)
    held = [s.level for s in misses if s.level == Level.DANGER]
    # last hit at t=0.2; held while t - 0.2 < clear_after
    assert len(held) == 19
    assert misses[-1].level == Level.CLEAR


def test_most_threatening_detection_is_target():
    state = run(ThreatTracker(P), [[Detection(0, 0, 5, 5, 0.99), on_path()]] * 3)[-1]
    assert state.level == Level.DANGER
    assert state.target == on_path()
    assert state.count == 2


def test_clock_jump_back_resets():
    tracker = ThreatTracker(P)
    run(tracker, [[far_side()]] * 3, t0=100.0)
    assert tracker.level == Level.WARN
    assert tracker.update(1.0, [far_side()], W, H).level == Level.CLEAR
