"""When the link counts as lost, and when it counts as back.

A flickering link must not start and cancel the fallback in turn, and a rover
nobody has ever driven must not run one at all.

No ROS import anywhere in here — link_monitor is deliberately standalone.
"""

from rover_link_guard.link_monitor import LinkMonitor, LinkParams, LinkState, SpeedMemory

PARAMS = LinkParams()
DT = 0.05   # a 20 Hz operator


def feed(monitor, t0, duration):
    """Commands every DT for `duration`; returns the time after the last one."""
    t = t0
    for _ in range(round(duration / DT)):
        monitor.on_command(t)
        t += DT
    return t


def test_ok_until_the_first_command_ever_arrives():
    monitor = LinkMonitor(PARAMS)
    assert monitor.update(0.0) is LinkState.OK
    assert monitor.update(1000.0) is LinkState.OK


def test_steady_commands_keep_the_link_ok():
    monitor = LinkMonitor(PARAMS)
    t = feed(monitor, 0.0, 5.0)
    assert monitor.update(t) is LinkState.OK


def test_silence_is_lost_only_after_lost_after():
    monitor = LinkMonitor(PARAMS)
    t = feed(monitor, 0.0, 1.0)
    last = t - DT
    assert monitor.update(last + PARAMS.lost_after - 0.01) is LinkState.OK
    assert monitor.update(last + PARAMS.lost_after + 0.01) is LinkState.LOST


def test_a_single_command_does_not_restore_the_link():
    monitor = LinkMonitor(PARAMS)
    t = feed(monitor, 0.0, 1.0)
    assert monitor.update(t + 5.0) is LinkState.LOST
    monitor.on_command(t + 5.0)
    assert monitor.update(t + 5.0) is LinkState.LOST


def test_the_link_is_back_after_restore_after_of_steady_commands():
    monitor = LinkMonitor(PARAMS)
    t = feed(monitor, 0.0, 1.0)
    assert monitor.update(t + 5.0) is LinkState.LOST
    resumed = t + 5.0
    t = feed(monitor, resumed, PARAMS.restore_after - 0.2)
    assert monitor.update(t) is LinkState.LOST
    t = feed(monitor, t, 0.5)
    assert monitor.update(t) is LinkState.OK


def test_a_pause_inside_the_restore_window_starts_it_over():
    monitor = LinkMonitor(PARAMS)
    t = feed(monitor, 0.0, 1.0)
    assert monitor.update(t + 5.0) is LinkState.LOST
    t = feed(monitor, t + 5.0, 0.8)
    t += 0.8   # longer than command_gap
    t = feed(monitor, t, 0.8)
    assert monitor.update(t) is LinkState.LOST
    t = feed(monitor, t, 0.5)
    assert monitor.update(t) is LinkState.OK


def test_loss_after_a_restore_is_detected_again():
    monitor = LinkMonitor(PARAMS)
    t = feed(monitor, 0.0, 1.0)
    assert monitor.update(t + 3.0) is LinkState.LOST
    t = feed(monitor, t + 3.0, 2.0)
    assert monitor.update(t) is LinkState.OK
    assert monitor.update(t + 3.0) is LinkState.LOST


def test_zero_twists_right_after_driving_do_not_hide_the_speed():
    speed = SpeedMemory(LinkParams())
    speed.on_command(10.00, 0.8)
    for t in (10.20, 10.25, 10.30):   # the zero burst a dropped gamepad sends
        speed.on_command(t, 0.0)
    assert speed.at_drop() == 0.8


def test_a_stop_older_than_the_memory_is_a_real_stop():
    speed = SpeedMemory(LinkParams())
    speed.on_command(10.0, 0.8)
    for i in range(40):
        speed.on_command(10.5 + i * 0.05, 0.0)
    assert speed.at_drop() == 0.0


def test_no_commands_means_no_speed():
    assert SpeedMemory(LinkParams()).at_drop() == 0.0
