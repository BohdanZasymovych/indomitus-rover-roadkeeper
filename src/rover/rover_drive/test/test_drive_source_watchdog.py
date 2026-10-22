"""Which input wins the twist_mux race, pinned.

DriveSourceWatchdog has no ROS in it and takes its clock as an argument
precisely so these can be tested here. The priorities/timeouts below
mirror rover_drive/config/twist_mux.yaml at the time this was written.
"""

from rover_drive.drive_source_watchdog import DriveSourceWatchdog, Source


class FakeClock:
    """Time only moves when a test says so."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


SOURCES = (
    Source('cmd_vel_joy', priority=100, timeout=0.5, autonomous=False),
    Source('cmd_vel_gs', priority=50, timeout=0.5, autonomous=False),
    Source('cmd_vel_remote', priority=30, timeout=0.5, autonomous=False),
    Source('cmd_vel_ext', priority=10, timeout=0.5, autonomous=False),
)


def make_watchdog(clock=None):
    return DriveSourceWatchdog(SOURCES, clock=clock or FakeClock())


# -- starting state ----------------------------------------------------

def test_starts_with_no_winner_before_any_message():
    watchdog = make_watchdog()
    assert watchdog.winner() is None
    assert watchdog.autonomous() is None


# -- priority ordering ----------------------------------------------------

def test_only_source_wins_by_default():
    watchdog = make_watchdog()
    watchdog.on_message('cmd_vel_ext')
    assert watchdog.winner() == 'cmd_vel_ext'
    assert watchdog.autonomous() is False


def test_higher_priority_source_wins_even_if_older():
    clock = FakeClock()
    watchdog = make_watchdog(clock)
    watchdog.on_message('cmd_vel_ext')
    clock.advance(0.2)
    watchdog.on_message('cmd_vel_joy')
    # Both are still fresh; joystick (100) outranks external (10) regardless
    # of which one published more recently - this is twist_mux's rule, not a
    # most-recent-wins one.
    assert watchdog.winner() == 'cmd_vel_joy'


def test_lower_priority_source_wins_once_the_higher_one_goes_stale():
    clock = FakeClock()
    watchdog = make_watchdog(clock)
    watchdog.on_message('cmd_vel_gs')
    clock.advance(0.6)                     # gs's 0.5s timeout elapses
    watchdog.on_message('cmd_vel_ext')
    assert watchdog.winner() == 'cmd_vel_ext'


# -- silence ----------------------------------------------------------------

def test_no_winner_once_the_only_source_goes_stale():
    clock = FakeClock()
    watchdog = make_watchdog(clock)
    watchdog.on_message('cmd_vel_joy')
    clock.advance(0.6)
    assert watchdog.winner() is None
    assert watchdog.autonomous() is None


def test_falls_through_to_the_next_fresh_source_on_timeout():
    clock = FakeClock()
    watchdog = make_watchdog(clock)
    watchdog.on_message('cmd_vel_joy')
    clock.advance(0.3)
    watchdog.on_message('cmd_vel_remote')
    clock.advance(0.3)                     # joystick's 0.5s timeout elapses
    assert watchdog.winner() == 'cmd_vel_remote'


def test_recovers_without_a_restart():
    clock = FakeClock()
    watchdog = make_watchdog(clock)
    watchdog.on_message('cmd_vel_joy')
    clock.advance(0.6)
    assert watchdog.winner() is None

    watchdog.on_message('cmd_vel_gs')
    assert watchdog.winner() == 'cmd_vel_gs'
    assert watchdog.autonomous() is False
