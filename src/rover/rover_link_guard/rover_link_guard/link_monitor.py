#!/usr/bin/env python3
"""
Whether the operator's link is up, judged from the commands that arrive over it.

The laptop teleop publishes its command at a steady rate whether or not the
sticks move, so silence on that topic means the link is down, not that the
operator is idle. This is the same signal twist_mux sees when it times the
source out, with two differences: loss is only declared after `lost_after`,
and the link only counts as back after commands have kept arriving for
`restore_after`. Without that hold, an outage that flickers (dev_0009 has six
in four minutes, some one second long) would start and cancel the fallback in
turn.

Nothing is declared lost before the first command has ever arrived, so a rover
that is started with no operator connected does not run a fallback.

No ROS import anywhere in here — link_monitor is deliberately standalone.
"""

from dataclasses import dataclass
from enum import Enum


class LinkState(str, Enum):
    OK = 'OK'
    LOST = 'LOST'


@dataclass(frozen=True)
class LinkParams:
    #: [s] Silence longer than this declares the link lost. Keep it above
    #: command_gap, or one dropped command is already a loss.
    lost_after: float = 1.0
    #: [s] Commands must keep arriving this long for the link to count as back.
    restore_after: float = 1.0
    #: [s] A pause between commands longer than this breaks the run that
    #: restore_after counts. Matches the cmd_vel_remote timeout in twist_mux.yaml.
    command_gap: float = 0.5


class LinkMonitor:
    """Tracks the link state from command arrival times."""

    def __init__(self, params: LinkParams):
        self._p = params
        self._last = None
        self._run_start = None
        self._lost = False

    def on_command(self, now: float):
        if self._last is None or now - self._last > self._p.command_gap:
            self._run_start = now
        self._last = now

    def update(self, now: float) -> LinkState:
        """The state at `now`; call it as often as the state is acted on."""
        if self._last is None:
            return LinkState.OK
        if not self._lost:
            self._lost = now - self._last > self._p.lost_after
        else:
            flowing = now - self._last <= self._p.command_gap
            self._lost = not (flowing and now - self._run_start >= self._p.restore_after)
        return LinkState.LOST if self._lost else LinkState.OK
