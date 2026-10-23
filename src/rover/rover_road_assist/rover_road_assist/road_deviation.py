"""Road offset and heading from a binary road mask.

No ROS in it, so it can be tested without a ROS environment.

The road centre is read off a few horizontal scanlines below the horizon
instead of from the centroid of the whole mask: the centroid is pulled around
by far-away road, wide verges and curves, a scanline is not. Sign convention
everywhere: + is to the right of the vehicle.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Deviation:
    lateral_offset: float = 0.0
    heading_error: float = 0.0
    confidence: float = 0.0
    valid: bool = False


INVALID = Deviation()

#: [road half-widths] Further off than this the road is out of view anyway.
MAX_OFFSET = 2.0


def _run_near(row, target_col, min_width):
    """(centre, width) of the road run in a row closest to target_col, or None."""
    padded = np.concatenate(([0], (row > 0).view(np.int8), [0]))
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)  # exclusive
    best = None
    best_dist = None
    for start, end in zip(starts, ends):
        if end - start < min_width:
            continue
        if start <= target_col < end:
            dist = 0
        else:
            dist = min(abs(start - target_col), abs(end - 1 - target_col))
        if best_dist is None or dist < best_dist:
            best, best_dist = (start, end), dist
    if best is None:
        return None
    start, end = best
    return (start + end - 1) / 2.0, float(end - start)


def estimate_deviation(mask, row_fractions=(0.6, 0.7, 0.8, 0.9), horizon_fraction=0.4,
                       target_col_fraction=0.5, min_run_fraction=0.05, min_rows=2,
                       min_road_fraction=0.02, full_confidence_road_fraction=0.2):
    """Deviation of the vehicle from the road in a 2-D mask, 0 = no road, else road."""
    mask = np.asarray(mask)
    height, width = mask.shape
    top = int(height * horizon_fraction)
    roi = mask[top:]
    road_fraction = float(np.count_nonzero(roi)) / roi.size if roi.size else 0.0
    if road_fraction < min_road_fraction:
        return INVALID

    target_col = target_col_fraction * (width - 1)
    min_width = max(1, int(min_run_fraction * width))

    rows, centres, half_widths = [], [], []
    for fraction in row_fractions:
        r = min(height - 1, max(top, int(fraction * height)))
        run = _run_near(mask[r], target_col, min_width)
        if run is not None:
            rows.append(r)
            centres.append(run[0])
            half_widths.append(run[1] / 2.0)
    if len(rows) < min_rows:
        return INVALID

    rows = np.asarray(rows, dtype=float)
    centres = np.asarray(centres, dtype=float)
    half_widths = np.asarray(half_widths, dtype=float)

    # In half-widths of the road itself, row by row, so it does not depend on
    # the camera's field of view or on how far the row is.
    # Nearer rows (larger index) weigh more: they say where the vehicle is now.
    weights = rows / rows.sum()
    lateral_offset = float(np.sum(weights * (centres - target_col) / half_widths))

    # Road direction: pixels moved sideways per pixel of row, upward in the
    # image is forward, so a centre that drifts right going up has a negative
    # slope against the row index.
    slope = np.polyfit(rows, centres, 1)[0]
    heading_error = float(np.arctan(-slope))

    coverage = len(rows) / len(row_fractions)
    road_score = min(1.0, road_fraction / full_confidence_road_fraction)
    return Deviation(
        lateral_offset=float(np.clip(lateral_offset, -MAX_OFFSET, MAX_OFFSET)),
        heading_error=heading_error,
        confidence=float(coverage * road_score),
        valid=True,
    )


class DeviationFilter:
    """Exponential smoothing that restarts from scratch after an invalid frame."""

    def __init__(self, alpha=0.5):
        self.alpha = float(alpha)
        self._last = None

    def update(self, deviation):
        if not deviation.valid:
            self._last = None
            return deviation
        if self._last is not None:
            a = self.alpha
            deviation = Deviation(
                lateral_offset=a * deviation.lateral_offset + (1 - a) * self._last.lateral_offset,
                heading_error=a * deviation.heading_error + (1 - a) * self._last.heading_error,
                confidence=a * deviation.confidence + (1 - a) * self._last.confidence,
                valid=True,
            )
        self._last = deviation
        return deviation
