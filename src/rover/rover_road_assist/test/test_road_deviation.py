"""Offset, heading and validity of the road deviation, pinned on synthetic masks."""

import numpy as np

from rover_road_assist.road_deviation import (
    Deviation, DeviationFilter, estimate_deviation)

H, W = 100, 200


def road(centre_at, half_width=30):
    """Mask whose road centre is centre_at(row), nothing above the horizon."""
    mask = np.zeros((H, W), dtype=np.uint8)
    for r in range(40, H):
        c = int(centre_at(r))
        mask[r, max(0, c - half_width):c + half_width] = 1
    return mask


# -- offset ------------------------------------------------------------

def test_centred_straight_road_has_no_deviation():
    d = estimate_deviation(road(lambda r: W / 2))
    assert d.valid
    assert abs(d.lateral_offset) < 0.02
    assert abs(d.heading_error) < 0.02


def test_road_to_the_right_gives_positive_offset():
    d = estimate_deviation(road(lambda r: W / 2 + 40))
    assert d.valid
    assert abs(d.lateral_offset - 40 / 30) < 0.05


def test_road_to_the_left_gives_negative_offset():
    d = estimate_deviation(road(lambda r: W / 2 - 40))
    assert abs(d.lateral_offset + 40 / 30) < 0.05


def test_target_column_shifts_the_reference():
    mask = road(lambda r: W / 2 + 40)
    d = estimate_deviation(mask, target_col_fraction=0.7)
    assert abs(d.lateral_offset) < 0.05


# -- heading -----------------------------------------------------------

def test_road_bending_right_ahead_gives_positive_heading():
    # Centre moves right as the row index falls, i.e. further up the image.
    d = estimate_deviation(road(lambda r: W / 2 - (r - 70) * 0.5))
    assert d.heading_error > 0.1


def test_road_bending_left_ahead_gives_negative_heading():
    d = estimate_deviation(road(lambda r: W / 2 + (r - 70) * 0.5))
    assert d.heading_error < -0.1


# -- validity ----------------------------------------------------------

def test_empty_mask_is_invalid():
    d = estimate_deviation(np.zeros((H, W), dtype=np.uint8))
    assert not d.valid
    assert d.confidence == 0.0


def test_speckle_without_a_run_is_invalid():
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[60:100:2, 10:200:7] = 1
    assert not estimate_deviation(mask).valid


def test_road_above_horizon_only_is_invalid():
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[:30, 50:150] = 1
    assert not estimate_deviation(mask).valid


def test_nearest_run_to_the_target_is_used():
    mask = road(lambda r: W / 2)
    mask[40:, 0:15] = 1
    d = estimate_deviation(mask)
    assert abs(d.lateral_offset) < 0.02


def test_offset_is_clipped_to_two():
    d = estimate_deviation(road(lambda r: W - 25, half_width=20), target_col_fraction=0.0)
    assert d.valid
    assert d.lateral_offset == 2.0


def test_more_road_means_more_confidence():
    wide = estimate_deviation(road(lambda r: W / 2, half_width=45))
    narrow = estimate_deviation(road(lambda r: W / 2, half_width=8))
    assert wide.confidence > narrow.confidence


def test_works_at_other_resolutions():
    mask = np.kron(road(lambda r: W / 2 + 40), np.ones((3, 3), dtype=np.uint8))
    d = estimate_deviation(mask)
    assert abs(d.lateral_offset - 40 / 30) < 0.05


# -- smoothing ---------------------------------------------------------

def test_filter_smooths_valid_frames():
    f = DeviationFilter(alpha=0.5)
    f.update(Deviation(0.0, 0.0, 1.0, True))
    d = f.update(Deviation(1.0, 0.0, 1.0, True))
    assert d.lateral_offset == 0.5


def test_filter_restarts_after_an_invalid_frame():
    f = DeviationFilter(alpha=0.5)
    f.update(Deviation(0.0, 0.0, 1.0, True))
    f.update(Deviation())
    d = f.update(Deviation(1.0, 0.0, 1.0, True))
    assert d.lateral_offset == 1.0


def test_filter_passes_invalid_through():
    assert not DeviationFilter().update(Deviation()).valid
