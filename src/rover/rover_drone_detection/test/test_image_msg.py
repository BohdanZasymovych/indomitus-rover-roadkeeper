from types import SimpleNamespace

import numpy as np
import pytest

from rover_drone_detection.image_msg import bgr_fields, to_bgr


def msg(encoding, img, pad=0):
    h, w = img.shape[:2]
    rows = img.reshape(h, -1)
    step = rows.shape[1] + pad
    data = np.zeros((h, step), np.uint8)
    data[:, :rows.shape[1]] = rows
    return SimpleNamespace(encoding=encoding, height=h, width=w, step=step, data=data.tobytes())


BGR = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)


def test_bgr8_passthrough():
    assert np.array_equal(to_bgr(msg('bgr8', BGR)), BGR)


def test_rgb8_swaps_channels():
    assert np.array_equal(to_bgr(msg('rgb8', BGR[:, :, ::-1].copy())), BGR)


def test_rgba8_drops_alpha():
    rgba = np.dstack([BGR[:, :, ::-1], np.full((2, 3), 255, np.uint8)])
    assert np.array_equal(to_bgr(msg('rgba8', rgba)), BGR)


def test_mono8_to_three_channels():
    grey = np.array([[0, 128, 255], [1, 2, 3]], np.uint8)
    out = to_bgr(msg('mono8', grey))
    assert out.shape == (2, 3, 3)
    assert np.array_equal(out[:, :, 0], grey) and np.array_equal(out[:, :, 2], grey)


def test_padded_rows():
    assert np.array_equal(to_bgr(msg('rgb8', BGR[:, :, ::-1].copy(), pad=7)), BGR)


def test_unsupported_encoding():
    with pytest.raises(ValueError, match='unsupported encoding'):
        to_bgr(msg('16UC1', BGR))


def test_short_data():
    m = msg('bgr8', BGR)
    m.data = m.data[:-1]
    with pytest.raises(ValueError, match='bytes'):
        to_bgr(m)


def test_bgr_fields_round_trip():
    h, w, step, data = bgr_fields(BGR)
    out = to_bgr(SimpleNamespace(encoding='bgr8', height=h, width=w, step=step, data=data))
    assert (h, w, step) == (2, 3, 9) and np.array_equal(out, BGR)
