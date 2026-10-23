import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rover_road_segmentation import config as C
from rover_road_segmentation.image_codec import image_to_array, mask_fields
from rover_road_segmentation.processing import clean_mask, hud_box, preprocess

RESOLUTIONS = [(364, 720), (364, 700), (720, 1280), (1520, 2688), (480, 640)]


def _road_prob(h, w):
    """A trapezoid road from the bottom edge towards the horizon, p = 0.9 inside."""
    prob = np.full((h, w), 0.1, np.float32)
    for r in range(int(0.45 * h), h):
        t = (r - 0.45 * h) / (0.55 * h)
        half = int((0.05 + 0.3 * t) * w)
        prob[r, w // 2 - half:w // 2 + half] = 0.9
    return prob


@pytest.mark.parametrize('h,w', RESOLUTIONS)
def test_mask_is_binary_and_keeps_resolution(h, w):
    mask = clean_mask(_road_prob(h, w))
    assert mask.shape == (h, w)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)) == {0, 1}


@pytest.mark.parametrize('h,w', RESOLUTIONS)
def test_preprocess_outputs_inference_size(h, w):
    frame = np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8)
    image, brightness = preprocess(frame, 'bgr')
    assert image.shape == (C.INFER_HEIGHT, C.INFER_WIDTH, 3)
    assert image.dtype == np.uint8
    assert 0.0 <= brightness <= 255.0


def test_isolated_false_positive_is_removed():
    h, w = 364, 720
    prob = _road_prob(h, w)
    prob[250:300, 20:60] = 0.95          # a bright patch in the field, away from the road
    mask = clean_mask(prob)
    assert mask[250:300, 20:60].sum() == 0
    assert mask[h - 5, w // 2] == 1


def test_threshold_is_strict():
    prob = np.full((100, 200), 0.6, np.float32)   # above 0.5 but below 0.65
    assert clean_mask(prob).sum() == 0


def test_tiny_component_means_no_road():
    prob = np.zeros((364, 720), np.float32)
    prob[340:350, 350:360] = 0.99
    assert clean_mask(prob).sum() == 0


@pytest.mark.parametrize('h,w', RESOLUTIONS)
def test_hud_box_is_never_road(h, w):
    mask = clean_mask(np.full((h, w), 0.99, np.float32))
    x0, y0, x1, y1 = hud_box(h, w)
    assert x1 > x0 and y1 > y0
    assert mask[y0:y1, x0:x1].sum() == 0
    assert clean_mask(np.full((h, w), 0.99, np.float32), mask_hud=False)[y0:y1, x0:x1].all()


def test_hud_box_reference_geometry():
    assert hud_box(364, 720) == (0, 364 - 35, 95, 364)


def _msg(arr, encoding, pad=0):
    h, w = arr.shape[:2]
    c = 1 if arr.ndim == 2 else arr.shape[2]
    step = w * c + pad
    buf = np.zeros((h, step), np.uint8)
    buf[:, :w * c] = arr.reshape(h, w * c)
    return SimpleNamespace(height=h, width=w, step=step, encoding=encoding, data=buf.tobytes())


@pytest.mark.parametrize('encoding,channels,order', [
    ('rgb8', 3, 'rgb'), ('bgr8', 3, 'bgr'), ('mono8', 1, 'gray'), ('bgra8', 4, 'bgr')])
@pytest.mark.parametrize('pad', [0, 7])
def test_image_to_array_round_trip(encoding, channels, order, pad):
    shape = (6, 5) if channels == 1 else (6, 5, channels)
    arr = np.random.default_rng(1).integers(0, 255, shape, dtype=np.uint8)
    out, got_order = image_to_array(_msg(arr, encoding, pad))
    assert got_order == order
    np.testing.assert_array_equal(out, arr)


def test_unsupported_encoding_raises():
    with pytest.raises(ValueError):
        image_to_array(SimpleNamespace(height=1, width=1, step=2, encoding='mono16', data=b'\0\0'))


def test_mask_fields_match_mono8_contract():
    mask = np.zeros((4, 9), np.uint8)
    mask[2:, 3:6] = 1
    f = mask_fields(mask)
    assert (f['height'], f['width'], f['step'], f['encoding'], f['is_bigendian']) == (4, 9, 9, 'mono8', 0)
    # Decoded the way rover_road_assist.road_deviation_node.mask_from_image does it:
    decoded = np.frombuffer(f['data'], np.uint8).reshape(f['height'], f['step'])[:, :f['width']]
    np.testing.assert_array_equal(decoded, mask)
    assert set(np.unique(decoded)) <= {0, 1}
