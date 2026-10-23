"""Boxes drawn on the annotated image, and the Detection they come from."""

import numpy as np

from rover_drone_detection.detector import Detection, resolve_model_path
from rover_drone_detection.draw import RED, annotate

W, H = 320, 240


def blank():
    return np.zeros((H, W, 3), np.uint8)


def test_no_detections_is_an_unchanged_copy():
    img = blank()
    out = annotate(img, [])
    assert out is not img and np.array_equal(out, img)


def test_box_is_drawn_and_input_untouched():
    img = blank()
    out = annotate(img, [Detection(100, 100, 200, 180, 0.6)])
    assert not img.any()
    assert tuple(out[140, 100]) == RED and tuple(out[140, 200]) == RED   # left and right edges
    assert not out[140, 150].any()                                       # inside stays clear


def test_box_at_the_top_edge_still_gets_its_label():
    out = annotate(blank(), [Detection(10, 0, 60, 40, 0.5)])
    assert out[43:80].any()   # label goes below the box


def test_detection_geometry():
    d = Detection(10, 20, 50, 80, 0.9)
    assert (d.width, d.height, d.cx, d.cy) == (40, 60, 30, 50)


def test_model_name_is_looked_up_in_models_dir(tmp_path):
    assert resolve_model_path('m.pt', '/share/models') == '/share/models/m.pt'
    real = tmp_path / 'x.engine'
    real.write_bytes(b'')
    assert resolve_model_path(str(real), '/share/models') == str(real)
