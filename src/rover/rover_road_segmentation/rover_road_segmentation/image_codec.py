"""sensor_msgs/Image <-> numpy without cv_bridge (and without importing ROS).

Works on anything with height/width/step/encoding/data attributes, so it can be
unit-tested with a plain namespace object.
"""
import numpy as np

#: encoding -> (channels, color order understood by processing.to_gray)
_ENCODINGS = {
    'rgb8': (3, 'rgb'), '8UC3': (3, 'bgr'), 'bgr8': (3, 'bgr'),
    'rgba8': (4, 'rgb'), 'bgra8': (4, 'bgr'), '8UC4': (4, 'bgr'),
    'mono8': (1, 'gray'), '8UC1': (1, 'gray'),
}
SUPPORTED_ENCODINGS = tuple(_ENCODINGS)


def image_to_array(msg):
    """(uint8 HxW or HxWxC array, color order) of an 8-bit Image message.

    Honours `step` (row padding). The array is a read-only view of msg.data.
    """
    try:
        channels, order = _ENCODINGS[msg.encoding]
    except KeyError:
        raise ValueError(f'unsupported image encoding {msg.encoding!r}, '
                         f'expected one of {SUPPORTED_ENCODINGS}') from None
    h, w, step = int(msg.height), int(msg.width), int(msg.step)
    if step < w * channels:
        raise ValueError(f'step {step} < width {w} * {channels} channels')
    data = np.frombuffer(msg.data, dtype=np.uint8)
    if data.size < h * step:
        raise ValueError(f'image data has {data.size} bytes, expected {h * step}')
    rows = data[:h * step].reshape(h, step)[:, :w * channels]
    if channels == 1:
        return rows, order
    return rows.reshape(h, w, channels), order


def mask_fields(mask):
    """Fields of a mono8 Image for a (H, W) uint8 0/1 mask: height, width, step, data."""
    mask = np.ascontiguousarray(mask, dtype=np.uint8)
    if mask.ndim != 2:
        raise ValueError(f'mask must be 2-D, got shape {mask.shape}')
    h, w = mask.shape
    return {'height': h, 'width': w, 'encoding': 'mono8', 'is_bigendian': 0,
            'step': w, 'data': mask.tobytes()}
