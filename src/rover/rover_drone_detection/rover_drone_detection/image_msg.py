"""sensor_msgs/Image <-> BGR numpy, without cv_bridge.

Humble's cv_bridge is built against NumPy 1.x and fails to import under the
NumPy 2 that the image (ZED base) and ultralytics bring. The encodings here
are what rover_cameras (ros_encoding) and Gazebo cameras publish.
"""

import cv2
import numpy as np

# encoding -> (channels, cv2 conversion to BGR or None)
ENCODINGS = {
    'bgr8': (3, None),
    'rgb8': (3, cv2.COLOR_RGB2BGR),
    'bgra8': (4, cv2.COLOR_BGRA2BGR),
    'rgba8': (4, cv2.COLOR_RGBA2BGR),
    'mono8': (1, cv2.COLOR_GRAY2BGR),
}


def to_bgr(msg):
    """HxWx3 uint8 BGR from an Image-like msg; ValueError on what it can't read."""
    if msg.encoding not in ENCODINGS:
        raise ValueError(f'unsupported encoding {msg.encoding!r}, '
                         f'expected one of {sorted(ENCODINGS)}')
    channels, code = ENCODINGS[msg.encoding]
    row = msg.width * channels
    if msg.step < row or len(msg.data) < msg.step * msg.height:
        raise ValueError(f'{msg.width}x{msg.height} {msg.encoding} needs step >= {row} and '
                         f'{msg.step * msg.height} bytes, got step {msg.step}, '
                         f'{len(msg.data)} bytes')
    buf = np.frombuffer(msg.data, dtype=np.uint8, count=msg.step * msg.height)
    # Rows may be padded to `step` (GStreamer strides): drop the padding.
    img = buf.reshape(msg.height, msg.step)[:, :row].reshape(msg.height, msg.width, channels)
    return np.ascontiguousarray(img) if code is None else cv2.cvtColor(img, code)


def bgr_fields(bgr):
    """(height, width, step, data) of a bgr8 Image for an HxWx3 uint8 array."""
    bgr = np.ascontiguousarray(bgr, dtype=np.uint8)
    h, w = bgr.shape[:2]
    return h, w, w * 3, bgr.tobytes()
