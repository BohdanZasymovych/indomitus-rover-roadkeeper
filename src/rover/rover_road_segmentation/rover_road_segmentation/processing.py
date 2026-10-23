"""Image preprocessing and mask post-processing: numpy + OpenCV only.

No torch and no ROS, so it is unit-testable anywhere. Everything is written
for an arbitrary frame size (H, W); sizes that depend on the resolution (HUD
box, morphology kernel) are scaled from the 720x364 reference frames.
"""
import cv2
import numpy as np

from rover_road_segmentation import config as C

_CLAHE = cv2.createCLAHE(clipLimit=C.CLAHE_CLIP_LIMIT, tileGridSize=C.CLAHE_TILE_GRID)

COLOR_ORDERS = ('bgr', 'rgb', 'gray')


def to_gray(frame, color_order='bgr'):
    """uint8 grey image from a HxW, HxWx1, HxWx3 or HxWx4 frame."""
    if color_order not in COLOR_ORDERS:
        raise ValueError(f'color_order must be one of {COLOR_ORDERS}, got {color_order!r}')
    frame = np.asarray(frame)
    if frame.dtype != np.uint8:
        raise TypeError(f'expected a uint8 frame, got {frame.dtype}')
    if frame.ndim == 2:
        return frame
    if frame.ndim != 3:
        raise ValueError(f'expected a 2-D or 3-D frame, got shape {frame.shape}')
    channels = frame.shape[2]
    if channels == 1:
        return frame[:, :, 0]
    if channels == 4:
        frame = frame[:, :, :3]
    elif channels != 3:
        raise ValueError(f'unsupported channel count {channels}')
    code = cv2.COLOR_RGB2GRAY if color_order == 'rgb' else cv2.COLOR_BGR2GRAY
    return cv2.cvtColor(np.ascontiguousarray(frame), code)


def preprocess(frame, color_order='bgr'):
    """Any-size frame -> (CLAHE'd 3-channel uint8 image at INFER size, mean grey level).

    The camera is grayscale IR, so the frame is reduced to grey, resized to the
    fixed DINOv2 inference size and contrast-equalised with CLAHE (wheel ruts
    and track edges stand out from grass). Resizing *before* CLAHE makes the
    equalisation identical for every source resolution. The grey channel is
    replicated to 3 channels for the ImageNet-pretrained backbone.
    """
    gray = to_gray(frame, color_order)
    if gray.shape != (C.INFER_HEIGHT, C.INFER_WIDTH):
        interp = cv2.INTER_AREA if gray.shape[0] > C.INFER_HEIGHT else cv2.INTER_LINEAR
        gray = cv2.resize(gray, (C.INFER_WIDTH, C.INFER_HEIGHT), interpolation=interp)
    brightness = float(gray.mean())
    gray = _CLAHE.apply(gray)
    return cv2.merge([gray, gray, gray]), brightness


def hud_box(height, width):
    """(x0, y0, x1, y1) of the bottom-left HUD box scaled to a (height, width) frame."""
    x1 = int(round(C.HUD_WIDTH_PX * width / C.REF_WIDTH))
    y0 = height - int(round(C.HUD_HEIGHT_PX * height / C.REF_HEIGHT))
    return 0, max(0, y0), min(width, x1), height


def morph_kernel_size(height, ref=C.MORPH_KERNEL_REF):
    """Odd kernel size, `ref` pixels at REF_HEIGHT scaled to `height`."""
    k = max(3, int(round(ref * height / C.REF_HEIGHT)))
    return k if k % 2 else k + 1


def clean_mask(prob, threshold=C.ROAD_PROB_THRESHOLD, mask_hud=True,
               morph_kernel_ref=C.MORPH_KERNEL_REF, lower_half_start=C.LOWER_HALF_START,
               min_component_fraction=C.MIN_COMPONENT_FRACTION):
    """p(road) map (H, W) -> uint8 (H, W) mask with values strictly 0 and 1.

    1. strict threshold (p > threshold, not argmax),
    2. the HUD box is never road,
    3. morphological open (drops speckle) then close (fills ruts/shadows),
    4. only the connected component with the largest area in the lower part
       of the frame is kept (the road the vehicle stands on); if even that is
       smaller than min_component_fraction of the lower part, nothing is road.
    """
    prob = np.asarray(prob)
    height, width = prob.shape
    mask = (prob > threshold).astype(np.uint8)

    if mask_hud:
        x0, y0, x1, y1 = hud_box(height, width)
        mask[y0:y1, x0:x1] = 0

    if morph_kernel_ref > 0:
        k = morph_kernel_size(height, morph_kernel_ref)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        if mask_hud:  # closing may have grown back into the box
            mask[y0:y1, x0:x1] = 0

    n, labels = cv2.connectedComponents(mask, connectivity=8)
    if n <= 1:
        return np.zeros((height, width), np.uint8)

    top = int(height * lower_half_start)
    lower_area = np.bincount(labels[top:].ravel(), minlength=n)
    lower_area[0] = 0
    best = int(np.argmax(lower_area))
    min_area = min_component_fraction * (height - top) * width
    if lower_area[best] < max(1.0, min_area):
        return np.zeros((height, width), np.uint8)
    return (labels == best).astype(np.uint8)
