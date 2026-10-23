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


def preprocess(frame, color_order='bgr', target_size=None):
    """Any-size frame -> (CLAHE'd 3-channel uint8 image at INFER size, mean grey level).

    The camera is grayscale IR, so the frame is reduced to grey, resized to the
    inference size and contrast-equalised with CLAHE (wheel ruts and track edges
    stand out from grass). Resizing *before* CLAHE makes the equalisation identical
    for every source resolution and runs fast (< 0.5 ms).
    """
    gray = to_gray(frame, color_order)
    target_w = target_size[0] if target_size is not None else C.INFER_WIDTH
    target_h = target_size[1] if target_size is not None else C.INFER_HEIGHT
    if gray.shape != (target_h, target_w):
        interp = cv2.INTER_AREA if gray.shape[0] > target_h else cv2.INTER_LINEAR
        gray = cv2.resize(gray, (target_w, target_h), interpolation=interp)
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


def bridge_ruts(mask, kw_horiz=C.BRIDGE_KERNEL_WIDTH, kh_horiz=C.BRIDGE_KERNEL_HEIGHT,
                kw_vert=C.VERTICAL_KERNEL_WIDTH, kh_vert=C.VERTICAL_KERNEL_HEIGHT,
                start_y_fraction=C.SPAN_FILL_START_Y):
    """Bridge center grass strip between left/right wheel ruts via directional closing."""
    h, w = mask.shape
    out = mask.copy()
    y_start = int(start_y_fraction * h)

    # 1. Horizontal bridging closing on lower portion of the image
    if kw_horiz > 1 and kh_horiz > 1:
        kw = max(3, int(round(kw_horiz * w / C.INFER_WIDTH)))
        kh = max(1, int(round(kh_horiz * h / C.INFER_HEIGHT)))
        k_horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
        out[y_start:] = cv2.morphologyEx(out[y_start:], cv2.MORPH_CLOSE, k_horiz)

    # 2. Vertical continuity closing across whole image
    if kw_vert > 1 and kh_vert > 1:
        kw_v = max(1, int(round(kw_vert * w / C.INFER_WIDTH)))
        kh_v = max(3, int(round(kh_vert * h / C.INFER_HEIGHT)))
        k_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (kw_v, kh_v))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k_vert)

    return out


def fill_contour_holes_and_keep_largest(mask, lower_half_start=C.LOWER_HALF_START,
                                        min_component_fraction=C.MIN_COMPONENT_FRACTION):
    """Keep only primary road corridor and fill all interior holes (puddles, shadows)."""
    h, w = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros((h, w), dtype=np.uint8)

    top = int(h * lower_half_start)
    best_cnt = None
    best_score = 0.0

    # Fast evaluation of lower-half road area without allocating masks
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if y + bh <= top:
            continue
        area = cv2.contourArea(cnt)
        if area <= best_score:
            continue
        if y >= top:
            score = area
        else:
            score = area * ((y + bh - top) / float(bh))
        if score > best_score:
            best_score = score
            best_cnt = cnt

    min_area = min_component_fraction * (h - top) * w
    if best_cnt is None or best_score < max(1.0, min_area):
        return np.zeros((h, w), dtype=np.uint8)

    # Draw the single best external contour filled -> every interior hole/shadow is 1
    clean_out = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(clean_out, [best_cnt], -1, 1, thickness=cv2.FILLED)
    return clean_out


def stabilize_scanline_spans(mask, start_y_fraction=C.SPAN_FILL_START_Y,
                             max_gap_fraction=C.MAX_SPAN_GAP_FRACTION):
    """Row-wise span fill: fills gap between leftmost and rightmost road pixels in each row.

    Guarantees every horizontal scanline used by rover_road_assist has strictly ONE
    contiguous road run with full road half_width, eliminating inverted-V center notches.
    Pure vectorized NumPy implementation without Python for-loops (< 0.2 ms).
    """
    h, w = mask.shape
    out = mask.copy()
    y_start = int(start_y_fraction * h)
    max_gap = int(max_gap_fraction * w)

    roi = out[y_start:]
    has_road = roi.any(axis=1)
    if not has_road.any():
        return out

    first_col = roi.argmax(axis=1)
    last_col = w - 1 - roi[:, ::-1].argmax(axis=1)

    span_width = last_col - first_col + 1
    road_counts = roi.sum(axis=1)
    gap = span_width - road_counts

    valid_rows = has_road & (gap > 0) & (gap <= max_gap) & (first_col < last_col)
    if not valid_rows.any():
        return out

    cols = np.arange(w)[None, :]
    fill_mask = valid_rows[:, None] & (cols >= first_col[:, None]) & (cols <= last_col[:, None])
    roi[fill_mask] = 1
    out[y_start:] = roi
    return out


def clean_mask(prob, threshold=C.ROAD_PROB_THRESHOLD, mask_hud=True,
               morph_kernel_ref=C.MORPH_KERNEL_REF, lower_half_start=C.LOWER_HALF_START,
               min_component_fraction=C.MIN_COMPONENT_FRACTION,
               bridge_kernel_width=C.BRIDGE_KERNEL_WIDTH,
               bridge_kernel_height=C.BRIDGE_KERNEL_HEIGHT,
               vertical_kernel_height=C.VERTICAL_KERNEL_HEIGHT,
               vertical_kernel_width=C.VERTICAL_KERNEL_WIDTH,
               enable_convex_span_fill=C.ENABLE_CONVEX_SPAN_FILL,
               max_span_gap_fraction=C.MAX_SPAN_GAP_FRACTION,
               span_fill_start_y=C.SPAN_FILL_START_Y):
    """p(road) map (H, W) -> regularized uint8 (H, W) mask strictly with values 0 and 1.

    1. strict threshold (p > threshold),
    2. light morphological opening to drop isolated noise speckles,
    3. horizontal rut bridging closing + vertical continuity closing,
    4. keep primary road corridor & fill all interior holes (puddles/shadows) via RETR_EXTERNAL,
    5. row-wise span stabilization to eliminate inverted-V bottom grass notches,
    6. blank HUD box if mask_hud is True.
    """
    prob = np.asarray(prob)
    height, width = prob.shape
    mask = (prob > threshold).astype(np.uint8)

    # Light opening to drop isolated noise speckles
    if morph_kernel_ref > 0:
        k = max(3, min(5, morph_kernel_size(height, morph_kernel_ref)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Horizontal rut bridging + vertical continuity closing
    mask = bridge_ruts(
        mask,
        kw_horiz=bridge_kernel_width,
        kh_horiz=bridge_kernel_height,
        kw_vert=vertical_kernel_width,
        kh_vert=vertical_kernel_height,
        start_y_fraction=span_fill_start_y,
    )

    # Primary road contour selection + solid hole-filling
    mask = fill_contour_holes_and_keep_largest(
        mask,
        lower_half_start=lower_half_start,
        min_component_fraction=min_component_fraction,
    )

    # Scanline span stabilization
    if enable_convex_span_fill and mask.any():
        mask = stabilize_scanline_spans(
            mask,
            start_y_fraction=span_fill_start_y,
            max_gap_fraction=max_span_gap_fraction,
        )

    # Zero out the HUD box
    if mask_hud:
        x0, y0, x1, y1 = hud_box(height, width)
        mask[y0:y1, x0:x1] = 0

    return mask.astype(np.uint8)
