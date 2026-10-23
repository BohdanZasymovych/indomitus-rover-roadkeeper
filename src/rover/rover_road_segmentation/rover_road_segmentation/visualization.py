"""Human-visible renderings of a 0/1 road mask (a raw mono8 0/1 image looks black)."""
import cv2
import numpy as np

ROAD_COLOR_BGR = (90, 255, 120)


def overlay(frame_bgr, mask, alpha=0.45, color=ROAD_COLOR_BGR):
    """Frame with the road tinted green and its outline drawn; same size as the frame."""
    out = frame_bgr.copy() if frame_bgr.ndim == 3 else cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)
    road = mask.astype(bool)
    if road.any():
        tint = np.empty_like(out)
        tint[:] = color
        out[road] = cv2.addWeighted(out, 1.0 - alpha, tint, alpha, 0.0)[road]
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        thickness = max(1, round(out.shape[0] / 300))
        cv2.drawContours(out, contours, -1, color, thickness, cv2.LINE_AA)
    return out


def info_panel(image, lines, origin=(8, 8)):
    """Draws `lines` of text on a translucent box in the top-left corner, in place."""
    if not lines:
        return image
    scale = max(0.45, image.shape[0] / 900)
    thick = max(1, round(scale * 2))
    line_h = int(22 * scale / 0.6)
    width = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0][0] for t in lines)
    x0, y0 = origin
    x1, y1 = x0 + width + 16, y0 + line_h * len(lines) + 10
    roi = image[y0:y1, x0:x1]
    roi[:] = (roi * 0.35).astype(np.uint8)
    for i, text in enumerate(lines):
        y = y0 + 6 + line_h * (i + 1) - line_h // 4
        cv2.putText(image, text, (x0 + 8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return image


def draw_scanlines_and_deviation(img, mask, row_fractions=(0.6, 0.7, 0.8, 0.9),
                                 target_col_fraction=0.5, estimate_dev=True):
    """Draws rover_road_assist scanlines, detected spans, and road centers.

    Optionally calculates and returns Deviation from rover_road_assist.
    """
    h, w = mask.shape[:2]
    target_col = int(target_col_fraction * (w - 1))

    # Rover vehicle centerline axis
    cv2.line(img, (target_col, int(h * 0.45)), (target_col, h), (0, 220, 255), 1, cv2.LINE_AA)

    # 4 scanlines used by rover_road_assist
    min_width = max(1, int(0.05 * w))
    for frac in row_fractions:
        r = min(h - 1, int(frac * h))
        cv2.line(img, (0, r), (w, r), (70, 70, 70), 1, cv2.LINE_4)

        row = mask[r]
        if np.any(row > 0):
            padded = np.concatenate(([0], (row > 0).view(np.int8), [0]))
            edges = np.diff(padded)
            starts = np.flatnonzero(edges == 1)
            ends = np.flatnonzero(edges == -1)
            for s, e in zip(starts, ends):
                span = e - s
                if span >= min_width:
                    cx = int((s + e - 1) / 2.0)
                    cv2.line(img, (s, r), (e - 1, r), (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.circle(img, (s, r), 3, (0, 200, 255), -1)
                    cv2.circle(img, (e - 1, r), 3, (0, 200, 255), -1)
                    cv2.circle(img, (cx, r), 4, (0, 255, 255), -1)

    dev = None
    if estimate_dev:
        try:
            from rover_road_assist.road_deviation import estimate_deviation
            dev = estimate_deviation(mask, row_fractions=row_fractions, target_col_fraction=target_col_fraction)
        except Exception:
            try:
                import sys
                from pathlib import Path
                assist_dir = Path(__file__).resolve().parent.parent.parent / 'rover_road_assist'
                if str(assist_dir) not in sys.path and assist_dir.exists():
                    sys.path.insert(0, str(assist_dir))
                from rover_road_assist.road_deviation import estimate_deviation
                dev = estimate_deviation(mask, row_fractions=row_fractions, target_col_fraction=target_col_fraction)
            except Exception:
                pass
    return dev


def side_by_side(frame_bgr, mask, lines=None, alpha=0.45, with_scanlines=True):
    """[overlay | mask*255] next to each other, with scanlines and info panel on overlay."""
    over = overlay(frame_bgr, mask, alpha)
    if with_scanlines:
        draw_scanlines_and_deviation(over, mask, estimate_dev=False)
    left = info_panel(over, lines)
    right = cv2.cvtColor(mask.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    return np.hstack([left, right])
