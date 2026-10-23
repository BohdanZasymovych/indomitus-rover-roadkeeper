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


def side_by_side(frame_bgr, mask, lines=None, alpha=0.45):
    """[overlay | mask*255] next to each other, with the info panel on the overlay."""
    left = info_panel(overlay(frame_bgr, mask, alpha), lines)
    right = cv2.cvtColor(mask.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    return np.hstack([left, right])
