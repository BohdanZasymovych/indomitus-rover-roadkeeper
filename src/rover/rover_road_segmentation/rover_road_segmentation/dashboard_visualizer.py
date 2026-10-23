"""Rich visual dashboard for validating road segmentation & road deviation.

Renders:
  1. Semi-transparent road mask overlay (green)
  2. Scanlines used by rover_road_assist (row_fractions=[0.6, 0.7, 0.8, 0.9])
  3. Scanline road centres and vehicle target axis
  4. Lateral offset gauge ([-1.0 .. +1.0] in road half-widths)
  5. Heading error indicator and direction arrow
  6. Confidence bar & validity status badge
  7. Performance metrics (FPS, frame resolution, road area %)
"""
import cv2
import numpy as np

ROAD_COLOR_BGR = (90, 230, 110)      # Bright translucent green
TARGET_AXIS_BGR = (0, 220, 255)     # Amber/yellow vehicle centerline
SCANLINE_COLOR_BGR = (180, 180, 50)  # Cyan/blue scanline
RUN_CENTER_BGR = (0, 255, 255)      # Yellow dot for scanline road centers


def _draw_gauge(img, x, y, width, height, value, min_val=-1.0, max_val=1.0, label="LATERAL OFFSET"):
    """Draws a horizontal zero-centered gauge bar."""
    # Background box
    cv2.rectangle(img, (x, y), (x + width, y + height), (35, 35, 35), -1)
    cv2.rectangle(img, (x, y), (x + width, y + height), (80, 80, 80), 1)

    # Center mark
    cx = x + width // 2
    cv2.line(img, (cx, y), (cx, y + height), (150, 150, 150), 1)

    # Clamped value position
    clamped = max(min_val, min(max_val, value))
    norm = (clamped - min_val) / (max_val - min_val)
    vx = int(x + norm * width)

    # Color depends on magnitude
    mag = abs(clamped)
    if mag < 0.15:
        bar_color = (60, 220, 60)    # Green (centered)
    elif mag < 0.45:
        bar_color = (0, 215, 255)    # Amber
    else:
        bar_color = (40, 40, 240)    # Red (drifting off road)

    # Fill bar from center to current value
    x_start = min(cx, vx)
    x_end = max(cx, vx)
    if x_end - x_start > 1:
        cv2.rectangle(img, (x_start, y + 2), (x_end, y + height - 2), bar_color, -1)

    # Value pointer
    cv2.circle(img, (vx, y + height // 2), 4, (255, 255, 255), -1)

    # Labels
    scale = max(0.38, img.shape[0] / 1200.0)
    thick = 1
    cv2.putText(img, label, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, scale, (200, 200, 200), thick, cv2.LINE_AA)
    val_str = f"{value:+.2f} hw"
    cv2.putText(img, val_str, (x + width - 65, y - 4), cv2.FONT_HERSHEY_SIMPLEX, scale, bar_color, thick, cv2.LINE_AA)
    cv2.putText(img, "L", (x + 3, y + height - 3), cv2.FONT_HERSHEY_SIMPLEX, scale * 0.8, (140, 140, 140), 1)
    cv2.putText(img, "R", (x + width - 12, y + height - 3), cv2.FONT_HERSHEY_SIMPLEX, scale * 0.8, (140, 140, 140), 1)


def _draw_confidence_bar(img, x, y, width, height, confidence, valid):
    """Draws a vertical or horizontal confidence bar."""
    cv2.rectangle(img, (x, y), (x + width, y + height), (35, 35, 35), -1)
    cv2.rectangle(img, (x, y), (x + width, y + height), (80, 80, 80), 1)

    conf_w = int(max(0.0, min(1.0, confidence)) * (width - 4))
    if valid and confidence >= 0.5:
        color = (60, 220, 60)
    elif valid:
        color = (0, 215, 255)
    else:
        color = (60, 60, 220)

    if conf_w > 0:
        cv2.rectangle(img, (x + 2, y + 2), (x + 2 + conf_w, y + height - 2), color, -1)

    scale = max(0.38, img.shape[0] / 1200.0)
    cv2.putText(img, "CONFIDENCE", (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, scale, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(img, f"{int(confidence * 100)}%", (x + width - 35, y - 4), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def render_dashboard(frame_bgr, mask, deviation, fps=0.0, latency_ms=0.0,
                     row_fractions=(0.6, 0.7, 0.8, 0.9), target_col_fraction=0.5,
                     alpha=0.45):
    """Combines frame, mask overlay, scanline centers, and a telemetry HUD dashboard.

    Args:
        frame_bgr: (H, W, 3) uint8 image.
        mask: (H, W) uint8 binary mask with values 0 and 1.
        deviation: Deviation dataclass or object with lateral_offset, heading_error,
                   confidence, valid.
        fps: Current inference/processing frames-per-second.
        latency_ms: Processing latency in milliseconds.
        row_fractions: Scanline row positions used by rover_road_assist.
        target_col_fraction: Target vehicle centerline column fraction (default 0.5).
        alpha: Overlay blend weight.

    Returns:
        (H, W, 3) rendered BGR dashboard image.
    """
    h, w = frame_bgr.shape[:2]
    out = frame_bgr.copy()

    # 1. Overlay road mask
    road = (mask > 0)
    if road.any():
        tint = np.empty_like(out)
        tint[:] = ROAD_COLOR_BGR
        out[road] = cv2.addWeighted(out, 1.0 - alpha, tint, alpha, 0.0)[road]
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, ROAD_COLOR_BGR, max(1, h // 350), cv2.LINE_AA)

    # 2. Draw vehicle centerline (target axis)
    target_col = int(target_col_fraction * (w - 1))
    cv2.line(out, (target_col, int(h * 0.45)), (target_col, h), TARGET_AXIS_BGR, 1, cv2.LINE_AA)
    cv2.putText(out, "ROVER AXIS", (target_col - 35, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, TARGET_AXIS_BGR, 1, cv2.LINE_AA)

    # 3. Draw scanlines and detected road center points
    for frac in row_fractions:
        r = min(h - 1, int(frac * h))
        cv2.line(out, (0, r), (w, r), (80, 80, 80), 1, cv2.LINE_4)

        # Find continuous road runs in this row
        row = mask[r]
        if np.any(row > 0):
            padded = np.concatenate(([0], (row > 0).view(np.int8), [0]))
            edges = np.diff(padded)
            starts = np.flatnonzero(edges == 1)
            ends = np.flatnonzero(edges == -1)
            for s, e in zip(starts, ends):
                if e - s >= max(1, int(0.05 * w)):
                    cx = int((s + e - 1) / 2.0)
                    cv2.line(out, (s, r), (e - 1, r), (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.circle(out, (cx, r), max(3, h // 150), (0, 215, 255), -1)

    # 4. Telemetry HUD overlay box in top-left
    hud_w = min(360, int(w * 0.45))
    hud_h = 175
    hud_bg = out[10:10 + hud_h, 10:10 + hud_w]
    hud_bg[:] = (hud_bg * 0.28).astype(np.uint8)
    cv2.rectangle(out, (10, 10), (10 + hud_w, 10 + hud_h), (90, 90, 90), 1)

    # Header title
    cv2.putText(out, "ROAD ASSIST TELEMETRY", (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # Validity badge
    val_text = "TRACKING VALID" if deviation.valid else "NO ROAD / SEARCHING"
    val_color = (60, 230, 60) if deviation.valid else (60, 60, 240)
    cv2.rectangle(out, (hud_w - 130, 14), (hud_w + 5, 34), (40, 40, 40), -1)
    cv2.rectangle(out, (hud_w - 130, 14), (hud_w + 5, 34), val_color, 1)
    cv2.putText(out, val_text, (hud_w - 125, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.36, val_color, 1, cv2.LINE_AA)

    # Lateral offset gauge
    _draw_gauge(out, 20, 54, hud_w - 30, 16, deviation.lateral_offset, label="LATERAL OFFSET")

    # Heading error display
    deg = np.rad2deg(deviation.heading_error)
    heading_str = f"HEADING: {deg:+.1f} deg"
    cv2.putText(out, heading_str, (20, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)

    # Direction arrow
    if abs(deg) >= 1.0:
        dir_text = "BENDS RIGHT >>" if deg > 0 else "<< BENDS LEFT"
        dir_color = (0, 215, 255)
    else:
        dir_text = "STRAIGHT"
        dir_color = (60, 220, 60)
    cv2.putText(out, dir_text, (hud_w - 120, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.40, dir_color, 1, cv2.LINE_AA)

    # Confidence bar
    _draw_confidence_bar(out, 20, 126, hud_w - 30, 14, deviation.confidence, deviation.valid)

    # Performance stats line
    road_pct = float(mask.mean()) * 100.0
    stats_str = f"Rate: {fps:4.1f} FPS ({latency_ms:3.0f} ms) | Road: {road_pct:4.1f}% | {w}x{h}"
    cv2.putText(out, stats_str, (20, 166), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (170, 170, 170), 1, cv2.LINE_AA)

    return out
