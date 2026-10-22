"""The operator's picture: boxes around the drones and a banner with the level.

Box colours: grey = seen but not confirmed yet, orange = WARN, red = DANGER.
The path zone of ThreatParams is outlined faintly so it is obvious why a box
counts as DANGER.
"""

import cv2

from rover_drone_detection.threat import Level, ThreatParams, frame_level

GREY = (160, 160, 160)
COLOURS = {Level.CLEAR: GREY, Level.WARN: (0, 165, 255), Level.DANGER: (0, 0, 255)}  # BGR


def annotate(bgr, dets, state, params: ThreatParams, draw_path_zone=True):
    """A copy of `bgr` with `dets` and the banner for `state` drawn on it."""
    img = bgr.copy()
    h, w = img.shape[:2]
    thick = max(2, round(min(w, h) / 300))
    scale = max(0.5, min(w, h) / 900)

    if draw_path_zone:
        x0 = int(w * (0.5 - params.path_half_width))
        x1 = int(w * (0.5 + params.path_half_width))
        y0 = int(h * params.horizon)
        cv2.rectangle(img, (x0, y0), (x1, h - 1), GREY, 1)

    for d in dets:
        level = frame_level(d, w, h, params) if state.level != Level.CLEAR else Level.CLEAR
        colour = COLOURS[level]
        p1, p2 = (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2))
        cv2.rectangle(img, p1, p2, colour, thick + (2 if d is state.target else 0))
        text = f'drone {d.score:.2f}'
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        ty = p1[1] - 4 if p1[1] - th - base - 4 > 0 else p2[1] + th + 4
        cv2.rectangle(img, (p1[0], ty - th - base), (p1[0] + tw + 4, ty + base), colour, -1)
        cv2.putText(img, text, (p1[0] + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (0, 0, 0), 1, cv2.LINE_AA)

    if state.level != Level.CLEAR:
        text = f'DRONE {state.level.name}  x{state.count}'
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale * 1.4, 2)
        x = (w - tw) // 2
        cv2.rectangle(img, (x - 10, 0), (x + tw + 10, th + base + 14), COLOURS[state.level], -1)
        cv2.putText(img, text, (x, th + 7), cv2.FONT_HERSHEY_SIMPLEX, scale * 1.4,
                    (255, 255, 255), 2, cv2.LINE_AA)
    return img
