"""The operator's picture: a box and score around every drone."""

import cv2

RED = (0, 0, 255)  # BGR


def annotate(bgr, dets):
    """A copy of `bgr` with a box and label for each of `dets` drawn on it."""
    img = bgr.copy()
    h, w = img.shape[:2]
    thick = max(2, round(min(w, h) / 300))
    scale = max(0.5, min(w, h) / 900)

    for d in dets:
        p1, p2 = (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2))
        cv2.rectangle(img, p1, p2, RED, thick)
        text = f'drone {d.score:.2f}'
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        # Label above the box, or below it when the box touches the top edge.
        ty = p1[1] - 4 if p1[1] - th - base - 4 > 0 else p2[1] + th + 4
        cv2.rectangle(img, (p1[0], ty - th - base), (p1[0] + tw + 4, ty + base), RED, -1)
        cv2.putText(img, text, (p1[0] + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return img
