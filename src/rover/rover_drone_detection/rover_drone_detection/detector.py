"""The model: an Ultralytics checkpoint (or an export of one) in, Detections out.

Anything Ultralytics can load works: the .pt from training, or an .onnx /
TensorRT .engine exported from it on the target, e.g. on the Jetson

    yolo export model=drone_yolo11s_640.pt format=engine quantize=16 imgsz=640

and then model:=/path/to/drone_yolo11s_640.engine. ultralytics is imported
lazily so the rest of the package (and its tests) does not need torch.
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    """One box in pixel coordinates of the image it was detected in."""
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    cls: int = 0
    label: str = 'waiter_fpv_drone'

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1

    @property
    def cx(self):
        return 0.5 * (self.x1 + self.x2)

    @property
    def cy(self):
        return 0.5 * (self.y1 + self.y2)


def resolve_model_path(model, models_dir):
    """`model` as given if it is a path that exists, else a file in `models_dir`."""
    path = os.path.expanduser(model)
    if os.path.isabs(path) or os.path.exists(path):
        return path
    return os.path.join(models_dir, path)


class Detector:

    def __init__(self, model_path, conf=0.25, iou=0.5, imgsz=640, device='', half=False,
                 classes=None):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'no model at {model_path} (git lfs pull?)')
        if os.path.getsize(model_path) < 1024:
            raise RuntimeError(f'{model_path} is a Git LFS pointer, run `git lfs pull`')
        if 'rtdetr' in os.path.basename(model_path).lower():
            from ultralytics import RTDETR
            self.model = RTDETR(model_path)
        else:
            from ultralytics import YOLO
            self.model = YOLO(model_path, task='detect')
        self.names = dict(self.model.names)
        self.path = model_path
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device or None    # '' = let Ultralytics pick (cuda:0 if there is one)
        self.half = half
        self.classes = list(classes) if classes else None

    def warmup(self):
        """One inference on a blank frame, so the first real one is not slow."""
        import numpy as np
        self.detect(np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8))

    def detect(self, bgr):
        """Detections in `bgr` (HxWx3 uint8, OpenCV order), in its pixels."""
        result = self.model.predict(bgr, conf=self.conf, iou=self.iou, imgsz=self.imgsz,
                                    device=self.device, quantize=16 if self.half else None,
                                    classes=self.classes,
                                    verbose=False)[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        return [Detection(float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(s), int(c),
                          self.names.get(int(c), str(int(c))))
                for b, s, c in zip(xyxy, scores, cls)]
