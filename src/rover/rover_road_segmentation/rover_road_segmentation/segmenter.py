"""The segmentation core: any-size frame in, (H, W) uint8 0/1 road mask out.

No ROS here; segmentation_node.py, test_segmentation_video.py and
train_dinov2.py all go through RoadSegmenter so they behave identically.

    seg = RoadSegmenter('weights/road_segmenter_dinov2.pt')
    mask = seg.segment(frame_bgr)                 # (H, W) uint8, values 0/1
    mask = seg.segment(frame_rgb, 'rgb')          # e.g. a ROS rgb8 image
"""
from dataclasses import dataclass, fields

import numpy as np
import torch
import torch.nn.functional as F

from rover_road_segmentation import config as C
from rover_road_segmentation.model import load_model
from rover_road_segmentation.processing import clean_mask, preprocess


@dataclass
class PostProcessParams:
    threshold: float = C.ROAD_PROB_THRESHOLD
    mask_hud: bool = True
    morph_kernel_ref: int = C.MORPH_KERNEL_REF
    lower_half_start: float = C.LOWER_HALF_START
    min_component_fraction: float = C.MIN_COMPONENT_FRACTION
    min_frame_brightness: float = C.MIN_FRAME_BRIGHTNESS

    def clean_kwargs(self):
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name != 'min_frame_brightness'}


def resolve_device(device='auto'):
    if device in (None, '', 'auto'):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device)


def to_tensor(image):
    """uint8 HxWx3 -> float (1, 3, H, W) in [0, 1]."""
    return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).unsqueeze(0).float().div_(255.0)


class RoadSegmenter:

    def __init__(self, weights=C.DEFAULT_WEIGHTS, device='auto', use_fp16=True,
                 post=None, model=None):
        self.device = resolve_device(device)
        if model is None:
            model, self.meta = load_model(weights, self.device)
        else:
            model, self.meta = model.to(self.device).eval(), {}
        self.model = model
        self.use_fp16 = bool(use_fp16) and self.device.type == 'cuda'
        self.post = post or PostProcessParams()

    @torch.no_grad()
    def predict_proba(self, frame, color_order='bgr'):
        """p(road), float32 (H_orig, W_orig), or None for a frame with no video."""
        h, w = frame.shape[:2]
        image, brightness = preprocess(frame, color_order)
        if brightness < self.post.min_frame_brightness:
            return None
        x = to_tensor(image).to(self.device, non_blocking=True)
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_fp16):
            logits = self.model(x)
        # Upsample the logits straight to the original frame, then squash.
        logits = F.interpolate(logits.float(), size=(h, w), mode='bilinear', align_corners=False)
        return torch.sigmoid(logits)[0, 0].cpu().numpy()

    def segment_with_proba(self, frame, color_order='bgr'):
        """(mask uint8 0/1 (H, W), p(road) float32 (H, W) or None)."""
        prob = self.predict_proba(frame, color_order)
        if prob is None:
            return np.zeros(frame.shape[:2], np.uint8), None
        return clean_mask(prob, **self.post.clean_kwargs()), prob

    def segment(self, frame, color_order='bgr'):
        """Binary road mask of `frame`: uint8, shape frame.shape[:2], values 0 and 1."""
        return self.segment_with_proba(frame, color_order)[0]
