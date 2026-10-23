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
    temporal_alpha: float = C.TEMPORAL_ALPHA
    mask_hud: bool = True
    morph_kernel_ref: int = C.MORPH_KERNEL_REF
    lower_half_start: float = C.LOWER_HALF_START
    min_component_fraction: float = C.MIN_COMPONENT_FRACTION
    min_frame_brightness: float = C.MIN_FRAME_BRIGHTNESS
    bridge_kernel_width: int = C.BRIDGE_KERNEL_WIDTH
    bridge_kernel_height: int = C.BRIDGE_KERNEL_HEIGHT
    vertical_kernel_height: int = C.VERTICAL_KERNEL_HEIGHT
    vertical_kernel_width: int = C.VERTICAL_KERNEL_WIDTH
    enable_convex_span_fill: bool = C.ENABLE_CONVEX_SPAN_FILL
    max_span_gap_fraction: float = C.MAX_SPAN_GAP_FRACTION
    span_fill_start_y: float = C.SPAN_FILL_START_Y

    def clean_kwargs(self):
        skip = {'min_frame_brightness', 'temporal_alpha'}
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name not in skip}


def resolve_device(device='auto'):
    if device in (None, '', 'auto'):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device)


def to_tensor(image):
    """uint8 HxWx3 -> float (1, 3, H, W) in [0, 1]."""
    return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).unsqueeze(0).float().div_(255.0)


class RoadSegmenter:

    def __init__(self, weights=C.DEFAULT_WEIGHTS, device='auto', use_fp16=True,
                 post=None, model=None, infer_size=None):
        self.device = resolve_device(device)
        if model is None:
            model, self.meta = load_model(weights, self.device)
        else:
            model, self.meta = model.to(self.device).eval(), {}
        self.model = model
        self.use_fp16 = bool(use_fp16) and self.device.type == 'cuda'
        self.post = post or PostProcessParams()
        self._prev_prob = None
        self._prev_shape = None
        self.last_profile = {}

        # Resolve inference resolution
        if infer_size is not None:
            w, h = infer_size
        elif 'infer_size' in self.meta:
            # Checkpoint meta may store (H, W) or (W, H)
            meta_sz = self.meta['infer_size']
            if meta_sz[0] > meta_sz[1]:
                w, h = meta_sz[0], meta_sz[1]
            else:
                h, w = meta_sz[0], meta_sz[1]
        else:
            w, h = C.INFER_WIDTH, C.INFER_HEIGHT

        # Ensure multiples of patch size
        self.infer_width = max(C.DINOV2_PATCH_SIZE, (w // C.DINOV2_PATCH_SIZE) * C.DINOV2_PATCH_SIZE)
        self.infer_height = max(C.DINOV2_PATCH_SIZE, (h // C.DINOV2_PATCH_SIZE) * C.DINOV2_PATCH_SIZE)

        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = getattr(C, 'TORCH_CUDNN_BENCHMARK', True)
            if hasattr(torch, 'set_float32_matmul_precision'):
                torch.set_float32_matmul_precision(getattr(C, 'TORCH_MATMUL_PRECISION', 'high'))

    def reset(self):
        """Resets temporal smoothing history (e.g. on seek, track switch, or outage)."""
        self._prev_prob = None
        self._prev_shape = None

    def _predict_raw_infer_prob(self, frame, color_order='bgr'):
        """Computes p(road) at inference resolution. Returns (prob, brightness)."""
        image, brightness = preprocess(frame, color_order, (self.infer_width, self.infer_height))
        if brightness < self.post.min_frame_brightness:
            self.reset()
            return None, brightness

        if self.device.type == 'cuda':
            t = torch.from_numpy(image).to(self.device, non_blocking=True)
            dtype = torch.float16 if self.use_fp16 else torch.float32
            x = t.permute(2, 0, 1).unsqueeze(0).to(dtype=dtype).div_(255.0)
        else:
            x = to_tensor(image).to(self.device)

        with torch.inference_mode():
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_fp16):
                logits = self.model(x)
            prob = torch.sigmoid(logits.float())[0, 0].cpu().numpy()
        return prob, brightness

    def _smooth_prob(self, prob_infer, frame_shape):
        """EMA temporal smoothing on the inference-resolution probability map."""
        if self._prev_prob is None or self._prev_shape != frame_shape or self.post.temporal_alpha >= 1.0:
            smoothed = prob_infer
        else:
            alpha = self.post.temporal_alpha
            smoothed = alpha * prob_infer + (1.0 - alpha) * self._prev_prob
        self._prev_prob = smoothed
        self._prev_shape = frame_shape
        return smoothed

    @torch.no_grad()
    def predict_proba(self, frame, color_order='bgr'):
        """p(road), float32 (H_orig, W_orig), or None for a frame with no video."""
        return self.segment_with_proba(frame, color_order)[1]

    def segment_with_proba(self, frame, color_order='bgr', profile=False):
        """(mask uint8 0/1 (H_orig, W_orig), p(road) float32 (H_orig, W_orig) or None)."""
        import cv2
        import time
        from rover_road_segmentation.processing import hud_box

        t0 = time.perf_counter() if profile else 0.0
        h_orig, w_orig = frame.shape[:2]

        if profile:
            t_pre0 = time.perf_counter()
            image, brightness = preprocess(frame, color_order, (self.infer_width, self.infer_height))
            if brightness < self.post.min_frame_brightness:
                self.reset()
                return np.zeros((h_orig, w_orig), np.uint8), None
            if self.device.type == 'cuda':
                t = torch.from_numpy(image).to(self.device, non_blocking=True)
                dtype = torch.float16 if self.use_fp16 else torch.float32
                x = t.permute(2, 0, 1).unsqueeze(0).to(dtype=dtype).div_(255.0)
                torch.cuda.synchronize()
            else:
                x = to_tensor(image).to(self.device)
            t_pre1 = time.perf_counter()

            t_inf0 = time.perf_counter()
            with torch.inference_mode():
                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_fp16):
                    logits = self.model(x)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                prob_infer = torch.sigmoid(logits.float())[0, 0].cpu().numpy()
            t_inf1 = time.perf_counter()
        else:
            prob_infer, brightness = self._predict_raw_infer_prob(frame, color_order)
            if prob_infer is None:
                return np.zeros((h_orig, w_orig), np.uint8), None

        # Step 1: Temporal smoothing at inference resolution
        t_post0 = time.perf_counter() if profile else 0.0
        smoothed_infer = self._smooth_prob(prob_infer, (h_orig, w_orig))

        # Steps 2 & 3: Rut bridging, hole filling, and scanline span stabilization
        clean_args = self.post.clean_kwargs()
        clean_args['mask_hud'] = False
        mask_infer = clean_mask(smoothed_infer, **clean_args)

        # Step 4: Upsample regularized mask back to original camera resolution
        if (h_orig, w_orig) != mask_infer.shape:
            mask_orig = cv2.resize(mask_infer, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        else:
            mask_orig = mask_infer

        # Blank out bottom-left HUD box at original frame resolution
        if self.post.mask_hud:
            x0, y0, x1, y1 = hud_box(h_orig, w_orig)
            mask_orig[y0:y1, x0:x1] = 0

        # Full-resolution smoothed probability map for debug/evaluation
        prob_full = cv2.resize(smoothed_infer, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

        if profile:
            t_post1 = time.perf_counter()
            self.last_profile = {
                't_pre_ms': (t_pre1 - t_pre0) * 1000.0,
                't_infer_ms': (t_inf1 - t_inf0) * 1000.0,
                't_post_ms': (t_post1 - t_post0) * 1000.0,
                't_total_ms': (t_post1 - t0) * 1000.0,
                'fps': 1.0 / max(1e-6, t_post1 - t0),
            }

        return mask_orig.astype(np.uint8), prob_full

    def segment(self, frame, color_order='bgr', profile=False):
        """Binary road mask of `frame`: uint8, shape frame.shape[:2], values strictly 0 and 1."""
        return self.segment_with_proba(frame, color_order, profile=profile)[0]
