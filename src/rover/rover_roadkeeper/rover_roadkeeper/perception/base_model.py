"""
Swappable road-segmenter interface + the preprocessing shared by training and inference.

Every concrete segmenter:
  * takes a float tensor (B, 3, FRAME_HEIGHT, FRAME_WIDTH) in [0, 1] (already CLAHE'd),
  * returns road logits (B, 1, FRAME_HEIGHT, FRAME_WIDTH).
predict_mask(frame_bgr) wraps that into a single-call numpy API returning p(road).
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional, Union

import cv2
import numpy as np
import torch
import torch.nn as nn

from rover_roadkeeper.config import (
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID,
    FRAME_HEIGHT,
    FRAME_WIDTH,
)

_CLAHE = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)


def preprocess_ir(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Resize to (FRAME_WIDTH, FRAME_HEIGHT), convert to grayscale and apply CLAHE so wheel
    ruts / road edges stand out from the field. Returns a 3-channel uint8 RGB image
    (gray replicated) so ImageNet-pretrained backbones can consume it.
    """
    if frame_bgr.shape[:2] != (FRAME_HEIGHT, FRAME_WIDTH):
        frame_bgr = cv2.resize(frame_bgr, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)
    gray = frame_bgr if frame_bgr.ndim == 2 else cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = _CLAHE.apply(gray)
    return cv2.merge([gray, gray, gray])


def to_tensor(image_rgb: np.ndarray) -> torch.Tensor:
    """uint8 HxWx3 -> float (3, H, W) in [0, 1]."""
    return torch.from_numpy(np.ascontiguousarray(image_rgb)).permute(2, 0, 1).float().div_(255.0)


class BaseRoadSegmenter(nn.Module, ABC):
    """Abstract binary road segmenter. Subclasses implement forward() and set ARCH."""

    ARCH: str = "base"

    def __init__(self):
        super().__init__()
        # ImageNet normalisation lives inside the model so callers only pass [0, 1] tensors.
        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.img_mean) / self.img_std

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) in [0, 1] -> road logits (B, 1, H, W)."""

    # --- checkpoint helpers -------------------------------------------------
    def trainable_state_dict(self) -> Dict[str, torch.Tensor]:
        """Only what is trained (frozen backbones are re-downloaded, not stored)."""
        return self.state_dict()

    def load_trainable_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self.load_state_dict(state, strict=True)

    # --- single-frame inference --------------------------------------------
    @torch.no_grad()
    def predict_mask(self, frame_bgr: np.ndarray, device: Optional[Union[str, torch.device]] = None) -> np.ndarray:
        """
        Any-resolution BGR frame -> float32 road probability map (FRAME_HEIGHT, FRAME_WIDTH).
        """
        device = device or next(self.parameters()).device
        x = to_tensor(preprocess_ir(frame_bgr)).unsqueeze(0).to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(torch.device(device).type == "cuda")):
            logits = self.forward(x)
        return torch.sigmoid(logits.float())[0, 0].cpu().numpy()
