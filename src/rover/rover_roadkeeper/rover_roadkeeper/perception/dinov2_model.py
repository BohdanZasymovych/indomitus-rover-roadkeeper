"""
Frozen DINOv2 ViT-S/14 + lightweight binary road head.

Design notes
  * Backbone is frozen (only ~0.4M head params are trained) -> no overfitting to the single
    301-frame forest ride, generalises to the dev_ videos.
  * Patch tokens of the last DINOV2_NUM_LAYERS blocks are concatenated (richer than last-only).
  * Patch grid is 26x51 (14 px cells), which blurs road edges. A tiny conv branch on the
    full-resolution CLAHE image is fused at 1/4 resolution to recover sharp boundaries.
"""
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from rover_roadkeeper.config import (
    DINOV2_EMBED_DIM,
    DINOV2_INPUT_HEIGHT,
    DINOV2_INPUT_WIDTH,
    DINOV2_MODEL_NAME,
    DINOV2_NUM_LAYERS,
    FRAME_HEIGHT,
    FRAME_WIDTH,
)
from rover_roadkeeper.perception.base_model import BaseRoadSegmenter


def _load_dinov2(model_name: str) -> nn.Module:
    """Load from the local torch.hub cache when available (works offline on the rover)."""
    local_repo = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
    if local_repo.exists():
        return torch.hub.load(str(local_repo), model_name, source="local", pretrained=True)
    return torch.hub.load("facebookresearch/dinov2", model_name, pretrained=True)


def _conv_bn_act(cin: int, cout: int, k: int = 3, stride: int = 1, act=nn.GELU) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, stride=stride, padding=k // 2, bias=False),
        nn.BatchNorm2d(cout),
        act(),
    )


class DINOv2RoadSegmenter(BaseRoadSegmenter):
    ARCH = "dinov2"

    def __init__(self, hidden_dim: int = 128, num_layers: int = DINOV2_NUM_LAYERS):
        super().__init__()
        self.num_layers = num_layers

        self.backbone = _load_dinov2(DINOV2_MODEL_NAME)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        # Semantic branch (patch grid 26x51)
        self.proj = _conv_bn_act(DINOV2_EMBED_DIM * num_layers, hidden_dim, k=1)
        self.refine = _conv_bn_act(hidden_dim, hidden_dim, k=3)
        self.dropout = nn.Dropout2d(0.1)

        # Edge branch on the full-res image -> 1/4 resolution, 32 channels
        self.edge = nn.Sequential(
            _conv_bn_act(3, 16, k=3, stride=2, act=nn.ReLU),
            _conv_bn_act(16, 32, k=3, stride=2, act=nn.ReLU),
        )

        # Fusion at 1/4 resolution -> single road logit
        self.fuse = nn.Sequential(
            _conv_bn_act(hidden_dim + 32, 64, k=3),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()   # frozen backbone always in eval mode
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        x = self.normalize(x)

        # 1. Frozen DINOv2 features
        x_vit = F.interpolate(x, size=(DINOV2_INPUT_HEIGHT, DINOV2_INPUT_WIDTH), mode="bilinear", align_corners=False)
        with torch.no_grad():
            layers = self.backbone.get_intermediate_layers(x_vit, n=self.num_layers, reshape=True, norm=True)
            feats = torch.cat(layers, dim=1)                       # (B, 384*N, 26, 51)

        # 2. Semantic head
        f = self.proj(feats)
        f = self.dropout(f + self.refine(f))

        # 3. Fuse with edge features at 1/4 resolution
        e = self.edge(x)                                           # (B, 32, H/4, W/4)
        f = F.interpolate(f, size=e.shape[-2:], mode="bilinear", align_corners=False)
        logits = self.fuse(torch.cat([f, e], dim=1))

        return F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)

    # Store only the head; the frozen backbone comes from torch.hub.
    def trainable_state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v for k, v in self.state_dict().items() if not k.startswith("backbone.")}

    def load_trainable_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        missing, unexpected = self.load_state_dict(state, strict=False)
        missing = [k for k in missing if not k.startswith("backbone.")]
        if missing or unexpected:
            raise RuntimeError(f"Checkpoint mismatch. missing={missing} unexpected={unexpected}")


assert (FRAME_HEIGHT, FRAME_WIDTH) == (364, 720), "Head geometry assumes 720x364 frames"
