import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Optional, Union
from pathlib import Path

from rover_roadkeeper.config import (
    ORIG_WIDTH,
    ORIG_HEIGHT,
    DINOV2_WIDTH,
    DINOV2_HEIGHT,
    DINOV2_PATCH_SIZE,
    DINOV2_EMBED_DIM,
    NUM_CLASSES,
    IGNORE_INDEX,
    HUD_MASK_BOX,
    ROI_Y_MIN_RATIO,
    ROI_Y_MAX_RATIO,
)


class LightweightConvHead(nn.Module):
    """
    Lightweight convolutional segmentation head.
    Transforms dense DINOv2 patch tokens (384 channels) into class logits (5 channels).
    """

    def __init__(self, in_channels: int = 384, hidden_dim: int = 128, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Dropout2d(p=0.1),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )
        self.classifier = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.block1(x)
        feat = feat + self.block2(feat)  # Residual connection
        logits = self.classifier(feat)
        return logits


class DINOv2Segmenter(nn.Module):
    """
    Few-Shot Road Segmenter powered by frozen DINOv2 ViT-S/14 backbone
    and lightweight multi-scale convolutional segmentation head.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        embed_dim: int = DINOV2_EMBED_DIM,
        model_name: str = "dinov2_vits14",
        pretrained_backbone: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.patch_size = DINOV2_PATCH_SIZE
        self.patch_h = DINOV2_HEIGHT // self.patch_size  # 26
        self.patch_w = DINOV2_WIDTH // self.patch_size   # 51

        # Load DINOv2 ViT-S/14
        self.backbone = None
        if pretrained_backbone:
            try:
                # Primary method: PyTorch Hub official repository
                self.backbone = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=True)
            except Exception as e:
                # Fallback method: timm
                try:
                    import timm
                    timm_name = "vit_small_patch14_dinov2.lvd142m"
                    self.backbone = timm.create_model(timm_name, pretrained=True, num_classes=0)
                except Exception as e_timm:
                    raise RuntimeError(f"Failed to load DINOv2 backbone via torch.hub and timm: {e} | {e_timm}")

        # Freeze all backbone parameters
        if self.backbone is not None:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        # Trainable segmentation head
        self.head = LightweightConvHead(
            in_channels=self.embed_dim,
            hidden_dim=128,
            num_classes=self.num_classes,
        )

        # Normalization constants (ImageNet mean & std)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def train(self, mode: bool = True):
        """Keep backbone in eval mode (BatchNorm/Dropout frozen) during training."""
        super().train(mode)
        if self.backbone is not None:
            self.backbone.eval()
        return self

    def extract_patch_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through frozen DINOv2 and extract (B, embed_dim, H_p, W_p) dense feature map.
        """
        B = x.shape[0]
        with torch.no_grad():
            if hasattr(self.backbone, "forward_features"):
                feats = self.backbone.forward_features(x)
                if isinstance(feats, dict) and "x_norm_patchtokens" in feats:
                    patch_tokens = feats["x_norm_patchtokens"]
                elif hasattr(self.backbone, "get_intermediate_layers"):
                    patch_tokens = self.backbone.get_intermediate_layers(x, n=1)[0]
                else:
                    # In timm ViT models, forward_features returns (B, N, C)
                    if feats.ndim == 3:
                        # Skip cls token if present (e.g. 1327 -> 1326)
                        if feats.shape[1] == (self.patch_h * self.patch_w + 1):
                            patch_tokens = feats[:, 1:, :]
                        else:
                            patch_tokens = feats[:, :self.patch_h * self.patch_w, :]
                    else:
                        raise ValueError(f"Unexpected feature shape: {feats.shape}")
            else:
                patch_tokens = self.backbone.get_intermediate_layers(x, n=1)[0]

            # Reshape (B, N_patches, 384) -> (B, 384, H_p, W_p)
            feat_map = patch_tokens.permute(0, 2, 1).contiguous()
            feat_map = feat_map.view(B, self.embed_dim, self.patch_h, self.patch_w)

        return feat_map

    def forward(self, x: torch.Tensor, out_size: Tuple[int, int] = (ORIG_HEIGHT, ORIG_WIDTH)) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, 3, H, W).
            out_size: Desired output logits resolution (H_out, W_out), default (364, 720).
        Returns:
            logits: Tensor of shape (B, num_classes, H_out, W_out).
        """
        # Resize to DINOv2 patch alignment (364, 714)
        if x.shape[-2:] != (DINOV2_HEIGHT, DINOV2_WIDTH):
            x_dino = F.interpolate(x, size=(DINOV2_HEIGHT, DINOV2_WIDTH), mode="bilinear", align_corners=False)
        else:
            x_dino = x

        # Extract frozen patch features
        feat_map = self.extract_patch_features(x_dino)

        # Conv decoder head
        head_logits = self.head(feat_map)

        # Bilinear upsample back to target original resolution (364, 720)
        logits = F.interpolate(head_logits, size=out_size, mode="bilinear", align_corners=False)

        return logits

    @torch.no_grad()
    def predict(
        self,
        frame_bgr: np.ndarray,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Runs perception inference on a single OpenCV BGR image frame.
        Handles arbitrary input resolutions automatically.

        Returns:
            class_mask: uint8 array of shape (ORIG_HEIGHT, ORIG_WIDTH) with class indices (0..4).
            road_conf_map: float32 array of shape (ORIG_HEIGHT, ORIG_WIDTH) with road probabilities [0..1].
            mean_road_confidence: float [0..1] representing road confidence in the drivable ROI.
        """
        if device is None:
            device = next(self.parameters()).device

        orig_h, orig_w = frame_bgr.shape[:2]

        # Resize to standard ORIG resolution (364, 720) if different
        if (orig_h, orig_w) != (ORIG_HEIGHT, ORIG_WIDTH):
            frame_resized = cv2.resize(frame_bgr, (ORIG_WIDTH, ORIG_HEIGHT), interpolation=cv2.INTER_LINEAR)
        else:
            frame_resized = frame_bgr

        # Convert BGR to RGB and normalize
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
        tensor = (tensor - self.mean.to(device)) / self.std.to(device)

        logits = self.forward(tensor, out_size=(ORIG_HEIGHT, ORIG_WIDTH))
        probs = F.softmax(logits, dim=1).squeeze(0)  # (NUM_CLASSES, H, W)

        # Predicted class indices (0..4)
        pred_classes = torch.argmax(probs, dim=0).cpu().numpy().astype(np.uint8)

        # Road confidence map (prob of class 1)
        road_conf_map = probs[1].cpu().numpy().astype(np.float32)

        # Mask HUD box in bottom-left to IGNORE_INDEX
        y_min, y_max = HUD_MASK_BOX["y_min"], HUD_MASK_BOX["y_max"]
        x_min, x_max = HUD_MASK_BOX["x_min"], HUD_MASK_BOX["x_max"]
        pred_classes[y_min:y_max, x_min:x_max] = IGNORE_INDEX
        road_conf_map[y_min:y_max, x_min:x_max] = 0.0

        # Compute mean road confidence inside drivable ROI
        roi_y_start = int(ORIG_HEIGHT * ROI_Y_MIN_RATIO)
        roi_y_end = int(ORIG_HEIGHT * ROI_Y_MAX_RATIO)
        roi_road_probs = road_conf_map[roi_y_start:roi_y_end, :]
        roi_road_pixels = pred_classes[roi_y_start:roi_y_end, :] == 1

        if np.any(roi_road_pixels):
            mean_road_conf = float(np.mean(roi_road_probs[roi_road_pixels]))
        else:
            mean_road_conf = float(np.mean(roi_road_probs))

        return pred_classes, road_conf_map, mean_road_conf
