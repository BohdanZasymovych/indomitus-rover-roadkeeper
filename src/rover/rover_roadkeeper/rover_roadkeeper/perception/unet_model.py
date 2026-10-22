"""
Optional lightweight U-Net road segmenter (segmentation_models_pytorch), drop-in
replacement for DINOv2RoadSegmenter through BaseRoadSegmenter.

    python train.py --arch unet
"""
import torch
import torch.nn.functional as F

from rover_roadkeeper.perception.base_model import BaseRoadSegmenter


class UNetRoadSegmenter(BaseRoadSegmenter):
    ARCH = "unet"

    def __init__(self, encoder_name: str = "resnet18", encoder_weights: str = "imagenet"):
        super().__init__()
        import segmentation_models_pytorch as smp   # imported lazily: optional dependency
        self.net = smp.Unet(encoder_name=encoder_name, encoder_weights=encoder_weights,
                            in_channels=3, classes=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        x = self.normalize(x)
        # U-Net needs H, W divisible by 32 -> reflect-pad, then crop back.
        ph, pw = (-H) % 32, (-W) % 32
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        return self.net(x)[..., :H, :W]
