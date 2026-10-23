"""Frozen DINOv2 ViT-S/14 backbone + a small binary road head.

    x (B, 3, 364, 714) in [0, 1]  ->  road logits (B, 1, 91, 179)

The backbone is frozen (only ~0.44 M head parameters are trained), which is
what keeps 240 labelled frames of one forest ride from overfitting. The
26x51 patch grid is coarse, so a two-layer conv branch on the CLAHE image is
fused at 1/4 resolution to recover sharp road edges. The logits stay at that
low resolution; the caller upsamples them to whatever the target is (labels
in training, the original camera frame at inference).
"""
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from rover_road_segmentation import config as C

CHECKPOINT_FORMAT = 'rover_road_segmentation/dinov2_binary_head/v1'


def load_dinov2_backbone(model_name=C.DINOV2_MODEL_NAME):
    """DINOv2 from the local torch.hub cache when present, so the rover works offline."""
    local_repo = Path(torch.hub.get_dir()) / 'facebookresearch_dinov2_main'
    if local_repo.exists():
        return torch.hub.load(str(local_repo), model_name, source='local', pretrained=True)
    return torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True)


def _conv_bn_act(cin, cout, k=3, stride=1, act=nn.GELU):
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, stride=stride, padding=k // 2, bias=False),
        nn.BatchNorm2d(cout),
        act(),
    )


class DINOv2RoadNet(nn.Module):

    def __init__(self, hidden_dim=C.HEAD_HIDDEN_DIM, num_layers=C.DINOV2_NUM_LAYERS):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.register_buffer('img_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
                             persistent=False)
        self.register_buffer('img_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
                             persistent=False)

        self.backbone = load_dinov2_backbone()
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        # Semantic branch on the 26x51 patch grid.
        self.proj = _conv_bn_act(C.DINOV2_EMBED_DIM * num_layers, hidden_dim, k=1)
        self.refine = _conv_bn_act(hidden_dim, hidden_dim, k=3)
        self.dropout = nn.Dropout2d(0.1)
        # Edge branch on the full inference image -> 1/4 resolution.
        self.edge = nn.Sequential(
            _conv_bn_act(3, 16, k=3, stride=2, act=nn.ReLU),
            _conv_bn_act(16, 32, k=3, stride=2, act=nn.ReLU),
        )
        self.fuse = nn.Sequential(
            _conv_bn_act(hidden_dim + 32, 64, k=3),
            nn.Conv2d(64, 1, kernel_size=1),
        )

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = getattr(C, 'TORCH_CUDNN_BENCHMARK', True)
            if hasattr(torch, 'set_float32_matmul_precision'):
                torch.set_float32_matmul_precision(getattr(C, 'TORCH_MATMUL_PRECISION', 'high'))

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()  # frozen: never update its norm statistics
        return self

    def forward(self, x):
        h, w = x.shape[-2:]
        if h % C.DINOV2_PATCH_SIZE != 0 or w % C.DINOV2_PATCH_SIZE != 0:
            raise ValueError(f'input {h}x{w} must be divisible by patch size {C.DINOV2_PATCH_SIZE}')
        x = (x - self.img_mean) / self.img_std
        with torch.no_grad():
            layers = self.backbone.get_intermediate_layers(x, n=self.num_layers, reshape=True, norm=True)
            feats = torch.cat(layers, dim=1)
        f = self.proj(feats)
        f = self.dropout(f + self.refine(f))
        e = self.edge(x)
        f = F.interpolate(f, size=e.shape[-2:], mode='bilinear', align_corners=False)
        return self.fuse(torch.cat([f, e], dim=1))

    # Only the head is stored; the frozen backbone comes from torch.hub.
    def head_state_dict(self):
        return {k: v for k, v in self.state_dict().items() if not k.startswith('backbone.')}

    def load_head_state_dict(self, state):
        missing, unexpected = self.load_state_dict(state, strict=False)
        missing = [k for k in missing if not k.startswith('backbone.')]
        if missing or unexpected:
            raise RuntimeError(f'checkpoint mismatch: missing={missing} unexpected={unexpected}')


def save_checkpoint(model, path, **meta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'format': CHECKPOINT_FORMAT,
        'backbone': C.DINOV2_MODEL_NAME,
        'hidden_dim': model.hidden_dim,
        'num_layers': model.num_layers,
        'infer_size': (C.INFER_HEIGHT, C.INFER_WIDTH),
        'head': {k: v.detach().cpu() for k, v in model.head_state_dict().items()},
        'meta': meta,
    }, path)


def load_model(path, device='cpu'):
    """DINOv2RoadNet with the head weights of a checkpoint, in eval mode."""
    ckpt = torch.load(str(path), map_location='cpu', weights_only=False)
    if ckpt.get('format') != CHECKPOINT_FORMAT:
        raise RuntimeError(f'{path}: not a {CHECKPOINT_FORMAT} checkpoint '
                           f'(got {ckpt.get("format")!r}); retrain with train_dinov2.py')
    model = DINOv2RoadNet(hidden_dim=ckpt['hidden_dim'], num_layers=ckpt['num_layers'])
    model.load_head_state_dict(ckpt['head'])
    meta = ckpt.get('meta', {})
    meta['infer_size'] = ckpt.get('infer_size', (C.INFER_HEIGHT, C.INFER_WIDTH))
    return model.to(device).eval(), meta
