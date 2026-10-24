#!/usr/bin/env python3
"""Train / evaluate the DINOv2 binary road head on track-1/seg/ (contiguous split 240 / 61).

    python train_dinov2.py                         # train, then evaluate the deployed pipeline
    python train_dinov2.py --eval_only             # evaluate weights/road_segmenter_dinov2.pt
    python train_dinov2.py --data_dir ~/def_tech_hackathon/track1 --epochs 40

Loss = masked BCE-with-logits (pos_weight < 1: false-positive road costs more)
       + soft Dice, on logits upsampled to the label resolution.
The best epoch is chosen by road IoU at the deployed threshold; the final report
also scores the full RoadSegmenter (threshold + morphology + largest component)
and its speed at several input resolutions.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rover_road_segmentation import config as C  # noqa: E402
from rover_road_segmentation.dataset import SegDataset, make_loaders  # noqa: E402
from rover_road_segmentation.model import DINOv2RoadNet, load_model, save_checkpoint  # noqa: E402
from rover_road_segmentation.segmenter import RoadSegmenter  # noqa: E402


def road_loss(logits, target, pos_weight, dice_weight):
    logits = F.interpolate(logits.float(), size=target.shape[-2:], mode='bilinear', align_corners=False)[:, 0]
    valid = (target != C.LABEL_IGNORE).float()
    y = (target == C.LABEL_ROAD).float()
    bce = F.binary_cross_entropy_with_logits(
        logits, y, pos_weight=torch.tensor(pos_weight, device=logits.device), reduction='none')
    bce = (bce * valid).sum() / valid.sum().clamp_min(1.0)
    p = torch.sigmoid(logits) * valid
    dice = 1.0 - (2.0 * (p * y).sum() + 1.0) / (p.sum() + (y * valid).sum() + 1.0)
    return bce + dice_weight * dice


class BinaryMetrics:

    def __init__(self):
        self.tp = self.fp = self.fn = self.tn = 0

    def update(self, pred, target):
        valid = target != C.LABEL_IGNORE
        p, t = pred.astype(bool)[valid], (target == C.LABEL_ROAD)[valid]
        self.tp += int(np.sum(p & t))
        self.fp += int(np.sum(p & ~t))
        self.fn += int(np.sum(~p & t))
        self.tn += int(np.sum(~p & ~t))

    def compute(self):
        eps = 1e-9
        total = self.tp + self.fp + self.fn + self.tn + eps
        return {
            'iou': self.tp / (self.tp + self.fp + self.fn + eps),
            'precision': self.tp / (self.tp + self.fp + eps),
            'recall': self.tp / (self.tp + self.fn + eps),
            'pred': (self.tp + self.fp) / total,
            'gt': (self.tp + self.fn) / total,
        }


def fmt(m):
    return (f"IoU {m['iou'] * 100:5.1f}% | Prec {m['precision'] * 100:5.1f}% | Rec {m['recall'] * 100:5.1f}% | "
            f"pred road {m['pred'] * 100:4.1f}% (gt {m['gt'] * 100:4.1f}%)")


@torch.no_grad()
def evaluate_raw(model, loader, device, threshold):
    model.eval()
    m = BinaryMetrics()
    for x, y in loader:
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
            logits = model(x.to(device))
        logits = F.interpolate(logits.float(), size=y.shape[-2:], mode='bilinear', align_corners=False)
        for pr, t in zip(torch.sigmoid(logits)[:, 0].cpu().numpy(), y.numpy()):
            m.update(pr > threshold, t)
    return m.compute()


def train(args, device):
    torch.manual_seed(C.SEED)
    np.random.seed(C.SEED)
    train_dl, val_dl = make_loaders(args.data_dir, args.batch_size, args.num_workers)
    tr, va = train_dl.dataset.names, val_dl.dataset.names
    print(f'[*] train {len(tr)} frames ({tr[0]}-{tr[-1]}) | val {len(va)} frames ({va[0]}-{va[-1]})')

    model = DINOv2RoadNet().to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f'[*] {C.DINOV2_MODEL_NAME} frozen, head {sum(p.numel() for p in params) / 1e6:.2f}M trainable params')

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=C.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(train_dl),
                                                pct_start=0.1)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')

    best, t0 = -1.0, time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y in tqdm(train_dl, desc=f'epoch {epoch:02d}/{args.epochs}', leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                logits = model(x)
            loss = road_loss(logits, y, args.pos_weight, C.DICE_WEIGHT)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())

        val = evaluate_raw(model, val_dl, device, C.ROAD_PROB_THRESHOLD)
        tag = ''
        if val['iou'] > best:
            best = val['iou']
            save_checkpoint(model, args.weights, epoch=epoch, val_iou=val['iou'],
                            val_precision=val['precision'], pos_weight=args.pos_weight)
            tag = '  <- best'
        print(f'epoch {epoch:02d} | loss {np.mean(losses):.4f} | val@{C.ROAD_PROB_THRESHOLD:.2f}: {fmt(val)}{tag}')
    print(f'[done] {time.time() - t0:.0f}s, best val IoU {best * 100:.2f}% -> {args.weights}')


@torch.no_grad()
def report(args, device):
    model, meta = load_model(args.weights, device)
    seg = RoadSegmenter(model=model, device=str(device))
    ds = SegDataset(args.data_dir, 'val')

    m_half, m_thr, m_clean = BinaryMetrics(), BinaryMetrics(), BinaryMetrics()
    for img, lbl in zip(ds.images, ds.labels):
        mask, prob = seg.segment_with_proba(img, 'bgr')
        assert mask.shape == lbl.shape and mask.dtype == np.uint8 and set(np.unique(mask)) <= {0, 1}
        m_half.update(prob > 0.5, lbl)
        m_thr.update(prob > seg.post.threshold, lbl)
        m_clean.update(mask, lbl)

    rule = '=' * 100
    print(f'\n{rule}\n VALIDATION ({ds.names[0]}-{ds.names[-1]}, {len(ds)} frames) | device={device} | '
          f'checkpoint epoch {meta.get("epoch")}\n{rule}')
    print(f' p > 0.50                          : {fmt(m_half.compute())}')
    print(f' p > {seg.post.threshold:.2f}                          : {fmt(m_thr.compute())}')
    print(f' p > {seg.post.threshold:.2f} + morph + largest comp.  : {fmt(m_clean.compute())}  <- deployed')

    # End-to-end speed of segment() at the resolutions the rover/videos produce.
    base = ds.images[0]
    for w, h in [(720, 364), (700, 364), (1280, 720), (2688, 1520)]:
        frame = cv2.resize(base, (w, h))
        for _ in range(5):
            seg.segment(frame)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        n, t = 30, time.perf_counter()
        for _ in range(n):
            out = seg.segment(frame)
        fps = n / (time.perf_counter() - t)
        print(f' segment() {w:4d}x{h:<4d} -> mask {out.shape[1]}x{out.shape[0]} : {fps:5.1f} FPS')
    print(rule)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data_dir', type=Path, default=C.find_data_dir(),
                    help='track-1 root (contains seg/frames, seg/markdown); env ROAD_SEG_DATA_DIR')
    ap.add_argument('--weights', type=Path, default=C.DEFAULT_WEIGHTS)
    ap.add_argument('--epochs', type=int, default=C.NUM_EPOCHS)
    ap.add_argument('--batch_size', type=int, default=C.BATCH_SIZE)
    ap.add_argument('--lr', type=float, default=C.LEARNING_RATE)
    ap.add_argument('--pos_weight', type=float, default=C.BCE_POS_WEIGHT)
    ap.add_argument('--num_workers', type=int, default=2)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--eval_only', action='store_true')
    a = ap.parse_args()
    if a.data_dir is None or not (Path(a.data_dir) / 'seg' / 'frames').is_dir():
        ap.error(f'no seg/frames under --data_dir {a.data_dir}')

    dev = torch.device(a.device)
    if not a.eval_only:
        train(a, dev)
    report(a, dev)
