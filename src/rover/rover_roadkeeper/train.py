#!/usr/bin/env python3
"""
Train / evaluate the binary road segmenter on track-1/seg/ (contiguous split 240 / 61).

    python train.py                       # train DINOv2 head (default), then full evaluation
    python train.py --arch unet           # swap model
    python train.py --eval_only           # evaluate weights/road_segmenter_best.pt

Loss = masked BCEWithLogits(pos_weight < 1 -> penalise false-positive road) + soft Dice.
Model selection uses road IoU at the DEPLOYED threshold (ROAD_PROB_THRESHOLD), and the final
report also evaluates the full post-processing (threshold + morphology + largest component).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rover_roadkeeper import config as C  # noqa: E402
from rover_roadkeeper.dataset import BinaryRoadDataset, get_dataloaders  # noqa: E402
from rover_roadkeeper.perception.offset_calculator import OffsetCalculator  # noqa: E402
from rover_roadkeeper.perception.pipeline import (  # noqa: E402
    RoadPerceptionCore, build_segmenter, load_segmenter, save_checkpoint)


# ----------------------------------------------------------------------------
# Loss & metrics
# ----------------------------------------------------------------------------
def road_loss(logits: torch.Tensor, target: torch.Tensor, pos_weight: float, dice_w: float) -> torch.Tensor:
    logits = logits[:, 0].float()
    valid = (target != C.LABEL_IGNORE).float()
    y = (target == C.LABEL_ROAD).float()

    bce = F.binary_cross_entropy_with_logits(
        logits, y, pos_weight=torch.tensor(pos_weight, device=logits.device), reduction="none")
    bce = (bce * valid).sum() / valid.sum().clamp_min(1.0)

    p = torch.sigmoid(logits) * valid
    dice = 1.0 - (2.0 * (p * y).sum() + 1.0) / (p.sum() + (y * valid).sum() + 1.0)
    return bce + dice_w * dice


class BinaryMetrics:
    def __init__(self):
        self.tp = self.fp = self.fn = self.tn = 0

    def update(self, pred: np.ndarray, target: np.ndarray) -> None:
        valid = target != C.LABEL_IGNORE
        p, t = pred.astype(bool)[valid], (target == C.LABEL_ROAD)[valid]
        self.tp += int(np.sum(p & t)); self.fp += int(np.sum(p & ~t))
        self.fn += int(np.sum(~p & t)); self.tn += int(np.sum(~p & ~t))

    def compute(self) -> dict:
        eps = 1e-9
        return {
            "iou": self.tp / (self.tp + self.fp + self.fn + eps),
            "precision": self.tp / (self.tp + self.fp + eps),
            "recall": self.tp / (self.tp + self.fn + eps),
            "pred_road_frac": (self.tp + self.fp) / (self.tp + self.fp + self.fn + self.tn + eps),
            "gt_road_frac": (self.tp + self.fn) / (self.tp + self.fp + self.fn + self.tn + eps),
        }


@torch.no_grad()
def evaluate_raw(model, loader, device, threshold: float) -> dict:
    model.eval()
    m = BinaryMetrics()
    for x, y in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            prob = torch.sigmoid(model(x.to(device)).float())[:, 0].cpu().numpy()
        for pr, t in zip(prob, y.numpy()):
            m.update(pr > threshold, t)
    return m.compute()


def fmt(d: dict) -> str:
    return (f"IoU {d['iou'] * 100:5.1f}% | Prec {d['precision'] * 100:5.1f}% | Rec {d['recall'] * 100:5.1f}% | "
            f"pred road {d['pred_road_frac'] * 100:4.1f}% (gt {d['gt_road_frac'] * 100:4.1f}%)")


# ----------------------------------------------------------------------------
# Train
# ----------------------------------------------------------------------------
def train(args, device) -> None:
    torch.manual_seed(C.SEED); np.random.seed(C.SEED)
    train_dl, val_dl = get_dataloaders(args.batch_size, args.num_workers)
    print(f"[*] Train {len(train_dl.dataset)} frames (001000-001239) | Val {len(val_dl.dataset)} frames (001240-001300)")

    model = build_segmenter(args.arch).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"[*] Arch '{args.arch}': {sum(p.numel() for p in params) / 1e6:.2f}M trainable params")

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=C.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(train_dl),
                                                pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_iou, t_start = -1.0, time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y in tqdm(train_dl, desc=f"Epoch {epoch:02d}/{args.epochs}", leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(x)
            loss = road_loss(logits, y, args.pos_weight, C.DICE_WEIGHT)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            losses.append(loss.item())

        val = evaluate_raw(model, val_dl, device, C.ROAD_PROB_THRESHOLD)
        tag = ""
        if val["iou"] > best_iou:
            best_iou = val["iou"]
            save_checkpoint(model, args.checkpoint, epoch=epoch, val=val, pos_weight=args.pos_weight)
            tag = "  <- best"
        print(f"Epoch {epoch:02d} | loss {np.mean(losses):.4f} | val@{C.ROAD_PROB_THRESHOLD:.2f}: {fmt(val)}{tag}")

    print(f"[DONE] {time.time() - t_start:.0f}s, best val IoU {best_iou * 100:.2f}% -> {args.checkpoint}")


# ----------------------------------------------------------------------------
# Full evaluation of the deployed pipeline
# ----------------------------------------------------------------------------
@torch.no_grad()
def full_report(args, device) -> None:
    model = load_segmenter(args.checkpoint, device)
    core = RoadPerceptionCore(model=model, device=str(device))
    calc = OffsetCalculator()
    ds = BinaryRoadDataset("val", augment=False)

    m_argmax, m_thr, m_clean = BinaryMetrics(), BinaryMetrics(), BinaryMetrics()
    for img, lbl in zip(ds.images, ds.labels):
        prob = model.predict_mask(img, device)
        m_argmax.update(prob > 0.5, lbl)
        m_thr.update(prob > C.ROAD_PROB_THRESHOLD, lbl)
        m_clean.update(calc.clean_mask(prob), lbl)

    # End-to-end speed: process_frame (model + post-proc + offset), no debug rendering.
    frame = ds.images[0]
    for _ in range(10):
        core.process_frame(frame, render_debug=False)
    n, t0 = 100, time.perf_counter()
    for i in range(n):
        core.process_frame(ds.images[i % len(ds)], render_debug=False)
    fps = n / (time.perf_counter() - t0)

    print("\n" + "=" * 96)
    print(f" VALIDATION (contiguous block 001240-001300, {len(ds)} frames)  |  arch={model.ARCH}  device={device}")
    print("=" * 96)
    print(f" p > 0.50 (argmax-equivalent)        : {fmt(m_argmax.compute())}")
    print(f" p > {C.ROAD_PROB_THRESHOLD:.2f}                           : {fmt(m_thr.compute())}")
    print(f" p > {C.ROAD_PROB_THRESHOLD:.2f} + morph + largest comp.   : {fmt(m_clean.compute())}   <- deployed")
    print(f" End-to-end process_frame throughput : {fps:.1f} FPS")
    print("=" * 96)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", default="dinov2", choices=["dinov2", "unet"])
    ap.add_argument("--epochs", type=int, default=C.NUM_EPOCHS)
    ap.add_argument("--batch_size", type=int, default=C.BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=C.LEARNING_RATE)
    ap.add_argument("--pos_weight", type=float, default=C.BCE_POS_WEIGHT)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--checkpoint", default=str(C.DEFAULT_CHECKPOINT))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--eval_only", action="store_true")
    a = ap.parse_args()

    dev = torch.device(a.device)
    if not a.eval_only:
        train(a, dev)
    full_report(a, dev)
