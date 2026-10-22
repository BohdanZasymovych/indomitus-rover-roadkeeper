#!/usr/bin/env python3
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import torch
from tabulate import tabulate

# Ensure package root is in python path
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from rover_roadkeeper.config import (
    DEFAULT_CHECKPOINT,
    ORIG_WIDTH,
    ORIG_HEIGHT,
    CLASS_NAMES,
)
from rover_roadkeeper.dataset import get_dataloaders
from rover_roadkeeper.models.dinov2_segmenter import DINOv2Segmenter
from rover_roadkeeper.utils.metrics import Evaluator


def run_evaluation(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[*] Running standalone evaluation on device: {device}")

    # 1. Dataloader for validation
    _, val_loader = get_dataloaders(batch_size=args.batch_size, num_workers=2)
    print(f"[*] Validation set size: {len(val_loader.dataset)} frames (contiguous block 001240..001300)")

    # 2. Build model
    model = DINOv2Segmenter(pretrained_backbone=True).to(device)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"[!] Warning: Checkpoint {ckpt_path} not found! Evaluating initialized weights.")
    else:
        print(f"[*] Loading checkpoint: {ckpt_path}")
        state = torch.load(str(ckpt_path), map_location=device)
        model.head.load_state_dict(state.get("head_state_dict", state))

    model.eval()
    evaluator = Evaluator()

    # Warmup
    dummy = torch.randn(1, 3, ORIG_HEIGHT, ORIG_WIDTH, device=device)
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy)

    # Evaluation loop
    print("[*] Evaluating...")
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)

            start_t = time.time()
            logits = model(images)
            duration = time.time() - start_t

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            evaluator.update(preds, targets.cpu().numpy(), batch_duration_sec=duration)

    metrics = evaluator.compute()

    # Format results table
    table_data = [
        ["Class", "IoU (%)"],
        ["------------------", "---------"],
    ]
    for c_id, name in CLASS_NAMES.items():
        if c_id == 0:
            continue
        iou_val = metrics.get(f"IoU_{name}", 0.0) * 100
        table_data.append([name.capitalize(), f"{iou_val:.2f}%"])

    table_data.append(["------------------", "---------"])
    table_data.append(["Mean IoU (mIoU)", f"{metrics['mIoU'] * 100:.2f}%"])
    table_data.append(["Throughput (FPS)", f"{metrics.get('FPS', 0.0):.1f}"])

    print("\n" + "=" * 35)
    print("      VALIDATION EVALUATION")
    print("=" * 35)
    for row in table_data:
        print(f"{row[0]:<20} {row[1]:>10}")
    print("=" * 35 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DINOv2 Road Segmenter")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT), help="Path to checkpoint")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    args = parser.parse_args()

    run_evaluation(args)
