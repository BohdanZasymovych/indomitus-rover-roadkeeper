#!/usr/bin/env python3
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Ensure package root is in python path
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from rover_roadkeeper.config import (
    CLASS_WEIGHTS,
    IGNORE_INDEX,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_EPOCHS,
    BATCH_SIZE,
    SEED,
    DEFAULT_CHECKPOINT,
    WEIGHTS_DIR,
    ORIG_WIDTH,
    ORIG_HEIGHT,
)
from rover_roadkeeper.dataset import get_dataloaders
from rover_roadkeeper.models.dinov2_segmenter import DINOv2Segmenter
from rover_roadkeeper.utils.metrics import Evaluator


class CombinedLoss(nn.Module):
    """
    Combines weighted CrossEntropyLoss (with ignore_index=0)
    and Multi-class Dice Loss for critical classes (Road & Obstacle).
    """

    def __init__(self, class_weights: list, ignore_index: int = IGNORE_INDEX):
        super().__init__()
        weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
        self.ce = nn.CrossEntropyLoss(weight=weights_tensor, ignore_index=ignore_index)
        self.ignore_index = ignore_index

    def dice_loss(self, probs: torch.Tensor, targets: torch.Tensor, class_id: int) -> torch.Tensor:
        p = probs[:, class_id, :, :]
        t = (targets == class_id).float()
        valid = (targets != self.ignore_index).float()

        p = p * valid
        t = t * valid

        intersection = 2.0 * torch.sum(p * t) + 1e-5
        cardinality = torch.sum(p) + torch.sum(t) + 1e-5
        return 1.0 - (intersection / cardinality)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)
        probs = F.softmax(logits, dim=1)

        # Dice loss focused on road (class 1) and obstacle (class 4)
        road_dice = self.dice_loss(probs, targets, class_id=1)
        obs_dice = self.dice_loss(probs, targets, class_id=4)

        return ce_loss + 0.5 * road_dice + 0.8 * obs_dice


def evaluate(model: nn.Module, val_loader, device: torch.device) -> dict:
    model.eval()
    evaluator = Evaluator()

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)

            start_t = time.time()
            logits = model(images)
            duration = time.time() - start_t

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            evaluator.update(preds, targets.cpu().numpy(), batch_duration_sec=duration)

    return evaluator.compute()


def train(args):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on device: {device}")

    # 1. Dataloaders
    print("[*] Initializing dataset and dataloaders...")
    train_loader, val_loader = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"[*] Dataset split: {len(train_loader.dataset)} train samples, {len(val_loader.dataset)} val samples.")

    # 2. Model & Optimizer
    print("[*] Building DINOv2Segmenter (frozen ViT-S/14 backbone)...")
    model = DINOv2Segmenter(pretrained_backbone=True).to(device)

    # Only optimize head parameters
    trainable_params = [p for p in model.head.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    criterion = CombinedLoss(class_weights=CLASS_WEIGHTS, ignore_index=IGNORE_INDEX).to(device)

    best_miou = 0.0
    output_checkpoint = Path(args.output_checkpoint)
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    print(f"[*] Starting training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.epochs:02d}")

        for images, targets in pbar:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        epoch_loss = running_loss / len(train_loader)

        # Validation
        metrics = evaluate(model, val_loader, device)
        val_miou = metrics["mIoU"]
        val_road_iou = metrics.get("IoU_road", 0.0)
        val_obs_iou = metrics.get("IoU_obstacle", 0.0)
        val_fps = metrics.get("FPS", 0.0)

        print(
            f"Epoch {epoch:02d} | Loss: {epoch_loss:.4f} | "
            f"Val mIoU: {val_miou * 100:.2f}% | Road IoU: {val_road_iou * 100:.2f}% | "
            f"Obs IoU: {val_obs_iou * 100:.2f}% | FPS: {val_fps:.1f}"
        )

        # Save best model
        if val_miou > best_miou:
            best_miou = val_miou
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "head_state_dict": model.head.state_dict(),
                "best_miou": best_miou,
                "metrics": metrics,
            }, str(output_checkpoint))
            print(f"  [+] Saved new best model to {output_checkpoint} (mIoU: {best_miou * 100:.2f}%)")

    print(f"\n[DONE] Training complete. Best Validation mIoU: {best_miou * 100:.2f}%")
    print(f"[DONE] Best checkpoint saved at: {output_checkpoint}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DINOv2 Segmentation Head")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of dataloader workers")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--output_checkpoint", type=str, default=str(DEFAULT_CHECKPOINT), help="Path to save best checkpoint")
    args = parser.parse_args()

    train(args)
