import numpy as np
import torch
import time
from typing import Dict, Tuple, Optional
from rover_roadkeeper.config import NUM_CLASSES, IGNORE_INDEX, CLASS_NAMES


def calculate_iou(
    confusion_matrix: np.ndarray,
    ignore_index: int = IGNORE_INDEX,
) -> Tuple[Dict[str, float], float]:
    """
    Computes per-class IoU and mean IoU from a confusion matrix.
    Args:
        confusion_matrix: 2D numpy array of shape (NUM_CLASSES, NUM_CLASSES)
                          rows: ground truth, cols: prediction
        ignore_index: Class ID to exclude from mIoU calculation
    Returns:
        per_class_iou: Dict mapping class_name -> IoU (0.0 .. 1.0)
        mean_iou: Mean IoU across non-ignored classes
    """
    per_class_iou = {}
    valid_ious = []

    for c in range(NUM_CLASSES):
        if c == ignore_index:
            continue

        tp = confusion_matrix[c, c]
        fp = confusion_matrix[:, c].sum() - tp
        fn = confusion_matrix[c, :].sum() - tp
        denom = tp + fp + fn

        iou = float(tp / denom) if denom > 0 else 0.0
        c_name = CLASS_NAMES.get(c, f"class_{c}")
        per_class_iou[c_name] = iou
        valid_ious.append(iou)

    mean_iou = float(np.mean(valid_ious)) if valid_ious else 0.0
    return per_class_iou, mean_iou


class Evaluator:
    """
    Accumulator for segmentation evaluation metrics across batches and datasets.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, ignore_index: int = IGNORE_INDEX):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        self.total_time_sec = 0.0
        self.total_frames = 0

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.total_time_sec = 0.0
        self.total_frames = 0

    def update(
        self,
        preds: np.ndarray,
        targets: np.ndarray,
        batch_duration_sec: Optional[float] = None,
    ):
        """
        Args:
            preds: numpy array of predicted class indices (H, W) or (B, H, W)
            targets: numpy array of ground truth class indices (H, W) or (B, H, W)
            batch_duration_sec: execution time in seconds for this batch (for FPS)
        """
        preds = np.asarray(preds).flatten()
        targets = np.asarray(targets).flatten()

        # Filter out pixels with ignore_index in ground truth
        valid_mask = targets != self.ignore_index
        p = preds[valid_mask]
        t = targets[valid_mask]

        # Fast 2D bincount confusion matrix accumulation
        indices = self.num_classes * t + p
        counts = np.bincount(indices, minlength=self.num_classes ** 2)
        self.confusion_matrix += counts.reshape(self.num_classes, self.num_classes)

        if batch_duration_sec is not None:
            self.total_time_sec += batch_duration_sec
            self.total_frames += 1

    def compute(self) -> Dict[str, float]:
        per_class_iou, mean_iou = calculate_iou(self.confusion_matrix, self.ignore_index)
        metrics = {
            "mIoU": mean_iou,
            **{f"IoU_{k}": v for k, v in per_class_iou.items()},
        }
        if self.total_time_sec > 0:
            metrics["FPS"] = self.total_frames / self.total_time_sec
        return metrics
