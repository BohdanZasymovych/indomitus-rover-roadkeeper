"""
Binary Road vs Non-Road dataset for track-1/seg/.

Labels:  1 = road (magenta), 0 = everything else, 255 = ignore (HUD box [+ unlabeled black
if UNLABELED_AS_BACKGROUND is False]).
Split:   CONTIGUOUS temporal blocks only (frames are consecutive & highly correlated):
         first TRAIN_SPLIT_COUNT -> train, rest -> val. Never random.
Images go through the SAME preprocess_ir() (gray + CLAHE) used at inference time.
Augmentations are plain OpenCV/numpy (IR degradations: exposure, gamma, blur, noise,
JPEG artifacts, small affine jitter for camera-mount differences).
"""
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from rover_roadkeeper import config as C
from rover_roadkeeper.perception.base_model import preprocess_ir, to_tensor


def rgb_mask_to_binary(mask_rgb: np.ndarray) -> np.ndarray:
    """RGB semantic mask -> uint8 {0: background, 1: road, 255: ignore}."""
    r, g, b = mask_rgb[..., 0], mask_rgb[..., 1], mask_rgb[..., 2]
    label = np.full(mask_rgb.shape[:2], C.LABEL_BACKGROUND, dtype=np.uint8)
    label[(r > 127) & (g < 128) & (b > 127)] = C.LABEL_ROAD
    if not C.UNLABELED_AS_BACKGROUND:
        label[(r < 128) & (g < 128) & (b < 128)] = C.LABEL_IGNORE
    x0, y0, x1, y1 = C.HUD_BOX
    label[y0:y1, x0:x1] = C.LABEL_IGNORE
    return label


# ----------------------------------------------------------------------------
# Augmentations (operate on raw BGR uint8 BEFORE CLAHE, like real sensor degradations)
# ----------------------------------------------------------------------------
def _augment(img: np.ndarray, lbl: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    h, w = lbl.shape

    # Geometric: small scale / rotation / shift (different mounts, pitch on bumps)
    if rng.random() < 0.7:
        angle = rng.uniform(-5, 5)
        scale = rng.uniform(0.9, 1.15)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        M[:, 2] += rng.uniform(-0.05, 0.05, size=2) * (w, h)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        lbl = cv2.warpAffine(lbl, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=C.LABEL_IGNORE)

    if rng.random() < 0.5:
        img, lbl = img[:, ::-1], lbl[:, ::-1]

    img = img.astype(np.float32)
    # Exposure / contrast / gamma (IR auto-gain swings)
    img = img * rng.uniform(0.7, 1.3) + rng.uniform(-30, 30)
    img = np.clip(img, 0, 255)
    if rng.random() < 0.5:
        img = 255.0 * (img / 255.0) ** rng.uniform(0.7, 1.5)

    # Blur: gaussian or horizontal/vertical motion blur
    if rng.random() < 0.3:
        if rng.random() < 0.5:
            img = cv2.GaussianBlur(img, (5, 5), 0)
        else:
            k = int(rng.integers(5, 11))
            kernel = np.zeros((k, k), np.float32)
            if rng.random() < 0.5:
                kernel[k // 2, :] = 1.0 / k
            else:
                kernel[:, k // 2] = 1.0 / k
            img = cv2.filter2D(img, -1, kernel)

    # Sensor noise
    if rng.random() < 0.3:
        img = img + rng.normal(0, rng.uniform(3, 12), img.shape)

    img = np.clip(img, 0, 255).astype(np.uint8)

    # Video compression artifacts
    if rng.random() < 0.3:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(25, 75))])
        if ok:
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    return np.ascontiguousarray(img), np.ascontiguousarray(lbl)


# ----------------------------------------------------------------------------
class BinaryRoadDataset(Dataset):
    def __init__(self, split: str = "train", augment: Optional[bool] = None,
                 frames_dir: Path = C.SEG_FRAMES_DIR, masks_dir: Path = C.SEG_MASKS_DIR, seed: int = C.SEED):
        frames: List[Path] = sorted(Path(frames_dir).glob("*.jpg"))
        if not frames:
            raise FileNotFoundError(f"No frames in {frames_dir}")
        if split == "train":
            frames = frames[:C.TRAIN_SPLIT_COUNT]
        elif split == "val":
            frames = frames[C.TRAIN_SPLIT_COUNT:]
        elif split != "all":
            raise ValueError("split must be 'train', 'val' or 'all'")

        self.names = [f.stem for f in frames]
        self.augment = (split == "train") if augment is None else augment
        self.rng = np.random.default_rng(seed)

        # 301 small frames -> just cache everything in RAM.
        self.images, self.labels = [], []
        for f in frames:
            img = cv2.imread(str(f), cv2.IMREAD_COLOR)
            m = cv2.imread(str(Path(masks_dir) / f"{f.stem}.png"), cv2.IMREAD_COLOR)
            if img is None or m is None:
                raise FileNotFoundError(f"Could not read pair for {f.stem}")
            if img.shape[:2] != (C.FRAME_HEIGHT, C.FRAME_WIDTH):
                img = cv2.resize(img, (C.FRAME_WIDTH, C.FRAME_HEIGHT), interpolation=cv2.INTER_AREA)
                m = cv2.resize(m, (C.FRAME_WIDTH, C.FRAME_HEIGHT), interpolation=cv2.INTER_NEAREST)
            self.images.append(img)
            self.labels.append(rgb_mask_to_binary(cv2.cvtColor(m, cv2.COLOR_BGR2RGB)))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img, lbl = self.images[idx], self.labels[idx]
        if self.augment:
            img, lbl = _augment(img, lbl, self.rng)
        x = to_tensor(preprocess_ir(img))
        return x, torch.from_numpy(lbl.astype(np.int64))


def _worker_init(worker_id: int) -> None:
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(torch.initial_seed() % 2**32)


def get_dataloaders(batch_size: int = C.BATCH_SIZE, num_workers: int = 2) -> Tuple[DataLoader, DataLoader]:
    train_ds, val_ds = BinaryRoadDataset("train"), BinaryRoadDataset("val")
    pin = torch.cuda.is_available()
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                          pin_memory=pin, drop_last=True, worker_init_fn=_worker_init,
                          persistent_workers=num_workers > 0)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    return train_dl, val_dl
