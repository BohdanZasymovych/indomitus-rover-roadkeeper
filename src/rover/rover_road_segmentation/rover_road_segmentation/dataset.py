"""Binary road dataset over track-1/seg/ (frames/*.jpg + markdown/*.png masks).

Labels: 1 = road (magenta), 0 = everything else, 255 = ignore (HUD box,
out-of-image pixels after augmentation).
Split:  contiguous temporal blocks only (neighbouring frames are nearly
        identical): first TRAIN_SPLIT_COUNT frames train, the rest validate.
Images go through processing.preprocess() exactly like at inference time;
labels stay at the frame's own resolution and the model's logits are
upsampled to them, just as they are upsampled to the camera frame on the rover.
"""
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from rover_road_segmentation import config as C
from rover_road_segmentation.processing import hud_box, preprocess


def rgb_mask_to_labels(mask_rgb):
    """RGB colour-coded mask -> uint8 {0 background, 1 road, 255 ignore}."""
    r, g, b = mask_rgb[..., 0], mask_rgb[..., 1], mask_rgb[..., 2]
    labels = np.full(mask_rgb.shape[:2], C.LABEL_BACKGROUND, np.uint8)
    labels[(r > 127) & (g < 128) & (b > 127)] = C.LABEL_ROAD
    x0, y0, x1, y1 = hud_box(*labels.shape)
    labels[y0:y1, x0:x1] = C.LABEL_IGNORE
    return labels


def _synthetic_negative(h, w, rng):
    """A frame with no road at all: black link-loss screen or a big glare blob."""
    if rng.random() < 0.5:
        img = np.full((h, w), rng.integers(0, 25), np.uint8)
        cv2.putText(img, 'Connecting...', (w // 4, h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    w / 700, 255, 2, cv2.LINE_AA)
    else:
        img = np.full((h, w), rng.integers(30, 90), np.uint8)
        cx, cy = int(rng.uniform(0.3, 0.7) * w), int(rng.uniform(0.3, 0.7) * h)
        cv2.circle(img, (cx, cy), int(rng.uniform(0.15, 0.4) * h), 255, -1)
        img = cv2.GaussianBlur(img, (0, 0), h / 12)
    img = np.clip(img + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)
    return cv2.merge([img, img, img]), np.zeros((h, w), np.uint8)


def augment(img, lbl, rng):
    """Geometric + IR sensor degradations on a raw BGR frame and its labels."""
    h, w = lbl.shape
    if rng.random() < C.SYNTHETIC_NEGATIVE_PROB:
        return _synthetic_negative(h, w, rng)

    # Scale / anisotropic stretch / rotation / shift: different mounts, pitch on
    # bumps, and the 16:9 sources (1280x720) squeezed into the 714x364 input.
    if rng.random() < 0.8:
        angle = rng.uniform(-5, 5)
        s = rng.uniform(0.9, 1.15)
        sx, sy = s * rng.uniform(0.88, 1.12), s
        a = np.deg2rad(angle)
        rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) @ np.diag([sx, sy])
        centre = np.array([w / 2, h / 2])
        shift = rng.uniform(-0.05, 0.05, size=2) * (w, h)
        M = np.hstack([rot, (centre - rot @ centre + shift)[:, None]]).astype(np.float32)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        lbl = cv2.warpAffine(lbl, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=C.LABEL_IGNORE)

    if rng.random() < 0.5:
        img, lbl = img[:, ::-1], lbl[:, ::-1]

    # Resolution: low-res relays (700x364) and upscaled streams look softer.
    if rng.random() < 0.3:
        f = rng.uniform(0.35, 0.8)
        small = cv2.resize(img, (max(16, int(w * f)), max(16, int(h * f))), interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    img = img.astype(np.float32)
    img = np.clip(img * rng.uniform(0.7, 1.3) + rng.uniform(-30, 30), 0, 255)   # IR auto-gain
    if rng.random() < 0.5:
        img = 255.0 * (img / 255.0) ** rng.uniform(0.7, 1.5)
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
            img = cv2.filter2D(img, -1, kernel)                                    # motion blur
    if rng.random() < 0.3:
        img = img + rng.normal(0, rng.uniform(3, 12), img.shape)                  # sensor noise
    img = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(25, 75))])
        if ok:
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)                              # compression
    return np.ascontiguousarray(img), np.ascontiguousarray(lbl)


class SegDataset(Dataset):

    def __init__(self, data_dir, split='train', augment_frames=None, seed=C.SEED):
        frames_dir = Path(data_dir) / 'seg' / 'frames'
        masks_dir = Path(data_dir) / 'seg' / 'markdown'   # the folder really is named "markdown"
        frames = sorted(frames_dir.glob('*.jpg'))
        if not frames:
            raise FileNotFoundError(f'no frames in {frames_dir}')
        if split == 'train':
            frames = frames[:C.TRAIN_SPLIT_COUNT]
        elif split == 'val':
            frames = frames[C.TRAIN_SPLIT_COUNT:]
        elif split != 'all':
            raise ValueError("split must be 'train', 'val' or 'all'")

        self.names = [f.stem for f in frames]
        self.augment = (split == 'train') if augment_frames is None else augment_frames
        self.rng = np.random.default_rng(seed)
        self.images, self.labels = [], []   # 301 small frames: keep them all in RAM
        for f in frames:
            img = cv2.imread(str(f), cv2.IMREAD_COLOR)
            m = cv2.imread(str(masks_dir / f'{f.stem}.png'), cv2.IMREAD_COLOR)
            if img is None or m is None:
                raise FileNotFoundError(f'cannot read the pair for {f.stem}')
            if m.shape[:2] != img.shape[:2]:
                m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            self.images.append(img)
            self.labels.append(rgb_mask_to_labels(cv2.cvtColor(m, cv2.COLOR_BGR2RGB)))
        shapes = {lbl.shape for lbl in self.labels}
        if len(shapes) != 1:
            raise ValueError(f'all seg frames must share one size, got {shapes}')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, lbl = self.images[idx], self.labels[idx]
        if self.augment:
            img, lbl = augment(img, lbl, self.rng)
        x, _ = preprocess(img, 'bgr')
        x = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1).float().div_(255.0)
        return x, torch.from_numpy(lbl.astype(np.int64))


def _worker_init(_):
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(torch.initial_seed() % 2**32)


def make_loaders(data_dir, batch_size=C.BATCH_SIZE, num_workers=2):
    train_ds, val_ds = SegDataset(data_dir, 'train'), SegDataset(data_dir, 'val')
    pin = torch.cuda.is_available()
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                          pin_memory=pin, drop_last=True, worker_init_fn=_worker_init,
                          persistent_workers=num_workers > 0)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    return train_dl, val_dl
