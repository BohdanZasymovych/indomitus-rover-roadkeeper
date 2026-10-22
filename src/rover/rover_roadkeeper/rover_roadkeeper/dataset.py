import os
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from rover_roadkeeper.config import (
    SEG_FRAMES_DIR,
    SEG_MASKS_DIR,
    ORIG_WIDTH,
    ORIG_HEIGHT,
    HUD_MASK_BOX,
    TRAIN_SPLIT_COUNT,
    BATCH_SIZE,
    IGNORE_INDEX,
)


def rgb_to_class_indices(mask_rgb: np.ndarray) -> np.ndarray:
    """
    Vectorized conversion from RGB semantic mask to class indices (0..4).
    RGB Mapping:
      - (0, 0, 0)       -> 0: ignore
      - (255, 0, 255)   -> 1: road
      - (0, 0, 255)     -> 2: sky
      - (0, 255, 255)   -> 3: vegetation
      - (255, 255, 0)   -> 4: obstacle
    """
    H, W = mask_rgb.shape[:2]
    class_mask = np.zeros((H, W), dtype=np.int64)

    r = mask_rgb[:, :, 0]
    g = mask_rgb[:, :, 1]
    b = mask_rgb[:, :, 2]

    # Road: high Red, low Green, high Blue (Magenta)
    is_road = (r > 128) & (g < 128) & (b > 128)
    class_mask[is_road] = 1

    # Sky: low Red, low Green, high Blue (Blue)
    is_sky = (r < 128) & (g < 128) & (b > 128)
    class_mask[is_sky] = 2

    # Vegetation: low Red, high Green, high Blue (Cyan)
    is_veg = (r < 128) & (g > 128) & (b > 128)
    class_mask[is_veg] = 3

    # Obstacle: high Red, high Green, low Blue (Yellow)
    is_obs = (r > 128) & (g > 128) & (b < 128)
    class_mask[is_obs] = 4

    # Always mask bottom-left burned-in HUD box to IGNORE_INDEX
    y_min, y_max = HUD_MASK_BOX["y_min"], HUD_MASK_BOX["y_max"]
    x_min, x_max = HUD_MASK_BOX["x_min"], HUD_MASK_BOX["x_max"]
    class_mask[y_min:y_max, x_min:x_max] = IGNORE_INDEX

    return class_mask


def get_train_transforms(target_height: int = 364, target_width: int = 720) -> A.Compose:
    """
    Albumentations pipeline for training with IR camera degradation augmentations.
    """
    return A.Compose([
        A.Resize(height=target_height, width=target_width),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.6),
        A.OneOf([
            A.MotionBlur(blur_limit=7, p=0.5),
            A.GaussianBlur(blur_limit=5, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        ], p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])


def get_val_transforms(target_height: int = 364, target_width: int = 720) -> A.Compose:
    """
    Validation pipeline without random degradations.
    """
    return A.Compose([
        A.Resize(height=target_height, width=target_width),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])


class RoadSegmentationDataset(Dataset):
    """
    Dataset for Track-1 road segmentation.
    Enforces contiguous temporal block split:
      - Train: first TRAIN_SPLIT_COUNT frames (001000..001239)
      - Val: remaining frames (001240..001300)
    """

    def __init__(
        self,
        frames_dir: Optional[Path] = None,
        masks_dir: Optional[Path] = None,
        split: str = "train",
        transform: Optional[A.Compose] = None,
    ):
        self.frames_dir = Path(frames_dir or SEG_FRAMES_DIR)
        self.masks_dir = Path(masks_dir or SEG_MASKS_DIR)
        self.split = split
        self.transform = transform

        if not self.frames_dir.exists():
            raise FileNotFoundError(f"Frames directory not found: {self.frames_dir}")
        if not self.masks_dir.exists():
            raise FileNotFoundError(f"Masks directory not found: {self.masks_dir}")

        all_frame_files = sorted([
            f for f in self.frames_dir.glob("*.jpg")
            if not f.name.endswith(":Zone.Identifier")
        ])

        if len(all_frame_files) == 0:
            raise RuntimeError(f"No JPG images found in {self.frames_dir}")

        # Contiguous temporal split
        if split == "train":
            self.frame_files = all_frame_files[:TRAIN_SPLIT_COUNT]
        elif split == "val":
            self.frame_files = all_frame_files[TRAIN_SPLIT_COUNT:]
        elif split == "all":
            self.frame_files = all_frame_files
        else:
            raise ValueError(f"Unknown split: {split}. Choose from ['train', 'val', 'all'].")

    def __len__(self) -> int:
        return len(self.frame_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_path = self.frame_files[idx]
        stem = frame_path.stem
        mask_path = self.masks_dir / f"{stem}.png"

        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found for frame: {mask_path}")

        # Load image (RGB)
        image_bgr = cv2.imread(str(frame_path))
        if image_bgr is None:
            raise ValueError(f"Failed to load image: {frame_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Load mask (RGB)
        mask_bgr = cv2.imread(str(mask_path))
        if mask_bgr is None:
            raise ValueError(f"Failed to load mask: {mask_path}")
        mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)

        # Convert RGB mask to class index (0..4)
        class_mask = rgb_to_class_indices(mask_rgb)

        # Apply Albumentations
        if self.transform is not None:
            augmented = self.transform(image=image_rgb, mask=class_mask)
            image_tensor = augmented["image"]
            mask_tensor = torch.as_tensor(augmented["mask"], dtype=torch.long)
        else:
            # Fallback tensor conversion
            image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
            mask_tensor = torch.from_numpy(class_mask).long()

        return image_tensor, mask_tensor


def get_dataloaders(
    batch_size: int = BATCH_SIZE,
    num_workers: int = 2,
    frames_dir: Optional[Path] = None,
    masks_dir: Optional[Path] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Helper function to instantiate train and validation DataLoaders.
    """
    train_transform = get_train_transforms(ORIG_HEIGHT, ORIG_WIDTH)
    val_transform = get_val_transforms(ORIG_HEIGHT, ORIG_WIDTH)

    train_dataset = RoadSegmentationDataset(
        frames_dir=frames_dir,
        masks_dir=masks_dir,
        split="train",
        transform=train_transform,
    )
    val_dataset = RoadSegmentationDataset(
        frames_dir=frames_dir,
        masks_dir=masks_dir,
        split="val",
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return train_loader, val_loader
