"""
Augmentation pipelines for optical and sonar modalities.
Uses albumentations for both, with modality-specific settings.
albumentations >= 2.0 API used throughout.
"""

import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ── Optical (TrashCan) ────────────────────────────────────────────────────────

def optical_train_transforms(img_size: int = 640):
    return A.Compose(
        [
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(img_size, img_size, border_mode=0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Rotate(limit=10, p=0.4),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05, p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.RandomScale(scale_limit=0.3, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"], min_visibility=0.3),
    )


def optical_val_transforms(img_size: int = 640):
    return A.Compose(
        [
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(img_size, img_size, border_mode=0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
    )


# ── Sonar (FLS) ───────────────────────────────────────────────────────────────

def sonar_train_transforms(img_size: int = 640):
    """
    Conservative augmentation for sonar — no color/HSV jitter.
    albumentations 2.x API: std_range instead of var_limit for GaussNoise,
    num_holes_range / hole_height_range / hole_width_range for CoarseDropout.
    """
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=15, p=0.4),
            A.ElasticTransform(alpha=120, sigma=12, p=0.2),
            # std_range ≈ sqrt(var_limit): sqrt(0.001)≈0.032, sqrt(0.005)≈0.071
            A.GaussNoise(std_range=(0.03, 0.07), p=0.3),
            A.CoarseDropout(num_holes_range=(1, 4),
                            hole_height_range=(16, 32),
                            hole_width_range=(16, 32), p=0.2),
        ],
        additional_targets={"mask": "mask"},
    )


def sonar_val_transforms(img_size: int = 640):
    return A.Compose(
        [A.Resize(img_size, img_size)],
        additional_targets={"mask": "mask"},
    )


def apply_sonar_transform(transform, img_tensor: torch.Tensor, mask_tensor: torch.Tensor):
    """Helper: tensor → numpy → augment → tensor."""
    img_np   = img_tensor.permute(1, 2, 0).numpy()         # [H, W, 3]
    mask_np  = mask_tensor.numpy().astype(np.uint8)
    out      = transform(image=img_np, mask=mask_np)
    img_out  = torch.from_numpy(out["image"]).permute(2, 0, 1).contiguous()
    mask_out = torch.from_numpy(out["mask"].astype(np.int64)).contiguous()
    return img_out, mask_out
