"""
Evaluation utilities: mAP (optical detection) and mIoU (sonar segmentation),
plus per-class metrics and confusion matrix plotting.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix


# ── Segmentation metrics ───────────────────────────────────────────────────────

def compute_iou_per_class(preds: np.ndarray, targets: np.ndarray,
                           num_classes: int) -> np.ndarray:
    """Pixel-level IoU per class over full dataset arrays."""
    iou = np.zeros(num_classes)
    for cls in range(num_classes):
        inter = ((preds == cls) & (targets == cls)).sum()
        union = ((preds == cls) | (targets == cls)).sum()
        iou[cls] = inter / (union + 1e-8)
    return iou


@torch.no_grad()
def evaluate_segmentation(model, loader, num_classes: int, device: str,
                           ignore_indices: tuple = (0, 11)) -> dict:
    """
    Compute pixel accuracy and mIoU, excluding background (0) and wall (11).
    ignore_indices: class indices excluded from mIoU computation.
    """
    model.eval()
    all_preds, all_targets = [], []

    for imgs, masks in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds.flatten())
        all_targets.append(masks.numpy().flatten())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    iou = compute_iou_per_class(all_preds, all_targets, num_classes)
    valid = [i for i in range(num_classes) if i not in ignore_indices]
    miou = iou[valid].mean()

    pixel_acc = (all_preds == all_targets).mean()

    return {
        "miou": float(miou),
        "iou_per_class": iou,
        "pixel_accuracy": float(pixel_acc),
    }


# ── Detection metrics (wraps pycocotools) ─────────────────────────────────────

def compute_map(coco_gt, coco_dt) -> dict:
    """
    Compute COCO mAP metrics.
    coco_gt: pycocotools COCO ground-truth object
    coco_dt: list of COCO-format detection dicts
    """
    from pycocotools.cocoeval import COCOeval
    if not coco_dt:
        return {"mAP50": 0.0, "mAP50_95": 0.0}

    coco_eval = COCOeval(coco_gt, coco_gt.loadRes(coco_dt), "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return {
        "mAP50_95": float(coco_eval.stats[0]),
        "mAP50": float(coco_eval.stats[1]),
        "mAP_small": float(coco_eval.stats[3]),
        "mAP_medium": float(coco_eval.stats[4]),
        "mAP_large": float(coco_eval.stats[5]),
    }


# ── Visualization ─────────────────────────────────────────────────────────────

def plot_class_distribution(counts: dict, title: str = "Class Distribution",
                             save_path: str = None):
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(counts.keys(), counts.values(), color="steelblue")
    ax.bar_label(bars)
    ax.set_title(title)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_iou_per_class(iou: np.ndarray, class_names: list,
                       title: str = "IoU per Class", save_path: str = None):
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(class_names, iou, color="darkcyan")
    ax.bar_label(bars, fmt="%.2f")
    ax.axhline(iou.mean(), color="red", linestyle="--", label=f"mean={iou.mean():.2f}")
    ax.set_title(title)
    ax.set_ylabel("IoU")
    ax.legend()
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_confusion_matrix(preds: np.ndarray, targets: np.ndarray,
                           class_names: list, save_path: str = None):
    cm = confusion_matrix(targets, preds, labels=list(range(len(class_names))),
                          normalize="true")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".2f", xticklabels=class_names,
                yticklabels=class_names, cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (normalized)")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_training_history(history: list, save_path: str = None):
    epochs = [h["epoch"] for h in history]
    train_losses = [h["train_loss"] for h in history]
    val_losses = [h["val_loss"] for h in history]
    mious = [h["miou"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, train_losses, label="train")
    ax1.plot(epochs, val_losses, label="val")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(epochs, mious, color="green")
    ax2.set_title("Validation mIoU")
    ax2.set_xlabel("Epoch")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
