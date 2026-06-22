"""
Training utilities: focal loss, dice loss, combined loss, and training loop
for the sonar segmentation model.

Optical detection (YOLOv8) training is handled via ultralytics CLI —
see notebooks/02_optical.ipynb.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


# ── Loss functions ─────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Multi-class focal loss for imbalanced segmentation."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 ignore_index: int = -100,
                 extra_ignore_indices: tuple = (11,),
                 reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.extra_ignore = extra_ignore_indices
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [B, C, H, W], targets: [B, H, W]
        ce = F.cross_entropy(logits, targets, ignore_index=self.ignore_index, reduction="none")
        pt = torch.exp(-ce)
        loss = self.alpha * (1 - pt) ** self.gamma * ce
        # zero out extra ignored classes (e.g. wall=11)
        for idx in self.extra_ignore:
            loss = loss * (targets != idx).float()
        return loss.mean() if self.reduction == "mean" else loss


class DiceLoss(nn.Module):
    """
    Soft Dice loss for segmentation, averaged over classes.
    ignore_indices: class indices excluded both as targets AND from pixel-level computation.
    Wall pixels (class 11) are excluded from ALL class computations so they don't
    inflate false-positive denominators.
    """

    def __init__(self, smooth: float = 1.0, ignore_indices: tuple = (0, 11)):
        super().__init__()
        self.smooth = smooth
        self._skip = set(ignore_indices)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        # Build a pixel-level validity mask: ignore ALL pixels whose ground-truth
        # label is in _skip (background and wall).  This prevents wall pixels from
        # contributing false-positive probability mass to debris-class denominators.
        valid_mask = torch.ones_like(targets, dtype=torch.float32)
        for idx in self._skip:
            valid_mask *= (targets != idx).float()

        dice = 0.0
        count = 0
        for cls in range(num_classes):
            if cls in self._skip:
                continue
            pred  = probs[:, cls] * valid_mask
            tgt   = (targets == cls).float() * valid_mask
            inter = (pred * tgt).sum()
            union = pred.sum() + tgt.sum()
            dice += 1 - (2 * inter + self.smooth) / (union + self.smooth)
            count += 1
        return dice / max(count, 1)


class CombinedLoss(nn.Module):
    def __init__(self, focal_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.focal = FocalLoss()   # extra_ignore_indices=(11,) by default
        self.dice  = DiceLoss()    # ignore_indices=(0, 11) by default
        self.fw = focal_weight
        self.dw = dice_weight

    def forward(self, logits, targets):
        return self.fw * self.focal(logits, targets) + self.dw * self.dice(logits, targets)


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader: DataLoader, optimizer, criterion,
                    device: str, scaler=None) -> float:
    model.train()
    total_loss = 0.0
    for imgs, masks in tqdm(loader, desc="Train", leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        if scaler:
            with torch.autocast("cuda"):
                logits = model(imgs)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


_WALL_CLASS = 11  # tank artefact — excluded from mIoU


@torch.no_grad()
def validate(model, loader: DataLoader, criterion, device: str) -> dict:
    model.eval()
    total_loss = 0.0
    num_classes = None
    intersection = None
    union = None

    for imgs, masks in tqdm(loader, desc="Val", leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)

        if num_classes is None:
            num_classes = logits.shape[1]
            intersection = torch.zeros(num_classes, device=device)
            union = torch.zeros(num_classes, device=device)

        total_loss += criterion(logits, masks).item()
        preds = logits.argmax(dim=1)

        for cls in range(num_classes):
            pred_c = preds == cls
            tgt_c = masks == cls
            intersection[cls] += (pred_c & tgt_c).sum()
            union[cls] += (pred_c | tgt_c).sum()

    iou_per_class = (intersection / (union + 1e-8)).cpu().numpy()
    # exclude background (0) and wall (11) from mIoU
    valid = [i for i in range(1, num_classes) if i != _WALL_CLASS]
    return {
        "val_loss": total_loss / len(loader),
        "iou_per_class": iou_per_class,
        "miou": float(iou_per_class[valid].mean()),
    }


def run_training(model, train_loader, val_loader, cfg: dict, device: str):
    """Full training loop with LR scheduling and best-model checkpointing."""
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"], eta_min=cfg["lr"] * 0.01
    )
    criterion = CombinedLoss(focal_weight=cfg.get("focal_weight", 0.5),
                              dice_weight=cfg.get("dice_weight", 0.5))
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

    best_miou = 0.0
    history = []

    for epoch in range(1, cfg["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})
        print(f"Epoch {epoch:3d} | loss {train_loss:.4f} | "
              f"val_loss {val_metrics['val_loss']:.4f} | mIoU {val_metrics['miou']:.4f}")

        if val_metrics["miou"] > best_miou:
            best_miou = val_metrics["miou"]
            torch.save(model.state_dict(), cfg["checkpoint_path"])
            print(f"  -> Saved best model (mIoU={best_miou:.4f})")

    return history
