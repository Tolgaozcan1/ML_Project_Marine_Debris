"""
Phase 5 — SegFormer-B2 segmentation training script.

Run from Dataset/ directory with marine-debris conda env active.
Requires: src/segformer_mps_patch.py (MPS view→contiguous fix).
"""

import sys, os, warnings
warnings.filterwarnings("ignore")

# Must be set before torch is imported to take effect
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")  # disable MPS memory limit enforcement

sys.path.insert(0, os.path.dirname(__file__))

# Apply MPS patch BEFORE importing or loading SegFormer
from src.segformer_mps_patch import apply_segformer_mps_patch
apply_segformer_mps_patch()

import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path

from transformers import SegformerForSemanticSegmentation

from src.datasets import FLSSegmentationDataset
from src.transforms import sonar_train_transforms, sonar_val_transforms, apply_sonar_transform
from src.train import FocalLoss, DiceLoss
from src.utils import get_device

# ───────────────────────── config ──────────────────────────
IMG_SIZE    = 512
BATCH_SIZE  = 4
EPOCHS      = 30
LR          = 6e-5
WEIGHT_DECAY= 1e-2
VAL_EVERY   = 5
SAVE_PATH   = Path("results/seg_segformer_best.pt")
NUM_CLASSES = 12
_WALL       = 11

SEG_ROOT = Path("marine-debris-fls-datasets/md_fls_dataset/data/watertank-segmentation")
# ────────────────────────────────────────────────────────────

device = get_device()
print(f"Device: {device}")

# ── Datasets ──
train_aug = sonar_train_transforms(img_size=IMG_SIZE)
val_aug   = sonar_val_transforms(img_size=IMG_SIZE)

def train_tf(img, msk): return apply_sonar_transform(train_aug, img, msk)
def val_tf(img, msk):   return apply_sonar_transform(val_aug,   img, msk)

train_ds = FLSSegmentationDataset(str(SEG_ROOT), split='train', transform=train_tf, img_size=IMG_SIZE)
val_ds   = FLSSegmentationDataset(str(SEG_ROOT), split='val',   transform=val_tf,   img_size=IMG_SIZE)
test_ds  = FLSSegmentationDataset(str(SEG_ROOT), split='test',  transform=val_tf,   img_size=IMG_SIZE)

print(f"Dataset: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

# Weighted sampler for pixel imbalance
sampler = train_ds.make_weighted_sampler()
print("Weighted sampler built.")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,    num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ── Model ──
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b2-finetuned-ade-512-512",
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True,
    attn_implementation="eager",   # disables MPS SDPA → fixes Metal internal error on M-chip
).to(device)

# ── Loss ──
fl = FocalLoss(alpha=0.25, gamma=2.0, ignore_index=0, extra_ignore_indices=(11,))
dl = DiceLoss(smooth=1.0, ignore_indices=(0, 11))

def combined_loss(logits, masks):
    return 0.5 * fl(logits, masks) + 0.5 * dl(logits, masks)

# ── Optimizer with cosine LR ──
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR/10)

# ── Validation ──
def validate(loader, desc="Val"):
    model.eval()
    iou_sum  = torch.zeros(NUM_CLASSES)
    iou_cnt  = torch.zeros(NUM_CLASSES)
    total_loss = 0.0
    with torch.no_grad():
        for imgs, masks in loader:
            imgs  = imgs.to(device).contiguous()
            masks = masks.to(device).contiguous()
            out    = model(pixel_values=imgs)
            logits = F.interpolate(out.logits, size=masks.shape[-2:],
                                   mode='bilinear', align_corners=False)
            total_loss += combined_loss(logits, masks).item()
            preds = logits.argmax(1)  # (B, H, W)
            for cls in range(NUM_CLASSES):
                if cls in (0, _WALL): continue
                pred_c  = (preds == cls)
                true_c  = (masks == cls)
                inter   = (pred_c & true_c).sum().item()
                union   = (pred_c | true_c).sum().item()
                if union > 0:
                    iou_sum[cls] += inter / union
                    iou_cnt[cls] += 1

    valid_cls = [i for i in range(1, NUM_CLASSES) if i != _WALL]
    miou = float((iou_sum[valid_cls] / iou_cnt[valid_cls].clamp(min=1)).mean())
    avg_loss = total_loss / len(loader)
    model.train()
    return miou, avg_loss

# ── Training loop ──
best_miou = 0.0
t_start = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    ep_loss = 0.0
    ep_t = time.time()

    for imgs, masks in train_loader:
        imgs  = imgs.to(device).contiguous()
        masks = masks.to(device).contiguous()
        optimizer.zero_grad()
        out    = model(pixel_values=imgs)
        logits = F.interpolate(out.logits, size=masks.shape[-2:],
                               mode='bilinear', align_corners=False)
        loss = combined_loss(logits, masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        ep_loss += loss.item()

    scheduler.step()
    avg_loss = ep_loss / len(train_loader)
    elapsed  = time.time() - ep_t

    log = f"Epoch {epoch:3d}/{EPOCHS} | loss {avg_loss:.4f} | {elapsed/60:.1f}min"

    if epoch % VAL_EVERY == 0 or epoch == EPOCHS:
        miou, val_loss = validate(val_loader)
        log += f" | val_loss {val_loss:.4f} | val_mIoU {miou:.4f}"
        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), SAVE_PATH)
            log += "  *saved*"

    print(log, flush=True)

# ── Final test evaluation ──
print("\n=== Final test evaluation ===")
model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
test_miou, test_loss = validate(test_loader, desc="Test")
print(f"Test mIoU (classes 1–10, excl wall): {test_miou:.4f}")
print(f"Best val mIoU: {best_miou:.4f}")
print(f"Total time: {(time.time()-t_start)/3600:.2f}h")
