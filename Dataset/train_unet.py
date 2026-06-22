"""
Phase 5 (alt) — U-Net + ResNet34 segmentation (paper baseline, mIoU target 0.748).

Uses segmentation_models_pytorch which has pure conv ops — no attention, no MPS crash.
Run from Dataset/ directory with marine-debris conda env active.

  conda activate marine-debris && cd ~/Desktop/ML-Project\ /Dataset
  caffeinate -i python train_unet.py | tee results/unet_train.log
"""
import os, sys, warnings, time
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path

import segmentation_models_pytorch as smp

from src.datasets import FLSSegmentationDataset
from src.transforms import sonar_train_transforms, sonar_val_transforms, apply_sonar_transform
from src.train import FocalLoss, DiceLoss
from src.utils import get_device

# ── Config ──────────────────────────────────────────────────────────────────
IMG_SIZE    = 256
BATCH_SIZE  = 8
EPOCHS      = 60
LR          = 1e-4
WEIGHT_DECAY= 1e-4
VAL_EVERY   = 5
SAVE_PATH   = Path("results/seg_unet_best.pt")
NUM_CLASSES = 12
_WALL       = 11

SEG_ROOT = Path("marine-debris-fls-datasets/md_fls_dataset/data/watertank-segmentation")
# ────────────────────────────────────────────────────────────────────────────

device = get_device()
print(f"Device: {device}")

train_aug = sonar_train_transforms(img_size=IMG_SIZE)
val_aug   = sonar_val_transforms(img_size=IMG_SIZE)

def train_tf(img, msk): return apply_sonar_transform(train_aug, img, msk)
def val_tf(img, msk):   return apply_sonar_transform(val_aug,   img, msk)

train_ds = FLSSegmentationDataset(str(SEG_ROOT), split='train', transform=train_tf, img_size=IMG_SIZE)
val_ds   = FLSSegmentationDataset(str(SEG_ROOT), split='val',   transform=val_tf,   img_size=IMG_SIZE)
test_ds  = FLSSegmentationDataset(str(SEG_ROOT), split='test',  transform=val_tf,   img_size=IMG_SIZE)

print(f"Dataset: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

sampler      = train_ds.make_weighted_sampler()
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,  num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,  num_workers=0)

# ── Model: U-Net + ResNet34, ImageNet-pretrained encoder ────────────────────
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=NUM_CLASSES,
).to(device)
print(f"Model: U-Net + ResNet34  params={sum(p.numel() for p in model.parameters()):,}")

# ── Loss ────────────────────────────────────────────────────────────────────
fl = FocalLoss(alpha=0.25, gamma=2.0, ignore_index=0, extra_ignore_indices=(_WALL,))
dl = DiceLoss(smooth=1.0, ignore_indices=(0, _WALL))

def combined_loss(logits, masks):
    return 0.5 * fl(logits, masks) + 0.5 * dl(logits, masks)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR / 10)

# ── Validation ──────────────────────────────────────────────────────────────
def validate(loader):
    model.eval()
    iou_sum = torch.zeros(NUM_CLASSES)
    iou_cnt = torch.zeros(NUM_CLASSES)
    total_loss = 0.0
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            logits = F.interpolate(logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)
            total_loss += combined_loss(logits, masks).item()
            preds = logits.argmax(1)
            for cls in range(1, NUM_CLASSES):
                if cls == _WALL:
                    continue
                pred_c = (preds == cls)
                true_c = (masks == cls)
                inter  = (pred_c & true_c).sum().item()
                union  = (pred_c | true_c).sum().item()
                if union > 0:
                    iou_sum[cls] += inter / union
                    iou_cnt[cls] += 1
    valid = [i for i in range(1, NUM_CLASSES) if i != _WALL]
    miou = float((iou_sum[valid] / iou_cnt[valid].clamp(min=1)).mean())
    model.train()
    return miou, total_loss / len(loader)

# ── Training loop ────────────────────────────────────────────────────────────
best_miou = 0.0
t_start = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    ep_loss = 0.0
    ep_t = time.time()

    for imgs, masks in train_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        logits = F.interpolate(logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)
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

# ── Final test ───────────────────────────────────────────────────────────────
print("\n=== Final test evaluation ===")
model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
test_miou, test_loss = validate(test_loader)
print(f"Test mIoU (classes 1–10, excl wall): {test_miou:.4f}")
print(f"Best val mIoU:                        {best_miou:.4f}")
print(f"Paper baseline (U-Net+ResNet34):       0.7481")
print(f"Total time: {(time.time()-t_start)/3600:.2f}h")
