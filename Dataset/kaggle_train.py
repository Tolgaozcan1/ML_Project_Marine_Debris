"""
Kaggle/Colab fallback training script for Marine Debris Detection.
Run this on Kaggle (free T4 GPU) if the local Mac crashes.

SETUP ON KAGGLE:
1. Upload the dataset folder (marine-debris-fls-datasets/) as a Kaggle Dataset
   - Go to kaggle.com → Datasets → New Dataset → upload a zip of marine-debris-fls-datasets/
2. Create a new Kaggle Notebook (GPU T4 x2)
3. Add your dataset under "Add Data"
4. Copy this script into a code cell and run

SETUP ON GOOGLE COLAB:
1. Upload dataset to Google Drive
2. Mount Drive:  from google.colab import drive; drive.mount('/content/drive')
3. Set DATA_ROOT below to your Drive path
4. Run: !pip install segmentation-models-pytorch albumentations transformers
"""

# ── Install dependencies (uncomment on Kaggle/Colab) ──────────────────────
# import subprocess
# subprocess.run(["pip", "install", "-q", "segmentation-models-pytorch",
#                 "albumentations", "transformers", "timm"])

import os, sys, time, warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from pathlib import Path

# ── Paths — edit for your environment ─────────────────────────────────────
# Kaggle:  DATA_ROOT = Path("/kaggle/input/<your-dataset-name>/md_fls_dataset/data")
# Colab:   DATA_ROOT = Path("/content/drive/MyDrive/marine-debris/md_fls_dataset/data")
# Local:   DATA_ROOT = Path("marine-debris-fls-datasets/md_fls_dataset/data")
DATA_ROOT  = Path("/kaggle/input/marine-debris-fls/md_fls_dataset/data")
SAVE_DIR   = Path("/kaggle/working")

SEG_ROOT   = DATA_ROOT / "watertank-segmentation"
SAVE_PATH  = SAVE_DIR / "seg_unet_best.pt"
LOG_PATH   = SAVE_DIR / "unet_train.log"

# ── Config ─────────────────────────────────────────────────────────────────
IMG_SIZE    = 256
BATCH_SIZE  = 16      # T4 has 16 GB — can go bigger
EPOCHS      = 60
LR          = 1e-4
WEIGHT_DECAY= 1e-4
VAL_EVERY   = 5
NUM_CLASSES = 12
_BG, _WALL  = 0, 11

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}  |  GPU: {torch.cuda.get_device_name(0) if DEVICE=='cuda' else 'none'}")

# ── Inline dataset (copy of src/datasets.py logic, self-contained) ─────────
import cv2, numpy as np
from torch.utils.data import Dataset

WATERTANK_SEG_CLASSES = [
    "background", "bottle", "can", "chain", "drink-carton",
    "hook", "propeller", "shampoo-bottle", "standing-bottle",
    "tire", "valve", "wall",
]

def _list_split(root: Path, split: str):
    split_file = root / f"{split}.txt"
    if split_file.exists():
        names = split_file.read_text().splitlines()
        return sorted(names)
    imgs = sorted((root / "Images").glob("*.png"))
    n = len(imgs); t, v = int(0.7*n), int(0.85*n)
    splits = {"train": imgs[:t], "val": imgs[t:v], "test": imgs[v:]}
    return [p.stem for p in splits[split]]

class FLSSegDataset(Dataset):
    def __init__(self, root, split="train", img_size=256):
        self.root = Path(root)
        self.img_size = img_size
        self.names = _list_split(self.root, split)
        self.classes = WATERTANK_SEG_CLASSES

    def __len__(self): return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        img_path  = self.root / "Images" / f"{name}.png"
        mask_path = self.root / "Masks"  / f"{name}.png"
        img  = cv2.imread(str(img_path),  cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        img  = cv2.resize(img,  (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        img  = img.astype(np.float32) / 255.0
        img  = torch.from_numpy(img.transpose(2, 0, 1))
        mask = torch.from_numpy(mask.astype(np.int64))
        return img, mask

    def make_weighted_sampler(self):
        pixel_counts = np.zeros(len(self.classes))
        for name in self.names:
            m = cv2.imread(str(self.root/"Masks"/f"{name}.png"), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                for c in np.unique(m):
                    if c < len(pixel_counts):
                        pixel_counts[c] += (m == c).sum()
        freq = pixel_counts / (pixel_counts.sum() + 1e-6)
        class_weight = 1.0 / (freq + 1e-6)
        sample_weights = []
        for name in self.names:
            m = cv2.imread(str(self.root/"Masks"/f"{name}.png"), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                debris = m[(m != _BG) & (m != _WALL)]
                w = class_weight[debris].mean() if len(debris) > 0 else class_weight[_BG]
            else:
                w = 1.0
            sample_weights.append(float(w))
        return WeightedRandomSampler(sample_weights, num_samples=len(self.names), replacement=True)

# ── Inline losses ───────────────────────────────────────────────────────────
class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1.0, ignore_indices=(0, 11)):
        super().__init__()
        self.smooth = smooth
        self.ignore = set(ignore_indices)

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        C = logits.shape[1]
        loss = 0.0; count = 0
        for c in range(C):
            if c in self.ignore: continue
            p = probs[:, c]
            t = (targets == c).float()
            inter = (p * t).sum()
            denom = p.sum() + t.sum() + self.smooth
            loss += 1 - (2 * inter + self.smooth) / denom
            count += 1
        return loss / max(count, 1)

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, ignore_indices=(0, 11)):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore = ignore_indices

    def forward(self, logits, targets):
        mask = targets.clone()
        for idx in self.ignore:
            mask[targets == idx] = -1
        valid = mask >= 0
        if not valid.any(): return logits.sum() * 0
        ce = F.cross_entropy(logits, mask.clamp(min=0), reduction='none')
        pt = torch.exp(-ce)
        fl = self.alpha * (1 - pt) ** self.gamma * ce
        return fl[valid].mean()

def combined_loss(logits, masks):
    return 0.5 * FocalLoss()(logits, masks) + 0.5 * DiceLoss()(logits, masks)

# ── Build datasets & loaders ────────────────────────────────────────────────
import segmentation_models_pytorch as smp

train_ds = FLSSegDataset(SEG_ROOT, split="train", img_size=IMG_SIZE)
val_ds   = FLSSegDataset(SEG_ROOT, split="val",   img_size=IMG_SIZE)
test_ds  = FLSSegDataset(SEG_ROOT, split="test",  img_size=IMG_SIZE)
print(f"Dataset: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

sampler      = train_ds.make_weighted_sampler()
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,    num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# ── Model ───────────────────────────────────────────────────────────────────
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=NUM_CLASSES,
).to(DEVICE)
print(f"Model: U-Net + ResNet34  params={sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR/10)

# ── Validation ──────────────────────────────────────────────────────────────
def validate(loader):
    model.eval()
    iou_sum = torch.zeros(NUM_CLASSES)
    iou_cnt = torch.zeros(NUM_CLASSES)
    total_loss = 0.0
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            logits = model(imgs)
            logits = F.interpolate(logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)
            total_loss += combined_loss(logits, masks).item()
            preds = logits.argmax(1)
            for cls in range(1, NUM_CLASSES):
                if cls == _WALL: continue
                pred_c = (preds == cls); true_c = (masks == cls)
                inter  = (pred_c & true_c).sum().item()
                union  = (pred_c | true_c).sum().item()
                if union > 0:
                    iou_sum[cls] += inter / union
                    iou_cnt[cls] += 1
    valid = [i for i in range(1, NUM_CLASSES) if i != _WALL]
    miou = float((iou_sum[valid] / iou_cnt[valid].clamp(min=1)).mean())
    model.train()
    return miou, total_loss / len(loader)

# ── Training ─────────────────────────────────────────────────────────────────
best_miou = 0.0
log_lines = []
t_start = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    ep_loss = 0.0
    ep_t = time.time()
    for imgs, masks in train_loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
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
    log_lines.append(log)

LOG_PATH.write_text("\n".join(log_lines))

# ── Test ─────────────────────────────────────────────────────────────────────
print("\n=== Final test evaluation ===")
model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
test_miou, _ = validate(test_loader)
print(f"Test mIoU (classes 1–10, excl wall): {test_miou:.4f}")
print(f"Best val mIoU:                        {best_miou:.4f}")
print(f"Paper baseline (U-Net+ResNet34):       0.7481")
print(f"Total time: {(time.time()-t_start)/3600:.2f}h")
