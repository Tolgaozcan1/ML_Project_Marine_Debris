"""
Marine Debris Detection — Kaggle Training Notebook
Covers Phase 4 (YOLO detection) + Phase 5 (U-Net + SegFormer segmentation)

HOW TO USE:
  1. Run create_kaggle_upload.sh to build the upload zip
  2. Go to kaggle.com → Datasets → New Dataset → upload marine_debris_kaggle.zip
  3. Create a new Kaggle Notebook → Add Data → select your dataset
  4. Set accelerator to GPU T4 x2
  5. Paste this file into a code cell (or upload as a script and run it)
  6. Set DATA_ROOT, YOLO_ROOT below to match your dataset path, then Run All

Results are saved to /kaggle/working/ and can be downloaded from Output tab.
"""

# ─── 0. Install dependencies ─────────────────────────────────────────────────
import subprocess, sys

def pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)

pip("segmentation-models-pytorch", "albumentations", "timm", "ultralytics")

import os, time, warnings, math
warnings.filterwarnings("ignore")

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ─── Paths ───────────────────────────────────────────────────────────────────
# Adjust these to match where you uploaded the dataset on Kaggle
DATA_ROOT  = Path("/kaggle/input/marine-debris-fls/data")
YOLO_ROOT  = Path("/kaggle/input/marine-debris-fls/yolo_dataset")
OUT        = Path("/kaggle/working")
OUT.mkdir(exist_ok=True)

SEG_ROOT   = DATA_ROOT / "watertank-segmentation"

# ─── Shared constants ────────────────────────────────────────────────────────
NUM_CLASSES = 12
_BG, _WALL  = 0, 11
WATERTANK_SEG_CLASSES = [
    "background", "bottle", "can", "chain", "drink-carton",
    "hook", "propeller", "shampoo-bottle", "standing-bottle",
    "tire", "valve", "wall",
]

# ─── 1. Dataset ───────────────────────────────────────────────────────────────
def _split_names(root: Path, split: str):
    f = root / f"{split}.txt"
    if f.exists():
        return f.read_text().splitlines()
    imgs = sorted((root / "Images").glob("*.png"))
    n = len(imgs); t, v = int(0.7*n), int(0.85*n)
    return [p.stem for p in {"train": imgs[:t], "val": imgs[t:v], "test": imgs[v:]}[split]]

class SegDataset(Dataset):
    def __init__(self, root, split="train", img_size=256, augment=False):
        self.root = Path(root)
        self.img_size = img_size
        self.augment = augment
        self.names = _split_names(self.root, split)

    def __len__(self): return len(self.names)

    def __getitem__(self, idx):
        n = self.names[idx]
        img  = cv2.imread(str(self.root/"Images"/f"{n}.png"), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(self.root/"Masks"/ f"{n}.png"), cv2.IMREAD_GRAYSCALE)
        img  = cv2.resize(img,  (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        if self.augment and np.random.rand() > 0.5:
            img  = cv2.flip(img,  1)
            mask = cv2.flip(mask, 1)
        if self.augment and np.random.rand() > 0.5:
            img  = cv2.flip(img,  0)
            mask = cv2.flip(mask, 0)
        img  = torch.from_numpy(img.astype(np.float32).transpose(2,0,1) / 255.0)
        mask = torch.from_numpy(mask.astype(np.int64))
        return img, mask

    def make_weighted_sampler(self):
        px = np.zeros(NUM_CLASSES)
        for n in self.names:
            m = cv2.imread(str(self.root/"Masks"/f"{n}.png"), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                for c in np.unique(m):
                    if c < NUM_CLASSES: px[c] += (m==c).sum()
        freq = px / (px.sum() + 1e-8)
        cw   = 1.0 / (freq + 1e-8)
        ws   = []
        for n in self.names:
            m = cv2.imread(str(self.root/"Masks"/f"{n}.png"), cv2.IMREAD_GRAYSCALE)
            debris = m[(m!=_BG)&(m!=_WALL)] if m is not None else np.array([])
            ws.append(float(cw[debris].mean()) if len(debris) > 0 else float(cw[_BG]))
        return WeightedRandomSampler(ws, num_samples=len(ws), replacement=True)

# ─── 2. Losses ────────────────────────────────────────────────────────────────
class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1.0, ignore=(0,11)):
        super().__init__(); self.smooth = smooth; self.ignore = set(ignore)
    def forward(self, logits, tgt):
        probs = torch.softmax(logits, 1)
        loss = count = 0
        for c in range(logits.shape[1]):
            if c in self.ignore: continue
            p = probs[:,c]; t = (tgt==c).float()
            i = (p*t).sum(); d = p.sum()+t.sum()+self.smooth
            loss += 1 - (2*i+self.smooth)/d; count += 1
        return loss / max(count,1)

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, ignore=(0,11)):
        super().__init__(); self.alpha=alpha; self.gamma=gamma; self.ignore=ignore
    def forward(self, logits, tgt):
        m = tgt.clone()
        for i in self.ignore: m[tgt==i] = -1
        valid = m>=0
        if not valid.any(): return logits.sum()*0
        ce = F.cross_entropy(logits, m.clamp(min=0), reduction='none')
        pt = torch.exp(-ce)
        fl = self.alpha*(1-pt)**self.gamma*ce
        return fl[valid].mean()

def seg_loss(logits, masks):
    return 0.5*FocalLoss()(logits,masks) + 0.5*DiceLoss()(logits,masks)

# ─── 3. Validation ────────────────────────────────────────────────────────────
def validate(model, loader, device):
    model.eval()
    iou_sum = torch.zeros(NUM_CLASSES); iou_cnt = torch.zeros(NUM_CLASSES)
    total_loss = 0.0
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            logits = F.interpolate(logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)
            total_loss += seg_loss(logits, masks).item()
            preds = logits.argmax(1)
            for c in range(1, NUM_CLASSES):
                if c == _WALL: continue
                pc = (preds==c); tc = (masks==c)
                inter = (pc&tc).sum().item(); union = (pc|tc).sum().item()
                if union > 0: iou_sum[c]+=inter/union; iou_cnt[c]+=1
    valid = [i for i in range(1,NUM_CLASSES) if i!=_WALL]
    miou = float((iou_sum[valid]/iou_cnt[valid].clamp(min=1)).mean())
    model.train()
    return miou, total_loss/len(loader)

# ─── 4. PHASE 4 — YOLO Detection ──────────────────────────────────────────────
def run_yolo():
    print("\n" + "="*60)
    print("PHASE 4 — YOLOv8m Detection")
    print("="*60)
    from ultralytics import YOLO

    # Fix dataset.yaml paths to point to Kaggle input
    yaml_src = YOLO_ROOT / "dataset.yaml"
    yaml_dst = OUT / "yolo_dataset.yaml"
    txt = yaml_src.read_text()
    # Replace local path with Kaggle path
    txt = txt.replace(
        str(YOLO_ROOT.parent / "yolo_dataset"),
        str(YOLO_ROOT)
    )
    # Make paths absolute
    for split in ("train", "val", "test"):
        txt = txt.replace(f"\n{split}:", f"\n{split}: {YOLO_ROOT}/{split}")
    yaml_dst.write_text(txt)

    model = YOLO("yolov8m.pt")
    model.train(
        data=str(yaml_dst),
        epochs=80,
        imgsz=640,
        batch=16,          # T4 has 16 GB
        device=0,          # CUDA:0
        project=str(OUT / "yolo_sonar"),
        name="yolov8m_watertank",
        exist_ok=True,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.1,
    )
    print(f"\nYOLO best weights: {OUT}/yolo_sonar/yolov8m_watertank/weights/best.pt")
    results = model.val(data=str(yaml_dst))
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")

# ─── 5. PHASE 5a — U-Net + ResNet34 ──────────────────────────────────────────
def run_unet():
    print("\n" + "="*60)
    print("PHASE 5a — U-Net + ResNet34 Segmentation (paper baseline)")
    print("="*60)
    import segmentation_models_pytorch as smp

    IMG_SIZE=256; BATCH=16; EPOCHS=60; LR=1e-4
    train_ds = SegDataset(SEG_ROOT, "train", IMG_SIZE, augment=True)
    val_ds   = SegDataset(SEG_ROOT, "val",   IMG_SIZE)
    test_ds  = SegDataset(SEG_ROOT, "test",  IMG_SIZE)

    train_loader = DataLoader(train_ds, BATCH, sampler=train_ds.make_weighted_sampler(), num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   BATCH, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  BATCH, shuffle=False, num_workers=2, pin_memory=True)

    model = smp.Unet("resnet34", encoder_weights="imagenet", in_channels=3, classes=NUM_CLASSES).to(DEVICE)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR/10)

    save = OUT / "seg_unet_best.pt"
    best = 0.0
    for ep in range(1, EPOCHS+1):
        model.train(); ep_loss=0; t=time.time()
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            opt.zero_grad()
            logits = F.interpolate(model(imgs), size=masks.shape[-2:], mode='bilinear', align_corners=False)
            loss = seg_loss(logits, masks); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            ep_loss += loss.item()
        sched.step()
        log = f"Epoch {ep:3d}/{EPOCHS} | loss {ep_loss/len(train_loader):.4f} | {(time.time()-t)/60:.1f}min"
        if ep%5==0 or ep==EPOCHS:
            miou, vl = validate(model, val_loader, DEVICE)
            log += f" | val_mIoU {miou:.4f}"
            if miou > best:
                best=miou; torch.save(model.state_dict(), save); log+="  *saved*"
        print(log, flush=True)

    model.load_state_dict(torch.load(save, map_location=DEVICE))
    test_miou, _ = validate(model, test_loader, DEVICE)
    print(f"\nU-Net Test mIoU: {test_miou:.4f}  (paper baseline: 0.7481)")

# ─── 6. PHASE 5b — SegFormer-B2 ──────────────────────────────────────────────
def run_segformer():
    print("\n" + "="*60)
    print("PHASE 5b — SegFormer-B2 Segmentation")
    print("="*60)
    from transformers import SegformerForSemanticSegmentation

    IMG_SIZE=512; BATCH=8; EPOCHS=30; LR=6e-5
    train_ds = SegDataset(SEG_ROOT, "train", IMG_SIZE, augment=True)
    val_ds   = SegDataset(SEG_ROOT, "val",   IMG_SIZE)
    test_ds  = SegDataset(SEG_ROOT, "test",  IMG_SIZE)

    train_loader = DataLoader(train_ds, BATCH, sampler=train_ds.make_weighted_sampler(), num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   BATCH, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  BATCH, shuffle=False, num_workers=2, pin_memory=True)

    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512",
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    ).to(DEVICE)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR/10)

    save = OUT / "seg_segformer_best.pt"
    best = 0.0

    def sf_validate(loader):
        model.eval(); iou_sum=torch.zeros(NUM_CLASSES); iou_cnt=torch.zeros(NUM_CLASSES); tl=0
        with torch.no_grad():
            for imgs, masks in loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                out = model(pixel_values=imgs)
                logits = F.interpolate(out.logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)
                tl += seg_loss(logits, masks).item()
                preds = logits.argmax(1)
                for c in range(1,NUM_CLASSES):
                    if c==_WALL: continue
                    pc=(preds==c); tc=(masks==c)
                    inter=(pc&tc).sum().item(); union=(pc|tc).sum().item()
                    if union>0: iou_sum[c]+=inter/union; iou_cnt[c]+=1
        valid=[i for i in range(1,NUM_CLASSES) if i!=_WALL]
        model.train()
        return float((iou_sum[valid]/iou_cnt[valid].clamp(min=1)).mean()), tl/len(loader)

    for ep in range(1, EPOCHS+1):
        model.train(); ep_loss=0; t=time.time()
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            opt.zero_grad()
            out = model(pixel_values=imgs)
            logits = F.interpolate(out.logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)
            loss = seg_loss(logits, masks); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            ep_loss += loss.item()
        sched.step()
        log = f"Epoch {ep:3d}/{EPOCHS} | loss {ep_loss/len(train_loader):.4f} | {(time.time()-t)/60:.1f}min"
        if ep%5==0 or ep==EPOCHS:
            miou, vl = sf_validate(val_loader)
            log += f" | val_mIoU {miou:.4f}"
            if miou > best:
                best=miou; torch.save(model.state_dict(), save); log+="  *saved*"
        print(log, flush=True)

    model.load_state_dict(torch.load(save, map_location=DEVICE))
    test_miou, _ = sf_validate(test_loader)
    print(f"\nSegFormer Test mIoU: {test_miou:.4f}  (paper baseline: 0.7481)")

# ─── Run all phases ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    t0 = time.time()
    run_yolo()
    run_unet()
    run_segformer()
    print(f"\nAll training done in {(time.time()-t0)/3600:.2f}h")
    print(f"Outputs in: {OUT}")
