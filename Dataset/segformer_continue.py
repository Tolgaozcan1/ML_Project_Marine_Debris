import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation
from torch.utils.data import DataLoader
from pathlib import Path
import time
import numpy as np
import cv2
from torch.utils.data import Dataset, WeightedRandomSampler

DEVICE = "cuda"
NUM_CLASSES = 12
_BG = 0
_WALL = 11
OUT = Path("/kaggle/working")
SEG_ROOT = Path("/kaggle/input/datasets/tolgaozcan/marine-debris-dts/marine-debris-fls-datasets/md_fls_dataset/data/watertank-segmentation")


def _split_names(root, split):
    f = root / f"{split}.txt"
    if f.exists():
        return f.read_text().splitlines()
    imgs = sorted((root / "Images").glob("*.png"))
    n = len(imgs)
    t = int(0.7 * n)
    v = int(0.85 * n)
    splits = {"train": imgs[:t], "val": imgs[t:v], "test": imgs[v:]}
    return [p.stem for p in splits[split]]


class SegDataset(Dataset):
    def __init__(self, root, split="train", img_size=512, augment=False):
        self.root = Path(root)
        self.img_size = img_size
        self.augment = augment
        self.names = _split_names(self.root, split)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        n = self.names[idx]
        img = cv2.imread(str(self.root / "Images" / f"{n}.png"), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(self.root / "Masks" / f"{n}.png"), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        if self.augment and np.random.rand() > 0.5:
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)
        if self.augment and np.random.rand() > 0.5:
            img = cv2.flip(img, 0)
            mask = cv2.flip(mask, 0)
        img = torch.from_numpy(img.astype(np.float32).transpose(2, 0, 1) / 255.0)
        mask = torch.from_numpy(mask.astype(np.int64))
        return img, mask

    def make_weighted_sampler(self):
        px = np.zeros(NUM_CLASSES)
        for n in self.names:
            m = cv2.imread(str(self.root / "Masks" / f"{n}.png"), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                for c in np.unique(m):
                    if c < NUM_CLASSES:
                        px[c] += (m == c).sum()
        freq = px / (px.sum() + 1e-8)
        cw = 1.0 / (freq + 1e-8)
        ws = []
        for n in self.names:
            m = cv2.imread(str(self.root / "Masks" / f"{n}.png"), cv2.IMREAD_GRAYSCALE)
            debris = m[(m != _BG) & (m != _WALL)] if m is not None else np.array([])
            ws.append(float(cw[debris].mean()) if len(debris) > 0 else float(cw[_BG]))
        return WeightedRandomSampler(ws, num_samples=len(ws), replacement=True)


class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1.0, ignore=(0, 11)):
        super().__init__()
        self.smooth = smooth
        self.ignore = set(ignore)

    def forward(self, logits, tgt):
        probs = torch.softmax(logits, 1)
        loss = 0
        count = 0
        for c in range(logits.shape[1]):
            if c in self.ignore:
                continue
            p = probs[:, c]
            t = (tgt == c).float()
            i = (p * t).sum()
            d = p.sum() + t.sum() + self.smooth
            loss += 1 - (2 * i + self.smooth) / d
            count += 1
        return loss / max(count, 1)


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, ignore=(0, 11)):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore = ignore

    def forward(self, logits, tgt):
        m = tgt.clone()
        for i in self.ignore:
            m[tgt == i] = -1
        valid = m >= 0
        if not valid.any():
            return logits.sum() * 0
        ce = F.cross_entropy(logits, m.clamp(min=0), reduction='none')
        pt = torch.exp(-ce)
        fl = self.alpha * (1 - pt) ** self.gamma * ce
        return fl[valid].mean()


def seg_loss(logits, masks):
    return 0.5 * FocalLoss()(logits, masks) + 0.5 * DiceLoss()(logits, masks)


def sf_validate(loader):
    model.eval()
    iou_sum = torch.zeros(NUM_CLASSES)
    iou_cnt = torch.zeros(NUM_CLASSES)
    total_loss = 0.0
    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)
            out = model(pixel_values=imgs)
            logits = F.interpolate(out.logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)
            total_loss += seg_loss(logits, masks).item()
            preds = logits.argmax(1)
            for c in range(1, NUM_CLASSES):
                if c == _WALL:
                    continue
                pc = (preds == c)
                tc = (masks == c)
                inter = (pc & tc).sum().item()
                union = (pc | tc).sum().item()
                if union > 0:
                    iou_sum[c] += inter / union
                    iou_cnt[c] += 1
    valid = [i for i in range(1, NUM_CLASSES) if i != _WALL]
    miou = float((iou_sum[valid] / iou_cnt[valid].clamp(min=1)).mean())
    model.train()
    return miou, total_loss / len(loader)


train_ds = SegDataset(SEG_ROOT, "train", 512, augment=True)
val_ds = SegDataset(SEG_ROOT, "val", 512)
test_ds = SegDataset(SEG_ROOT, "test", 512)
train_loader = DataLoader(train_ds, 8, sampler=train_ds.make_weighted_sampler(), num_workers=2, pin_memory=True)
val_loader = DataLoader(val_ds, 8, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_ds, 8, shuffle=False, num_workers=2, pin_memory=True)

model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b2-finetuned-ade-512-512",
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True,
).to(DEVICE)

ckpt = OUT / "seg_segformer_best.pt"

EXTRA = 50
LR = 6e-5
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EXTRA, eta_min=LR / 10)
best = 0.0

for ep in range(1, EXTRA + 1):
    model.train()
    ep_loss = 0
    t = time.time()
    for imgs, masks in train_loader:
        imgs = imgs.to(DEVICE)
        masks = masks.to(DEVICE)
        opt.zero_grad()
        out = model(pixel_values=imgs)
        logits = F.interpolate(out.logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)
        loss = seg_loss(logits, masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ep_loss += loss.item()
    sched.step()
    avg = ep_loss / len(train_loader)
    mins = (time.time() - t) / 60
    log = f"Epoch {ep:3d}/50 | loss {avg:.4f} | {mins:.1f}min"
    if ep % 5 == 0 or ep == EXTRA:
        miou, _ = sf_validate(val_loader)
        log += f" | val_mIoU {miou:.4f}"
        if miou > best:
            best = miou
            torch.save(model.state_dict(), ckpt)
            log += "  *saved*"
    print(log, flush=True)

model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
test_miou, _ = sf_validate(test_loader)
print(f"\nSegFormer Test mIoU (50 epochs): {test_miou:.4f}")
