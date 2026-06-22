# Marine Debris Detection — Project Overview

## Goal
Detect underwater marine debris using sonar imagery. Three task tracks:
- **Task A** — Sonar Classification (Turntable-Cropped + Watertank-Cropped)
- **Task B** — Sonar Detection (Watertank-Segmentation, bounding boxes + pixel masks)
- **Task C** — Fusion Comparison (cross-domain pretrain → finetune vs single-domain)

Optical comparison (TrashCan dataset) is queued — not yet downloaded.

---

## Dataset

All data lives in `marine-debris-fls-datasets/md_fls_dataset/data/`.

### Sub-datasets

| Name | Path | Images | Format | Use |
|---|---|---|---|---|
| Turntable-Cropped | `turntable-cropped/<class>/` | 4,892 PNG | class = folder | Classification (18 classes) |
| Watertank-Cropped | `watertank-cropped/<class>/` | 1,955 PNG | class = folder | Classification (11 classes) |
| Watertank-Segmentation | `watertank-segmentation/` | 1,868 PNG | Masks/ + BoxAnnotations/ XML | Detection + Segmentation |
| Quarry-Fullsize | `quarry-fullsize/<session>/` | 7,209 PNG | no annotations | Domain pretraining (optional) |

### Turntable-Cropped — 18 Classes
brown-glass-bottle (210), can (410), drink-carton (428), drink-sachet (222),
glass-bottle (480), glass-jar (432), large-tire (204), metal-bottle (222),
metal-box (176), plastic-bidon (196), plastic-bottle (440), plastic-pipe (220),
plastic-propeller (224), potion-glass-bottle (218), rotating-platform (186),
small-tire (234), valve (200), wrench (240)

### Watertank-Cropped — 10 Classes
bottle (449), can (367), chain (226), drink-carton (349), hook (133),
propeller (137), shampoo-bottle (99), standing-bottle (65), tire (331), valve (208)

### Watertank-Segmentation — 11 Classes + Background + Wall
| Pixel Value | Class |
|---|---|
| 0 | Background |
| 1 | Bottle |
| 2 | Can |
| 3 | Chain |
| 4 | Drink-carton |
| 5 | Hook |
| 6 | Propeller |
| 7 | Shampoo-bottle |
| 8 | Standing-bottle |
| 9 | Tire |
| 10 | Valve |
| 11 | Wall (non-debris) |

Image size: **320 × 480 px**, grayscale sonar.
XML bounding box format: `<x><y><w><h>` (COCO-style, not standard Pascal VOC).

---

## Preprocessing & Normalization

| Dataset | Steps |
|---|---|
| All sonar | CLAHE (clip=2.0, tile=8×8) → Gaussian blur σ=1 |
| Classification input | Resize 224×224 → normalize [0,1] → replicate to 3-channel |
| Detection/Seg input | Resize 640×640 → normalize [0,1] → replicate to 3-channel |
| Class imbalance | Focal loss α=0.25 γ=2 + WeightedRandomSampler |

---

## Project File Map

```
Dataset/
├── marine-debris-fls-datasets/md_fls_dataset/data/   ← DATASET ROOT
├── src/
│   ├── datasets.py      FLSClassificationDataset, FLSDetectionDataset, FLSSegmentationDataset
│   ├── transforms.py    Sonar augmentation pipeline
│   ├── train.py         FocalLoss, DiceLoss, training loop
│   ├── evaluate.py      mAP, mIoU, confusion matrix, plots
│   ├── fusion.py        Cross-domain transfer + decision ensemble
│   └── utils.py         get_device(), MPS/CUDA helpers
├── notebooks/
│   ├── 01_eda.ipynb     Class distribution, imbalance, CLAHE visualization
│   ├── 02_optical.ipynb Pending TrashCan download
│   ├── 03_sonar.ipynb   SegFormer-B2 on Watertank-Segmentation (masks)
│   └── 04_fusion.ipynb  Turntable pretrain → Watertank finetune comparison
├── configs/
│   ├── optical.yaml     Placeholder
│   └── sonar.yaml       Detection config (11 classes, CLAHE)
└── results/             Saved weights, plots, metric logs
```

---

## Environment

```bash
conda activate marine-debris
cd ~/Desktop/ML-Project/Dataset
jupyter notebook
```

PyTorch 2.12, MPS (Apple M4), Python 3.11.

---

## Expected Results

| Approach | Metric | Target |
|---|---|---|
| Watertank classification (ResNet-50) | top-1 acc | > 75% |
| Watertank detection (YOLOv8m) | mAP50 | > 35% |
| Turntable → Watertank transfer | mAP50 | > watertank-only |
| Segmentation (SegFormer-B2) | mIoU | > 30% |
