# CLAUDE.md — Implementation Progress Tracker

Last updated: June 2026

---

## Environment
- **Conda env**: `marine-debris` (Python 3.11, PyTorch 2.12, MPS)
- **Project root**: `~/Desktop/ML-Project /Dataset/`
- **Dataset root**: `marine-debris-fls-datasets/md_fls_dataset/data/`
- **Activate**: `conda activate marine-debris && cd ~/Desktop/ML-Project\ /Dataset`

---

## Phase Status — ALL COMPLETE ✅

### Phase 0 — Setup ✅
- Conda env `marine-debris` created (MPS available on M4)
- Dependencies: torch, ultralytics, transformers, albumentations, pycocotools, opencv, segmentation-models-pytorch, fpdf2, weasyprint
- `src/utils.py`, `src/transforms.py`, `src/train.py`, `src/evaluate.py`, `src/datasets.py`, `src/fusion.py`
- `configs/sonar.yaml` — 10 detection classes (no bg, no wall)

### Phase 1 — Data Inspection ✅
- Format confirmed: PNG images + Pascal VOC XML + PNG pixel masks
- `FLSClassificationDataset`, `FLSDetectionDataset`, `FLSSegmentationDataset` in `src/datasets.py`

### Phase 2 — EDA ✅
- CLAHE visualization: `results/eda_fls_clahe.png`
- Pixel distribution: `results/eda_fls_pixel_dist.png`
- Debris-only imbalance confirmed: **13× tire vs standing-bottle**

### Phase 3 — Classification ✅
- ResNet-50 — Watertank-Cropped: **val acc = 100.00%** (355/355)
- ResNet-50 — Turntable-Cropped: **val acc = 99.19%** (736/742)
- Checkpoints: `results/cls_watertank_best.pt`, `results/cls_turntable_best.pt`

### Phase 4 — Detection ✅
- Trained on Kaggle T4 GPU (MPS bincount bug prevented local training)
- YOLOv8m — 80 epochs — **mAP50 = 0.967**, mAP50-95 = 0.702
- Training time: 1.072 hours
- Weights: `../Kaggle outputs/yolov8m.pt` (52 MB)
- MPS fix: `run_yolo_resume.py` monkey-patches `v8DetectionLoss.preprocess` so `counts.max()` runs on CPU

### Phase 5 — Segmentation ✅
- U-Net + ResNet34 trained locally (MPS-stable): **test mIoU = 0.638** (60 epochs)
- SegFormer-B2 trained on Kaggle T4: **test mIoU = 0.658** (30 epochs)
- SegFormer abandoned on M4 Metal — attention backward crashes GPU driver
- Weights LOST (Kaggle session expired) — scores documented from training logs
- Scripts: `train_unet.py`, `segformer_continue.py`

### Phase 6 — Report & Visualizations ✅
- All figures generated: `python generate_visualizations.py`
- Figures saved to: `results/figures/`
- Two PDF reports generated (HTML+CSS via WeasyPrint):
  - `results/Clean_Academic_Report.pdf` — full technical report
  - `results/Simple_Guide_Report.pdf` — beginner-friendly guide (10 chapters)
- Generator scripts: `create_html_report.py`, `create_html_simple_report.py`

---

## Final Results

| Task | Model | Dataset | Metric | Score | Notes |
|---|---|---|---|---|---|
| Classification | ResNet-50 | Watertank-Cropped (10 cls) | top-1 acc | **100.00%** | 355/355 val |
| Classification | ResNet-50 | Turntable-Cropped (18 cls) | top-1 acc | **99.19%** | 736/742 val |
| Detection | YOLOv8m | Watertank-Seg | mAP50 | **0.967** | beats published research |
| Detection | YOLOv8m | Watertank-Seg | mAP50-95 | 0.702 | |
| Segmentation | U-Net + ResNet34 | Watertank-Seg | mIoU (cls 1–10) | **0.638** | paper baseline: 0.748 |
| Segmentation | SegFormer-B2 | Watertank-Seg | mIoU (cls 1–10) | **0.658** | 30 epochs Kaggle T4 |

### YOLOv8m Per-Class mAP50
| hook | shampoo-bottle | standing-bottle | tire | drink-carton | propeller | bottle | chain | valve | can |
|---|---|---|---|---|---|---|---|---|---|
| 0.995 | 0.995 | 0.995 | 0.989 | 0.984 | 0.975 | 0.968 | 0.964 | 0.914 | 0.891 |

---

## Dataset Quick Stats

| Sub-dataset | Images | Classes | Imbalance |
|---|---|---|---|
| Watertank-Cropped | 2,364 | 10 | 449/65 = 6.9× |
| Turntable-Cropped | 4,942 | 18 | 480/176 = 2.7× |
| Watertank-Seg (pixel, debris only) | 1,868 | 10 | **13× tire vs standing-bottle** |

---

## Key File Paths

```
DATA_ROOT         = marine-debris-fls-datasets/md_fls_dataset/data/
WATERTANK_C       = DATA_ROOT/watertank-cropped/
TURNTABLE_C       = DATA_ROOT/turntable-cropped/
WATERTANK_S       = DATA_ROOT/watertank-segmentation/   # Images/ Masks/ BoxAnnotations/
YOLO_DATA         = results/yolo_dataset/               # converted XML→YOLO labels
YOLO_WEIGHTS      = ../Kaggle outputs/yolovb8m.pt
CLS_WATERTANK_PT  = results/cls_watertank_best.pt
CLS_TURNTABLE_PT  = results/cls_turntable_best.pt
FIGURES           = results/figures/
PDF_TECHNICAL     = results/Clean_Academic_Report.pdf
PDF_SIMPLE        = results/Simple_Guide_Report.pdf
```

---

## Scripts Reference

| Script | Purpose |
|---|---|
| `train_classifier.py` | ResNet-50 classification training (local MPS) |
| `run_yolo_resume.py` | YOLOv8m training with MPS bincount monkey-patch |
| `train_unet.py` | U-Net + ResNet34 segmentation (local MPS, stable) |
| `segformer_continue.py` | SegFormer-B2 for Kaggle T4 GPU |
| `generate_visualizations.py` | Generates all 6 report figures to results/figures/ |
| `create_html_report.py` | Builds Clean_Academic_Report.pdf (WeasyPrint) |
| `create_html_simple_report.py` | Builds Simple_Guide_Report.pdf (WeasyPrint) |

---

## Known Issues & Fixes

| Issue | Fix |
|---|---|
| YOLOv8 MPS crash: 320 GiB allocation | Monkey-patch in `run_yolo_resume.py`: `counts.max()` moved to CPU |
| SegFormer Metal crash on M4 | Abandoned local; trained on Kaggle T4 |
| Kaggle session expired, weights lost | Results documented from training logs |
| fpdf2 text layout jumbled | Replaced with HTML+CSS→WeasyPrint pipeline |
| WeasyPrint needs Pango on Mac | Install via: `conda install -c conda-forge weasyprint` (NOT pip) |
