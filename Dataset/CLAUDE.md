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
- Live-recomputed (`train_classifier.py`, see `results/classification/classification_runs.json`):
  ResNet-50 scratch — Watertank-Cropped: **test acc = 98.59%**; transfer (Turntable&rarr;Watertank): **98.87%**;
  ResNet-50 scratch — Turntable-Cropped: **test acc = 86.39%** (random split — same pipeline, ~12pts below
  the old undocumented checkpoint's 99.19%/98.38%, evidence of leakage risk in the random image-level split).
- `--leakage_safe` flag on `train_classifier.py` reruns all stages under an approximate, leakage-reducing
  contiguous-block split (`src/datasets.py` `split_strategy="blocked"`); results saved under `*_blocked` keys
  in the same JSON, reported side by side in the report (see `Clean_Academic_Report.pdf` §4.1/§5.6).
- Old checkpoints (`results/cls_watertank_best.pt`, `results/cls_turntable_best.pt`) are from a lost training
  script — undocumented config, kept only for historical confusion-matrix figures.

### Phase 4 — Detection ✅
- Trained on Kaggle T4 GPU (MPS bincount bug prevented local training)
- YOLOv8m — 80 epochs — live-recomputed via `.val()` on saved checkpoint: **mAP50 = 0.937**, mAP50-95 = 0.697
  (an older training-time log reported 0.967; that figure is no longer cited anywhere in the report)
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
| Classification | ResNet-50 (scratch) | Watertank-Cropped (10 cls) | test acc | **98.59%** | random split; live-recomputed |
| Classification | ResNet-50 (transfer) | Watertank-Cropped (10 cls) | test acc | **98.87%** | random split; live-recomputed |
| Classification | ResNet-50 (scratch) | Turntable-Cropped (18 cls) | test acc | **86.39%** | random split; live-recomputed |
| Detection | YOLOv8m | Watertank-Seg | mAP50 | **0.937** | live-recomputed from saved checkpoint |
| Detection | YOLOv8m | Watertank-Seg | mAP50-95 | 0.697 | |
| Segmentation (preliminary) | U-Net + ResNet34 | Watertank-Seg | mIoU (cls 1–10) | **0.638** | paper baseline 0.748; not reproducible, see Known Issues |
| Segmentation (preliminary) | SegFormer-B2 | Watertank-Seg | mIoU (cls 1–10) | **0.658** | 30 epochs Kaggle T4; not reproducible |

### YOLOv8m Per-Class mAP50 (live-recomputed, same checkpoint as aggregate above)
| hook | tire | bottle | chain | drink-carton | propeller | standing-bottle | shampoo-bottle | can | valve |
|---|---|---|---|---|---|---|---|---|---|
| 0.995 | 0.994 | 0.989 | 0.984 | 0.984 | 0.945 | 0.913 | 0.866 | 0.855 | 0.848 |

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
