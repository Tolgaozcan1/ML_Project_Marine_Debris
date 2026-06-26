# Marine Debris Detection in Sonar Imagery

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.12-EE4C2C?logo=pytorch&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Apple%20M4%20%7C%20Kaggle%20T4-lightgrey?logo=apple)
![Status](https://img.shields.io/badge/Status-Complete-success)
![License](https://img.shields.io/badge/License-Academic-informational)

> Automatically detect, localize, and classify underwater marine debris from Forward-Looking Sonar (FLS) imagery using deep learning — enabling autonomous underwater vehicles to map and clean ocean pollution without human divers.

---

## Results

| Task | Model | Dataset | Metric | Score |
|---|---|---|---|---|
| Classification | ResNet-50 (scratch) | Watertank-Cropped (10 cls) | Test Accuracy | **98.59%** |
| Classification | ResNet-50 (transfer) | Watertank-Cropped (10 cls) | Test Accuracy | **98.87%** |
| Classification | ResNet-50 (scratch) | Turntable-Cropped (18 cls) | Test Accuracy | **86.39%** |
| Detection | YOLOv8m | Watertank-Segmentation | mAP50 | **0.937** |
| Detection | YOLOv8m | Watertank-Segmentation | mAP50-95 | **0.697** |
| Segmentation | U-Net + ResNet34 | Watertank-Segmentation | mIoU | **0.638** |
| Segmentation | SegFormer-B2 | Watertank-Segmentation | mIoU | **0.658** |

---

## Pipeline

```
Raw FLS Sonar Images (320×480, grayscale)
          │
          ▼
  Preprocessing
  CLAHE (clip=2.0, tile=8×8) → Gaussian blur σ=1
  Resize → Normalize → 3-channel replicate
          │
          ├──────────────────┬──────────────────────┐
          ▼                  ▼                      ▼
   Classification        Detection              Segmentation
   ResNet-50            YOLOv8m               U-Net + ResNet34
   (scratch/transfer)   (Kaggle T4)           SegFormer-B2
          │                  │                      │
          ▼                  ▼                      ▼
   Class Label         Bounding Box           Pixel Mask
   (10–18 classes)     + Class Label          (11 classes)
```

---

## Dataset

Data from the **Marine Debris FLS Dataset** — captured using an ARIS Explorer 3000 FLS sensor at Heriot-Watt University's Ocean Systems Lab Water Tank.

| Sub-dataset | Images | Classes | Task | Imbalance |
|---|---|---|---|---|
| Watertank-Cropped | 2,364 | 10 | Classification | 6.9× |
| Turntable-Cropped | 4,942 | 18 | Classification | 2.7× |
| Watertank-Segmentation | 1,868 | 10 + bg | Detection & Segmentation | **13×** |

**10 debris classes:** bottle, can, chain, drink-carton, hook, propeller, shampoo-bottle, standing-bottle, tire, valve

> The raw dataset is excluded from this repository (large binaries). See the [dataset paper](https://arxiv.org/abs/2503.22880) for download instructions.

---

## Project Structure

```
ML-Project/
├── Dataset/
│   ├── src/
│   │   ├── datasets.py          # FLSClassification/Detection/SegmentationDataset
│   │   ├── transforms.py        # Sonar augmentation pipeline (CLAHE, flips, etc.)
│   │   ├── train.py             # FocalLoss, DiceLoss, training loop
│   │   ├── evaluate.py          # mAP, mIoU, confusion matrix
│   │   ├── fusion.py            # Cross-domain transfer + decision ensemble
│   │   └── utils.py             # Device helpers (MPS/CUDA)
│   ├── notebooks/
│   │   ├── 01_eda.ipynb         # Class distribution, imbalance, CLAHE visualization
│   │   ├── 01_eda_executed.ipynb
│   │   └── 03_sonar.ipynb       # SegFormer-B2 on Watertank-Segmentation
│   ├── configs/
│   │   └── sonar.yaml           # Detection config (10 classes, CLAHE)
│   ├── results/
│   │   ├── Clean_Academic_Report.pdf
│   │   ├── Project_Update_Report.pdf
│   │   ├── classification_*.log
│   │   ├── yolo_*.log
│   │   └── figures/
│   ├── train_classifier.py      # ResNet-50 classification (local MPS)
│   ├── run_yolo_resume.py       # YOLOv8m with MPS bincount patch
│   ├── train_unet.py            # U-Net + ResNet34 segmentation (local MPS)
│   ├── segformer_continue.py    # SegFormer-B2 for Kaggle T4
│   └── Marine_Debris_Presentation.ipynb
└── Kaggle outputs/
    └── yolov8m.pt               # Detection checkpoint (52 MB)
```

---

## Getting Started

**1. Clone and set up the environment**

```bash
git clone https://github.com/Tolgaozcan1/ML_Project_Marine_Debris.git
cd ML_Project_Marine_Debris/Dataset
conda create -n marine-debris python=3.11
conda activate marine-debris
pip install -r requirements.txt
```

**2. Download the dataset**

Follow the instructions in the [Marine Debris FLS Dataset paper](https://arxiv.org/abs/2503.22880) and place data under:

```
Dataset/marine-debris-fls-datasets/md_fls_dataset/data/
```

**3. Run the EDA notebook**

```bash
jupyter notebook notebooks/01_eda.ipynb
```

**4. Train a model**

```bash
# Classification
python train_classifier.py

# Detection (requires Kaggle T4 or CUDA — MPS has a known bincount bug)
python run_yolo_resume.py

# Segmentation
python train_unet.py
```

---

## Key Engineering Notes

| Issue | Solution |
|---|---|
| YOLOv8 MPS crash (320 GiB alloc) | `run_yolo_resume.py` monkey-patches `counts.max()` to run on CPU |
| SegFormer Metal crash on Apple M4 | Trained on Kaggle T4 GPU instead |
| Class imbalance (13× tire vs standing-bottle) | Focal loss α=0.25 γ=2 + `WeightedRandomSampler` |
| PDF generation | Uses WeasyPrint via conda-forge — **not** fpdf2 (layout breaks) |

---

## Reports

| Document | Description |
|---|---|
| [`Clean_Academic_Report.pdf`](Dataset/results/Clean_Academic_Report.pdf) | Full technical report with methodology and results |
| [`Project_Update_Report.pdf`](Dataset/results/Project_Update_Report.pdf) | Mid-project progress update |

---

## References

1. [The Marine Debris FLS Datasets (2025)](https://arxiv.org/abs/2503.22880)
2. [FLS Semantic Segmentation with U-Net (2021)](https://arxiv.org/abs/2108.06800)
3. [Deep Neural Networks for Marine Debris Detection in Sonar (2019)](https://arxiv.org/abs/1905.05241)

---

*University ML Project — Apple M4 + Kaggle T4 · PyTorch 2.12 · Python 3.11*
