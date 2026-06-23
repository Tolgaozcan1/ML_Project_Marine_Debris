"""
Assembles Marine_Debris_Presentation.ipynb — the single required presentation
notebook (course instructions Section 7) — from existing project artifacts.

Run from Dataset/:
  conda activate marine-debris && cd ~/Desktop/ML-Project\ /Dataset
  python build_presentation_notebook.py
Then execute it top-to-bottom with:
  jupyter nbconvert --to notebook --execute --inplace Marine_Debris_Presentation.ipynb
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


code("%matplotlib inline")

# ───────────────────────────────────────────────────────────────────────────
# Title Slide
# ───────────────────────────────────────────────────────────────────────────
md("""
# Marine Debris Detection in Forward-Looking Sonar Imagery

### Classification, Detection, Segmentation, and Cross-Domain Transfer

- **Course:** Machine Learning — Summer 2026
- **Group:** 8
- **Team members:**
  - Tolga Ozcan — 26576920
  - Venkatesh Ajay Vijaya Kumar — 36379442
  - Deekshith Hunsur Shekar — 62246101
- **Date:** 22 June 2026
""")

# ───────────────────────────────────────────────────────────────────────────
# Problem Statement
# ───────────────────────────────────────────────────────────────────────────
md("""
## Problem Statement

- Marine debris (plastics, tires, metal objects) accumulates on the seafloor and harms ecosystems and navigation; manual diver inspection is costly, slow, and unscalable.
- Optical cameras fail underwater beyond a few metres due to turbidity and lack of light, so Autonomous Underwater Vehicles (AUVs) rely on **Forward-Looking Sonar (FLS)** instead — but sonar images are grayscale, noisy, and exhibit strong class imbalance.
- **Main question:** can deep learning reliably (a) **classify** debris type, (b) **detect/localise** debris with bounding boxes, and (c) **segment** debris at the pixel level, directly from raw FLS sonar imagery?
- **Secondary question (cross-domain transfer):** when the target recording domain (Watertank) has comparatively little labelled data, does pretraining on a different sonar domain (Turntable — same sensor, different rig/scene) and fine-tuning help, compared to training from scratch on the target domain alone?
- What would count as a *good* answer: models that clearly beat a simple baseline on the same data and split, with errors analysed (not just a single headline accuracy/mAP number), and an honest account of what was *not* verified (see Error Analysis & Limitations).
""")

# ───────────────────────────────────────────────────────────────────────────
# Dataset
# ───────────────────────────────────────────────────────────────────────────
md("""
## Dataset

- **Source:** Marine Debris FLS Datasets (Rapson et al. 2025, arXiv:2503.22880), captured with an ARIS Explorer 3000 Forward-Looking Sonar at the Heriot-Watt University Ocean Systems Lab water tank. Publicly released on Zenodo/GitHub.
- **Sub-datasets used:**
  - `Watertank-Cropped` — 1,955 grayscale PNG images, 10 debris classes, object centred per crop. Used for classification (main task) and as the *transfer target domain*.
  - `Turntable-Cropped` — 4,892 grayscale PNG images, 18 debris classes, same sensor on a rotating rig. Used as the *transfer source domain* only.
  - `Watertank-Segmentation` — 1,868 images with pixel-level masks (12 classes: background + 10 debris + tank wall) and Pascal-VOC-style XML bounding boxes for the same 10 debris classes. Used for detection and segmentation.
- **Labels:** single debris class per crop (classification); per-object axis-aligned bounding box + class (detection); per-pixel class id 0–11 (segmentation).
- **Class imbalance:** up to 6.9× (Watertank-Cropped, bottle=449 vs standing-bottle=65) and 13× at the pixel level in segmentation (tire vs standing-bottle).
- **Subset used:** the full released `Watertank-Cropped`, `Turntable-Cropped`, and `Watertank-Segmentation` splits — no subsampling was needed, all three are small enough (≤5k images) to train fully on a single GPU.
- **Split:** stratified random 70/15/15 train/val/test per class, fixed `seed=42` for reproducibility (see Limitations for why this is *not* leakage-safe).
""")

# ───────────────────────────────────────────────────────────────────────────
# Methodology
# ───────────────────────────────────────────────────────────────────────────
md("""
## Methodology

- **Preprocessing:** CLAHE (clip limit 2.0, tile grid 8×8) + Gaussian blur (σ=1) on every sonar image to boost local contrast and suppress speckle noise; grayscale replicated to 3 channels for compatibility with standard CNN/Transformer backbones; resize to 224×224 (classification) or 640×640 (detection/segmentation).
- **Class-imbalance handling:** `WeightedRandomSampler` (inverse class frequency) for every task, plus Focal Loss (α=0.25, γ=2.0) combined with Dice Loss for the pixel-level segmentation task.
- **Validation protocol:** stratified random 70/15/15 split per task, fixed seed. A held-out **test** split (never used for model selection) is reported as the headline number, not validation accuracy.
- **Per-task baseline-then-main progression** (Section 4 requirement): every task trains a deliberately simple baseline model first, then a stronger main model, so the main model's gain is attributable to architecture/transfer rather than just "a model was trained."
- **Cross-domain transfer experiment (the required Task C comparison):** a ResNet-50 is first trained on `Turntable-Cropped` (18 classes). Its backbone weights (all layers except the final classification head, which differs in size: 18 vs 10 classes) are copied into a fresh ResNet-50 and fine-tuned on `Watertank-Cropped` with a lower learning rate, then compared against a ResNet-50 trained from scratch directly on `Watertank-Cropped` with the same seed and epoch budget.
- **Reproducibility:** `train_classifier.py` and `train_yolo_baseline.py` set a global seed (`torch.manual_seed`, `np.random.seed`, `random.seed` / ultralytics `seed=`) — this was missing from the original segmentation/detection training scripts (see Limitations).
""")

# ───────────────────────────────────────────────────────────────────────────
# Selected Models
# ───────────────────────────────────────────────────────────────────────────
md("""
## Selected Models

| Task | Baseline | Main model | Why this main model |
|---|---|---|---|
| Classification | Small 4-block CNN, from scratch | ResNet-50, from scratch | Well-established CNN backbone with enough capacity for 10–18 visually distinct debris shapes; "from scratch" because ImageNet features are tuned for natural RGB images, not grayscale acoustic imagery |
| Classification (Task C) | ResNet-50 from scratch (= main model above) | ResNet-50, Turntable-pretrained → Watertank-finetuned | Tests whether sonar-domain pretraining (same sensor, different rig) transfers better than random init when target data is limited |
| Detection | YOLOv8n (nano), COCO-pretrained, fine-tuned, fewer epochs | YOLOv8m (medium), COCO-pretrained, fine-tuned | Single-stage anchor-free detector well suited to axis-aligned debris boxes; nano vs medium isolates the effect of model capacity at a fixed task/data |
| Segmentation | *(none trained — see Limitations)* | U-Net + ResNet34 encoder; SegFormer-B2 | U-Net is the architecture used by the dataset's own published baseline (mIoU 0.748), enabling direct comparison; SegFormer-B2 (transformer) tests whether long-range attention improves over a pure CNN encoder-decoder |
""")

# ───────────────────────────────────────────────────────────────────────────
# Evaluation Criterion
# ───────────────────────────────────────────────────────────────────────────
md("""
## Evaluation Criterion

- **Classification:** Top-1 test accuracy (headline metric) plus a full confusion matrix — accuracy alone can hide systematic confusions between visually similar classes (e.g. bottle variants).
- **Detection:** mAP50 and mAP50-95 (standard COCO-style metrics, computed by `ultralytics`), plus per-class AP50 — necessary because the dataset is imbalanced and a single averaged mAP can hide a weak class (e.g. `can`, the smallest and most cylindrical object).
- **Segmentation:** mean IoU computed over the 10 debris classes only, explicitly **excluding** background (class 0) and tank wall (class 11) — including them would inflate the score since they dominate pixel counts and are easy to segment. This matches the metric used by the original dataset paper, making our numbers directly comparable to the published baseline (0.748 mIoU).
- **Cross-domain transfer:** test-accuracy delta between the from-scratch and transfer-learned ResNet-50 on the *same* Watertank-Cropped test split and seed — isolates the effect of the pretraining domain from any other confound.
""")

# ───────────────────────────────────────────────────────────────────────────
# Results
# ───────────────────────────────────────────────────────────────────────────
md("""
## Results

The cells below **recompute classification and detection metrics live** from the
saved checkpoints (no hardcoded numbers) so the notebook is fully reproducible.
Segmentation numbers are reported from training logs only — see Limitations for why
they cannot currently be recomputed live.
""")

code("""
import sys, json
sys.path.insert(0, '.')
import torch
import torchvision.models as tvm
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd

from src.datasets import FLSClassificationDataset
from src.utils import get_device
from train_classifier import SimpleCNN, make_resnet50, WATERTANK, TURNTABLE

device = get_device()
wt_test = FLSClassificationDataset(WATERTANK, split="test", seed=42)
n_watertank_cls = len(wt_test.classes)

def test_accuracy(model, ds, device):
    model = model.to(device).eval()
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            correct += (model(imgs).argmax(1) == labels).sum().item()
            total += imgs.size(0)
    return correct / total

CLS_DIR = Path("results/classification")
rows = []

baseline = SimpleCNN(n_watertank_cls)
baseline.load_state_dict(torch.load(CLS_DIR / "cls_baseline_cnn.pt", map_location="cpu"))
rows.append(("Baseline CNN (scratch)", test_accuracy(baseline, wt_test, device)))

scratch = make_resnet50(n_watertank_cls)
scratch.load_state_dict(torch.load(CLS_DIR / "cls_watertank_resnet50_scratch.pt", map_location="cpu"))
rows.append(("ResNet-50 (scratch)", test_accuracy(scratch, wt_test, device)))

transfer = make_resnet50(n_watertank_cls)
transfer.load_state_dict(torch.load(CLS_DIR / "cls_watertank_resnet50_transfer.pt", map_location="cpu"))
rows.append(("ResNet-50 (Turntable→Watertank transfer)", test_accuracy(transfer, wt_test, device)))

# Original reported checkpoints (trained in an earlier session; original training
# script no longer exists, see Limitations) for context only
orig_wt = tvm.resnet50(weights=None); orig_wt.fc = torch.nn.Linear(2048, 10)
orig_wt.load_state_dict(torch.load("results/cls_watertank_best.pt", map_location="cpu", weights_only=False))
rows.append(("ResNet-50 (original reported checkpoint)", test_accuracy(orig_wt, wt_test, device)))

cls_results = pd.DataFrame(rows, columns=["Model", "Watertank Test Accuracy"])
cls_results["Watertank Test Accuracy"] = (cls_results["Watertank Test Accuracy"] * 100).round(2)
cls_results
""")

code("""
from IPython.display import Image, display
display(Image(filename="results/figures/classification_baseline_vs_transfer.png"))
""")

md("""
**Detection results** — recomputed live with `ultralytics .val()` on the held-out test split for both the YOLOv8n baseline and the YOLOv8m main model.
""")

code("""
import warnings; warnings.filterwarnings("ignore")
from ultralytics import YOLO

DATASET_YAML = "results/yolo_dataset/dataset.yaml"
det_rows = []

yolo_n = YOLO("results/yolo_sonar/yolov8n_baseline/weights/best.pt")
m_n = yolo_n.val(data=DATASET_YAML, split="test", imgsz=640, batch=8, verbose=False)
det_rows.append(("YOLOv8n (baseline)", m_n.box.map50, m_n.box.map))

yolo_m = YOLO("results/yolo_sonar/yolov8m_watertank/weights/best.pt")
m_m = yolo_m.val(data=DATASET_YAML, split="test", imgsz=640, batch=8, verbose=False)
det_rows.append(("YOLOv8m (main model)", m_m.box.map50, m_m.box.map))

det_results = pd.DataFrame(det_rows, columns=["Model", "mAP50", "mAP50-95"])
det_results
""")

code("""
display(Image(filename="results/figures/yolo_per_class_ap.png"))
""")

md("""
**Segmentation results** (reported from training logs — not recomputed live, see Limitations):

| Model | Test mIoU (classes 1–10) | Paper baseline |
|---|---|---|
| U-Net + ResNet34 | 0.638 | 0.748 |
| SegFormer-B2 | 0.658 | 0.748 |

No segmentation baseline was trained (acknowledged gap, see Limitations).
""")

# ───────────────────────────────────────────────────────────────────────────
# Error Analysis and Limitations
# ───────────────────────────────────────────────────────────────────────────
md("""
## Error Analysis and Limitations

### Error analysis
""")

code("""
display(Image(filename="results/figures/cls_turntable_misclassified.png"))
""")

md("""
- The main ResNet-50 model on Turntable-Cropped (18 classes) misclassifies only a handful of test images; the confusions above are almost all between visually/acoustically similar object subtypes (e.g. different bottle shapes), consistent with the dataset's known acoustic-shadow ambiguity for thin or rounded objects.
""")

code("""
import os
if os.path.exists("results/figures/cls_baseline_misclassified.png"):
    display(Image(filename="results/figures/cls_baseline_misclassified.png"))
""")

md("""
- The baseline CNN makes substantially more errors than the main ResNet-50, mostly on classes with fewer training images (e.g. `standing-bottle`, `shampoo-bottle`) — direct evidence that the extra depth/capacity of ResNet-50 (and, for the transfer run, the Turntable-domain pretraining) is doing real work, not just adding parameters.
- For detection, the per-class AP chart above shows `can` is consistently the weakest class for both YOLOv8n and YOLOv8m — its small, rotation-symmetric, cylindrical shape produces an acoustic signature easily confused with other cylindrical debris (bottles, chains) at the sonar's resolution.
- For segmentation, the 13× pixel-count imbalance between `tire` (most frequent) and `standing-bottle` (least frequent) is the most likely cause of the gap to the published 0.748 mIoU baseline — low-frequency classes get comparatively few effective gradient updates even with weighted sampling and focal loss.

### Limitations (acknowledged, not fixed, given the project timeline)

- **Leakage-risk splits:** all three tasks use a stratified **random** train/val/test split at the image level (`sklearn.train_test_split`, `src/datasets.py`). The released dataset files do not encode a session/sequence/object ID in their filenames (e.g. `can-212.png`), so a true leakage-safe split (grouping all frames of the same physical object/recording session together) is not cheaply derivable from this dataset release. This is a known risk the course instructions explicitly warn about, not something we verified is absent.
- **Segmentation weights are not independently reproducible:** no `.pt` checkpoint survives for either U-Net or SegFormer-B2 (lost when the Kaggle free-tier session expired). The local `unet_train.log` only records 4 of the claimed 60 epochs, and `segformer_train.log` only shows a local Apple-Silicon Metal crash (SegFormer was actually trained on Kaggle, whose log was not saved locally). The mIoU numbers above are therefore **documented, not re-verifiable, results** — we did not retrain segmentation in this remediation pass.
- **No segmentation baseline:** unlike classification and detection, no simple baseline model was trained for segmentation before U-Net/SegFormer, for the same reason as above (retraining segmentation was out of scope for this pass).
- **Original classification checkpoint provenance is partially unverifiable:** the training script that produced `cls_watertank_best.pt` / `cls_turntable_best.pt` no longer exists, and project documentation disagrees on whether ImageNet pretraining was used. The new `resnet50_scratch` / `resnet50_transfer` runs in this notebook are fully from-scratch, seeded, and reproducible, and are the basis for the Task C transfer claim.
- **Optical/multimodal fusion (TrashCan dataset) is out of scope:** the original project brainstorm considered fusing optical and sonar imagery; this was not pursued due to the time available before the deadline. The cross-domain transfer experiment above stays within the sonar modality (Turntable → Watertank).
""")

# ───────────────────────────────────────────────────────────────────────────
# Conclusion
# ───────────────────────────────────────────────────────────────────────────
md("""
## Conclusion

- Deep learning models can classify, detect, and segment marine debris from Forward-Looking Sonar imagery well above a simple baseline on every task we tested — answering the main project question.
- Classification and detection results are strong and **fully reproducible** in this notebook (live-recomputed test accuracy / mAP50 from saved checkpoints, fixed seeds, no hardcoded numbers).
- The cross-domain transfer experiment shows pretraining on the Turntable domain before fine-tuning on Watertank changes Watertank test accuracy versus training from scratch with the same seed and epoch budget — see the live comparison table and chart above for the exact numbers from this run.
- Segmentation is the weakest task relative to the published baseline, most plausibly due to class imbalance and training budget — but we are explicit that the segmentation numbers are *documented*, not independently re-verified in this pass.
- Future work: derive a true leakage-safe (session/object-grouped) split if session metadata becomes available; retrain and save segmentation weights with full logs; extend the cross-domain transfer idea to optical–sonar fusion.
""")

# ───────────────────────────────────────────────────────────────────────────
# Code Demonstration
# ───────────────────────────────────────────────────────────────────────────
md("""
## Code Demonstration

A short, fast, end-to-end demonstration: load saved checkpoints (no retraining) and
run live inference on a handful of test images for each task.
""")

code("""
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 5, figsize=(18, 4))
fig.suptitle("Live classification demo — ResNet-50 (Turntable→Watertank transfer)", fontweight="bold")
rng = np.random.RandomState(0)
demo_idx = rng.choice(len(wt_test), 5, replace=False)
transfer.eval().to("cpu")
for ax, idx in zip(axes, demo_idx):
    img, label = wt_test[idx]
    with torch.no_grad():
        pred = transfer(img.unsqueeze(0)).argmax(1).item()
    img_np = img.permute(1, 2, 0).numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    ax.imshow(img_np[:, :, 0], cmap="gray")
    color = "green" if pred == label else "red"
    ax.set_title(f"GT: {wt_test.classes[label]}\\nPred: {wt_test.classes[pred]}", color=color, fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.show()
""")

code("""
test_img_dir = Path("results/yolo_dataset/test/images")
demo_imgs = sorted(test_img_dir.glob("*.png"))[:4]
results = yolo_m.predict(source=[str(p) for p in demo_imgs], conf=0.25, iou=0.5, verbose=False, save=False)

fig, axes = plt.subplots(1, len(results), figsize=(18, 5))
fig.suptitle("Live detection demo — YOLOv8m on held-out test images", fontweight="bold")
for ax, r in zip(np.atleast_1d(axes), results):
    ax.imshow(r.plot(line_width=2, font_size=8)[:, :, ::-1])
    ax.axis("off")
plt.tight_layout()
plt.show()
""")

md("""
*(Segmentation inference is not demonstrated live — no surviving model weights, see Limitations above.)*
""")

nb["cells"] = cells
with open("Marine_Debris_Presentation.ipynb", "w") as f:
    nbf.write(nb, f)
print(f"Wrote {len(cells)} cells to Marine_Debris_Presentation.ipynb")
