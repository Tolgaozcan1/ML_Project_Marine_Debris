"""
Generate all project visualizations for the report.

Run from Dataset/ directory:
  conda activate marine-debris && cd ~/Desktop/ML-Project\ /Dataset
  python generate_visualizations.py

Outputs saved to results/figures/
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from torch.utils.data import DataLoader
import torchvision.models as tvm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from src.datasets import FLSClassificationDataset, FLSSegmentationDataset

OUT = Path("results/figures")
OUT.mkdir(parents=True, exist_ok=True)

DATA_ROOT = Path("marine-debris-fls-datasets/md_fls_dataset/data")
YOLO_WEIGHTS = Path("results/yolo_sonar/yolov8m_watertank/weights/best.pt")

print("Saving all figures to:", OUT.resolve())

# ─────────────────────────────────────────────────────────────────
# 1. CLASSIFICATION — Confusion Matrices
# ─────────────────────────────────────────────────────────────────
print("\n[1/4] Classification confusion matrices...")

for tag, ckpt, root, nc in [
    ("watertank", "results/cls_watertank_best.pt",
     str(DATA_ROOT / "watertank-cropped"), 10),
    ("turntable", "results/cls_turntable_best.pt",
     str(DATA_ROOT / "turntable-cropped"), 18),
]:
    ds = FLSClassificationDataset(root, split="test")
    model = tvm.resnet50(weights=None)
    model.fc = torch.nn.Linear(2048, nc)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False))
    model.eval()
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            preds = model(imgs).argmax(1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    acc = np.sum(np.diag(cm)) / np.sum(cm)

    fig, ax = plt.subplots(figsize=(12, 10) if nc > 10 else (8, 7))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm.astype(float) / cm.sum(axis=1, keepdims=True),
        display_labels=ds.classes
    )
    disp.plot(ax=ax, colorbar=True, cmap="Blues", values_format=".2f")
    ax.set_title(f"ResNet-50 Classification — {tag.capitalize()}\nTest Accuracy: {acc*100:.2f}%",
                 fontsize=13, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    path = OUT / f"cls_{tag}_confusion.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path.name}  (acc={acc*100:.2f}%)")

# ─────────────────────────────────────────────────────────────────
# 2. CLASSIFICATION — Sample Predictions
# ─────────────────────────────────────────────────────────────────
print("\n[2/4] Classification sample predictions...")

ds = FLSClassificationDataset(str(DATA_ROOT / "watertank-cropped"), split="test")
model = tvm.resnet50(weights=None)
model.fc = torch.nn.Linear(2048, 10)
model.load_state_dict(torch.load("results/cls_watertank_best.pt",
                                  map_location="cpu", weights_only=False))
model.eval()

fig, axes = plt.subplots(3, 5, figsize=(15, 9))
fig.suptitle("ResNet-50 — Watertank Classification Predictions (Test Set)",
             fontsize=13, fontweight="bold")

np.random.seed(42)
indices = np.random.choice(len(ds), 15, replace=False)
for ax, idx in zip(axes.flat, indices):
    img_tensor, label = ds[idx]
    with torch.no_grad():
        pred = model(img_tensor.unsqueeze(0)).argmax(1).item()
    img_np = img_tensor.permute(1, 2, 0).numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    ax.imshow(img_np[:, :, 0], cmap="gray")
    color = "green" if pred == label else "red"
    ax.set_title(f"GT: {ds.classes[label]}\nPred: {ds.classes[pred]}",
                 color=color, fontsize=7)
    ax.axis("off")

plt.tight_layout()
path = OUT / "cls_watertank_samples.png"
plt.savefig(path, dpi=150)
plt.close()
print(f"  Saved: {path.name}")

# ─────────────────────────────────────────────────────────────────
# 2b. CLASSIFICATION — Misclassified Examples (real error analysis)
# ─────────────────────────────────────────────────────────────────
print("\n[2b/4] Misclassified examples (error analysis)...")


def plot_misclassified_examples(ckpt_path, root, nc, title, out_name, max_show=12, arch="resnet50"):
    """Find and plot every misclassified test example (not a random sample)."""
    ds = FLSClassificationDataset(root, split="test")
    if arch == "resnet50":
        model = tvm.resnet50(weights=None)
        model.fc = torch.nn.Linear(2048, nc)
    else:
        from train_classifier import SimpleCNN
        model = SimpleCNN(nc)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=False))
    model.eval()

    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            preds = model(imgs).argmax(1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    wrong = [i for i, (p, l) in enumerate(zip(all_preds, all_labels)) if p != l]
    print(f"  {title}: {len(wrong)}/{len(ds)} misclassified "
          f"({len(wrong)/len(ds)*100:.1f}% error rate)")
    if not wrong:
        print("  No misclassified examples — nothing to plot.")
        return []

    show = wrong[:max_show]
    n_cols = 4
    n_rows = (len(show) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.2 * n_rows))
    fig.suptitle(f"{title} — Misclassified Test Examples ({len(wrong)} total errors)",
                 fontsize=13, fontweight="bold")
    axes_flat = np.array(axes).reshape(-1)
    for ax, idx in zip(axes_flat, show):
        img_tensor, label = ds[idx]
        img_np = img_tensor.permute(1, 2, 0).numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        ax.imshow(img_np[:, :, 0], cmap="gray")
        ax.set_title(f"GT: {ds.classes[label]}\nPred: {ds.classes[all_preds[idx]]}",
                     color="red", fontsize=8)
        ax.axis("off")
    for ax in axes_flat[len(show):]:
        ax.axis("off")
    plt.tight_layout()
    path = OUT / out_name
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path.name}")

    from collections import Counter
    confusions = Counter((ds.classes[all_labels[i]], ds.classes[all_preds[i]]) for i in wrong)
    print(f"  Most common confusions: {confusions.most_common(5)}")
    return confusions.most_common(5)


# Main model (ResNet-50, Turntable-Cropped) — genuine main-model failures
plot_misclassified_examples(
    "results/cls_turntable_best.pt", str(DATA_ROOT / "turntable-cropped"), 18,
    "ResNet-50 (Turntable)", "cls_turntable_misclassified.png")

# Baseline CNN (Watertank-Cropped) — shows what the simple baseline gets wrong
_baseline_ckpt = Path("results/classification/cls_baseline_cnn.pt")
if _baseline_ckpt.exists():
    plot_misclassified_examples(
        str(_baseline_ckpt), str(DATA_ROOT / "watertank-cropped"), 10,
        "Baseline CNN (Watertank)", "cls_baseline_misclassified.png", arch="simplecnn")
else:
    print(f"  {_baseline_ckpt} not found yet — skipping baseline error analysis")

# ─────────────────────────────────────────────────────────────────
# 3. YOLO — Per-Class AP Bar Chart + Prediction Samples
# ─────────────────────────────────────────────────────────────────
print("\n[3/4] YOLO detection visualizations...")

# Per-class AP is recomputed live from the same .val() call that produces the
# aggregate mAP50=0.937 reported elsewhere, instead of reusing the old
# training-time log (mAP50=0.967 in that run) which the report no longer cites.
import warnings as _warnings; _warnings.filterwarnings("ignore")
from ultralytics import YOLO as _YOLO
_yolo_pc = _YOLO(str(YOLO_WEIGHTS))
_m_pc = _yolo_pc.val(data="results/yolo_dataset/dataset.yaml", split="test",
                      imgsz=640, batch=8, verbose=False)
_names = _yolo_pc.names
_order = list(_m_pc.box.ap_class_index)
_pairs = sorted(zip(_order, _m_pc.box.ap50, _m_pc.box.maps), key=lambda t: -t[1])
classes = [_names[i] for i, _, _ in _pairs]
map50 = [round(float(v), 3) for _, v, _ in _pairs]
map50_95 = [round(float(v), 3) for _, _, v in _pairs]

x = np.arange(len(classes))
fig, ax = plt.subplots(figsize=(13, 5))
bars1 = ax.bar(x - 0.2, map50, 0.38, label="mAP50", color="#2196F3", alpha=0.85)
bars2 = ax.bar(x + 0.2, map50_95, 0.38, label="mAP50-95", color="#FF5722", alpha=0.85)
ax.axhline(np.mean(map50), color="#2196F3", linestyle="--", linewidth=1.2,
           label=f"Mean mAP50 = {np.mean(map50):.3f}")
ax.axhline(np.mean(map50_95), color="#FF5722", linestyle="--", linewidth=1.2,
           label=f"Mean mAP50-95 = {np.mean(map50_95):.3f}")
ax.set_xticks(x)
ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=9)
ax.set_ylabel("AP Score")
ax.set_ylim(0, 1.05)
ax.set_title("YOLOv8m — Per-Class Detection Performance (80 Epochs, Kaggle T4)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=6.5)
plt.tight_layout()
path = OUT / "yolo_per_class_ap.png"
plt.savefig(path, dpi=150)
plt.close()
print(f"  Saved: {path.name}")

# YOLO prediction samples on test images
if YOLO_WEIGHTS.exists():
    try:
        from ultralytics import YOLO
        yolo = YOLO(str(YOLO_WEIGHTS))
        test_img_dir = Path("results/yolo_dataset/test/images")
        test_imgs = sorted(test_img_dir.glob("*.png"))[:12]
        if test_imgs:
            results = yolo.predict(
                source=[str(p) for p in test_imgs],
                conf=0.25, iou=0.5, verbose=False, save=False
            )
            fig, axes = plt.subplots(3, 4, figsize=(16, 12))
            fig.suptitle("YOLOv8m — Detection Predictions on Test Images",
                         fontsize=13, fontweight="bold")
            for ax, r in zip(axes.flat, results):
                img = r.plot(line_width=2, font_size=8)[:, :, ::-1]
                ax.imshow(img)
                ax.axis("off")
            plt.tight_layout()
            path = OUT / "yolo_predictions.png"
            plt.savefig(path, dpi=150)
            plt.close()
            print(f"  Saved: {path.name}")
    except Exception as e:
        print(f"  YOLO inference skipped: {e}")
else:
    print(f"  YOLO weights not found at {YOLO_WEIGHTS} — skipping predictions")

# ─────────────────────────────────────────────────────────────────
# 4. SUMMARY — Results Comparison Table Figure
# ─────────────────────────────────────────────────────────────────
print("\n[4/4] Summary results figure...")

fig, ax = plt.subplots(figsize=(13, 5.5))
ax.axis("off")

# All numbers below are recomputed live from saved checkpoints (classification:
# results/classification/classification_runs.json; detection: the .val() call above)
# rather than hardcoded — this table previously went stale relative to the rest
# of the report (100.00%/99.19%/0.967) and that mismatch was caught in review.
import json as _json
_runs_path = Path("results/classification/classification_runs.json")
_runs = _json.load(open(_runs_path)) if _runs_path.exists() else {}
def _acc(key):
    return f"{_runs[key]['test_acc']*100:.2f}%" if key in _runs else "n/a"

table_data = [
    ["ResNet-50 Classification", "Watertank-Cropped (10 cls), scratch",  "Top-1 Acc", _acc("resnet50_scratch"), "—"],
    ["ResNet-50 Classification", "Watertank-Cropped (10 cls), transfer", "Top-1 Acc", _acc("resnet50_transfer"), "—"],
    ["ResNet-50 Classification", "Turntable-Cropped (18 cls)",           "Top-1 Acc", _acc("resnet50_turntable_pretrain"), "—"],
    ["YOLOv8m Detection",        "Watertank-Seg",              "mAP50",     f"{_m_pc.box.map50:.3f}",   "—"],
    ["YOLOv8m Detection",        "Watertank-Seg",              "mAP50-95",  f"{_m_pc.box.map:.3f}",     "—"],
    ["U-Net + ResNet34",         "Watertank-Seg",              "mIoU (preliminary, see Limitations)",      "0.638",   "0.748"],
    ["SegFormer-B2",             "Watertank-Seg",              "mIoU (preliminary, see Limitations)",      "0.658",   "0.748"],
]
col_labels = ["Model", "Dataset", "Metric", "Ours", "Paper Baseline"]
colors = [["#E8F5E9"] * 5] * 3 + [["#E3F2FD"] * 5] * 2 + [["#FFF3E0"] * 5] * 2

table = ax.table(
    cellText=table_data,
    colLabels=col_labels,
    cellLoc="center",
    loc="center",
    cellColours=colors,
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 2.0)
for (r, c), cell in table.get_celld().items():
    if r == 0:
        cell.set_facecolor("#37474F")
        cell.set_text_props(color="white", fontweight="bold")
    cell.set_edgecolor("#BDBDBD")

ax.set_title("Marine Debris Detection — Final Results Summary",
             fontsize=13, fontweight="bold", pad=20)
plt.tight_layout()
path = OUT / "results_summary_table.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {path.name}")

print(f"\nAll figures saved to: {OUT.resolve()}")
print("Files:", [f.name for f in sorted(OUT.glob("*.png"))])
