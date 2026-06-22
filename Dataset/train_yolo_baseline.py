"""
Detection baseline: YOLOv8n (nano) trained on the same sonar dataset and dataset.yaml
as the main YOLOv8m model, for far fewer epochs — the "simple but correct baseline"
the course instructions require before the main model.

Applies the same MPS bincount monkey-patch as run_yolo_resume.py (otherwise
training crashes on Apple Silicon — see CLAUDE.md "Known Issues & Fixes").

Run from Dataset/:
  conda activate marine-debris && cd ~/Desktop/ML-Project\ /Dataset
  python train_yolo_baseline.py
"""
import torch

# ── Patch before importing ultralytics (MPS bincount bug, see run_yolo_resume.py) ──
import ultralytics.utils.loss as _loss_mod

_orig_preprocess = _loss_mod.v8DetectionLoss.preprocess


def _patched_preprocess(self, targets, batch_size, scale_tensor):
    from ultralytics.utils.ops import xywh2xyxy
    nl, ne = targets.shape
    if nl == 0:
        return torch.zeros(batch_size, 0, ne - 1, device=self.device)
    batch_idx = targets[:, 0].long()
    _, counts = batch_idx.unique(return_counts=True)
    counts = counts.to(dtype=torch.int32)
    max_count = int(counts.cpu().max().item())
    out = torch.zeros(batch_size, max_count, ne - 1, device=self.device)
    offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=self.device)
    offsets.scatter_add_(0, batch_idx + 1, torch.ones_like(batch_idx))
    offsets = offsets.cumsum(0)
    within_idx = torch.arange(nl, device=self.device) - offsets[batch_idx]
    out[batch_idx, within_idx] = targets[:, 1:]
    out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
    return out


_loss_mod.v8DetectionLoss.preprocess = _patched_preprocess
print("[mps_fix] Patched v8DetectionLoss.preprocess — counts.max() now runs on CPU")

# ── Train YOLOv8n baseline ───────────────────────────────────────────────────
from ultralytics import YOLO

DATASET_YAML = "results/yolo_dataset/dataset.yaml"

model = YOLO("yolov8n.pt")  # downloads COCO-pretrained nano weights on first run
results = model.train(
    data=DATASET_YAML,
    epochs=30,
    imgsz=640,
    batch=8,
    lr0=0.001,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=5,
    seed=42,
    project="results/yolo_sonar",
    name="yolov8n_baseline",
    flipud=0.3,
    fliplr=0.5,
    degrees=15.0,
    scale=0.3,
    mosaic=0.5,
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.1,
)

best_weights = results.save_dir / "weights" / "best.pt"
model_eval = YOLO(str(best_weights))
metrics = model_eval.val(data=DATASET_YAML, split="test", imgsz=640, batch=8)

print("\n=== YOLOv8n baseline — test set ===")
print(f"mAP50:    {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"Precision:{metrics.box.mp:.4f}")
print(f"Recall:   {metrics.box.mr:.4f}")

with open("results/yolo_sonar/yolov8n_baseline_test_metrics.txt", "w") as f:
    f.write(f"mAP50: {metrics.box.map50:.4f}\n")
    f.write(f"mAP50-95: {metrics.box.map:.4f}\n")
    f.write(f"Precision: {metrics.box.mp:.4f}\n")
    f.write(f"Recall: {metrics.box.mr:.4f}\n")
