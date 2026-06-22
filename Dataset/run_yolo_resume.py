"""
Wrapper that patches the ultralytics MPS bincount bug before resuming YOLO training.

Bug: unique(return_counts=True) on MPS returns garbage values → counts.max() is
     astronomically large → torch.zeros tries to allocate 320 GiB and crashes.
Fix: compute counts.max() on CPU (correct value) then use it for the allocation.
"""
import torch

# ── Patch before importing ultralytics ──────────────────────────────────────
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
    max_count = int(counts.cpu().max().item())   # CPU gives correct value
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

# ── Resume YOLO training ─────────────────────────────────────────────────────
from ultralytics import YOLO

model = YOLO("results/yolo_sonar/yolov8m_watertank/weights/last.pt")
model.train(resume=True)
