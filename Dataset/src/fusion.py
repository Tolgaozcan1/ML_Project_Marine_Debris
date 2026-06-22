"""
Fusion strategies:
  A) Decision-level ensemble: weighted combination of optical + sonar outputs
  B) Cross-modal transfer: load optical backbone weights into sonar model
"""

import numpy as np
import torch
import torch.nn.functional as F


# ── A: Decision-level ensemble ─────────────────────────────────────────────────

def ensemble_detections(optical_dets: list, sonar_dets: list,
                        alpha: float = 0.5, iou_threshold: float = 0.5) -> list:
    """
    Merge optical bounding-box detections and sonar segmentation-derived boxes.

    Args:
        optical_dets: list of dicts {boxes, scores, labels} per image (optical model output)
        sonar_dets:   list of dicts {boxes, scores, labels} per image (sonar model output,
                      boxes derived from mask contours)
        alpha: weight for optical confidence; (1-alpha) for sonar
        iou_threshold: IoU threshold for NMS merge

    Returns:
        merged list of {boxes, scores, labels} per image
    """
    merged = []
    for opt, son in zip(optical_dets, sonar_dets):
        if len(opt["boxes"]) == 0 and len(son["boxes"]) == 0:
            merged.append({"boxes": [], "scores": [], "labels": []})
            continue

        all_boxes = opt["boxes"] + son["boxes"]
        all_scores = [s * alpha for s in opt["scores"]] + \
                     [s * (1 - alpha) for s in son["scores"]]
        all_labels = opt["labels"] + son["labels"]

        # Simple greedy NMS
        keep = _nms(np.array(all_boxes), np.array(all_scores), iou_threshold)
        merged.append({
            "boxes": [all_boxes[i] for i in keep],
            "scores": [all_scores[i] for i in keep],
            "labels": [all_labels[i] for i in keep],
        })
    return merged


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list:
    if len(boxes) == 0:
        return []
    order = scores.argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        ious = _batch_iou(boxes[i], boxes[order[1:]])
        order = order[1:][ious <= threshold]
    return keep


def _batch_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area_box + area_boxes - inter + 1e-8)


def masks_to_boxes(masks: torch.Tensor, labels: torch.Tensor,
                   scores: torch.Tensor) -> dict:
    """Convert per-class segmentation masks to bounding boxes for ensemble."""
    boxes, box_scores, box_labels = [], [], []
    for cls_idx in range(masks.shape[0]):
        m = masks[cls_idx].bool()
        if not m.any():
            continue
        rows, cols = torch.where(m)
        boxes.append([cols.min().item(), rows.min().item(),
                      cols.max().item(), rows.max().item()])
        box_scores.append(scores[cls_idx].item())
        box_labels.append(labels[cls_idx].item())
    return {"boxes": boxes, "scores": box_scores, "labels": box_labels}


# ── B: Cross-modal transfer learning ──────────────────────────────────────────

def load_optical_backbone_into_sonar_model(sonar_model: torch.nn.Module,
                                            optical_weights_path: str,
                                            strict: bool = False) -> torch.nn.Module:
    """
    Load backbone (encoder) weights from a trained optical model into
    the sonar model. Non-matching keys (heads) are skipped.

    Args:
        sonar_model: the sonar segmentation model (e.g., SegFormer or UNet)
        optical_weights_path: path to optical model state_dict (.pt)
        strict: if False, silently ignores mismatched keys

    Returns:
        sonar_model with backbone weights initialized from optical model
    """
    optical_state = torch.load(optical_weights_path, map_location="cpu")
    if "model" in optical_state:
        optical_state = optical_state["model"]

    sonar_state = sonar_model.state_dict()
    matched, skipped = 0, 0
    transfer_state = {}

    for k, v in optical_state.items():
        if k in sonar_state and sonar_state[k].shape == v.shape:
            transfer_state[k] = v
            matched += 1
        else:
            skipped += 1

    sonar_state.update(transfer_state)
    sonar_model.load_state_dict(sonar_state, strict=strict)
    print(f"Transfer: {matched} layers loaded, {skipped} skipped (shape mismatch or absent)")
    return sonar_model


def get_differential_lr_optimizer(model: torch.nn.Module,
                                   backbone_lr: float = 1e-4,
                                   head_lr: float = 1e-3,
                                   backbone_prefix: str = "backbone") -> torch.optim.Optimizer:
    """
    Lower LR for pretrained backbone, higher LR for freshly initialized head.
    Typical for transfer learning from optical → sonar.
    """
    backbone_params = [p for n, p in model.named_parameters() if backbone_prefix in n]
    head_params = [p for n, p in model.named_parameters() if backbone_prefix not in n]

    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": head_lr},
        ],
        weight_decay=1e-4,
    )
