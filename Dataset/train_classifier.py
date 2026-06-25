"""
Classification training: baseline CNN, ResNet-50 from scratch, and ResNet-50
cross-domain transfer (Turntable-Cropped pretrain -> Watertank-Cropped finetune).

Regenerates the classification training code (the original script that produced
results/cls_watertank_best.pt / cls_turntable_best.pt was lost) and adds:
  1. a simple baseline CNN (required by the course instructions: every task
     needs a justified baseline before the main model),
  2. a fresh, reproducible ResNet-50-from-scratch run on Watertank-Cropped,
  3. the required Task C cross-domain comparison: ResNet-50 pretrained on
     Turntable-Cropped then fine-tuned on Watertank-Cropped, vs scratch.

Run from Dataset/:
  conda activate marine-debris && cd ~/Desktop/ML-Project\ /Dataset
  python train_classifier.py
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm
from torch.utils.data import DataLoader

from src.datasets import FLSClassificationDataset
from src.utils import get_device

SEED = 42
DATA_ROOT = Path("marine-debris-fls-datasets/md_fls_dataset/data")
WATERTANK = str(DATA_ROOT / "watertank-cropped")
TURNTABLE = str(DATA_ROOT / "turntable-cropped")
RESULTS_DIR = Path("results/classification")
FIGURES_DIR = Path("results/figures")


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SimpleCNN(nn.Module):
    """Small from-scratch CNN baseline: 4 conv blocks + global pool + linear head."""

    def __init__(self, num_classes: int):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
        self.features = nn.Sequential(
            block(3, 32), block(32, 64), block(64, 128), block(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def make_resnet50(num_classes: int) -> nn.Module:
    model = tvm.resnet50(weights=None)
    model.fc = nn.Linear(2048, num_classes)
    return model


def run_epoch(model, loader, optimizer, criterion, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += imgs.size(0)
    return total_loss / total, correct / total


def train_model(model, train_ds, val_ds, test_ds, device, epochs: int, lr: float,
                 tag: str, batch_size: int = 32):
    sampler = train_ds.make_weighted_sampler()
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    history = []
    best_val_acc = 0.0
    best_state = None
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, optimizer, criterion, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, optimizer, criterion, device, train=False)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                         "val_loss": val_loss, "val_acc": val_acc})
        print(f"[{tag}] epoch {epoch}/{epochs} | train_loss {train_loss:.4f} acc {train_acc:.4f} "
              f"| val_loss {val_loss:.4f} acc {val_acc:.4f}")
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    elapsed = time.time() - t0

    model.load_state_dict(best_state)
    test_loss, test_acc = run_epoch(model, test_loader, optimizer, criterion, device, train=False)
    print(f"[{tag}] FINAL test_acc={test_acc:.4f}  (best_val_acc={best_val_acc:.4f}, {elapsed/60:.1f} min)")

    return model, {
        "tag": tag, "epochs": epochs, "lr": lr, "best_val_acc": best_val_acc,
        "test_acc": test_acc, "train_minutes": elapsed / 60, "history": history,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs_baseline", type=int, default=15)
    parser.add_argument("--epochs_scratch", type=int, default=15)
    parser.add_argument("--epochs_pretrain", type=int, default=10)
    parser.add_argument("--epochs_transfer", type=int, default=10)
    parser.add_argument("--stage", choices=["baseline", "scratch", "pretrain", "transfer", "all"],
                         default="all", help="Run a single stage (resume-friendly) or all stages")
    parser.add_argument("--leakage_safe", action="store_true",
                         help="Use the approximate object/session-grouped block split "
                              "(src.datasets split_strategy='blocked') instead of the random "
                              "image-level split, to bound the leakage risk in the random split. "
                              "Writes to separate checkpoint files and run-log keys so the "
                              "random-split numbers are preserved for side-by-side comparison.")
    args = parser.parse_args()

    set_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    split_strategy = "blocked" if args.leakage_safe else "random"
    suffix = "_blocked" if args.leakage_safe else ""

    wt_train = FLSClassificationDataset(WATERTANK, split="train", seed=SEED, split_strategy=split_strategy)
    wt_val = FLSClassificationDataset(WATERTANK, split="val", seed=SEED, split_strategy=split_strategy)
    wt_test = FLSClassificationDataset(WATERTANK, split="test", seed=SEED, split_strategy=split_strategy)
    n_watertank_cls = len(wt_train.classes)
    print(f"Watertank-Cropped ({split_strategy} split): {n_watertank_cls} classes | "
          f"train={len(wt_train)} val={len(wt_val)} test={len(wt_test)}", flush=True)

    runs_path = RESULTS_DIR / "classification_runs.json"
    runs = json.load(open(runs_path)) if runs_path.exists() else {}
    stage = args.stage

    # 1. Baseline: simple CNN from scratch on Watertank-Cropped
    if stage in ("baseline", "all"):
        set_seed(SEED)
        baseline = SimpleCNN(n_watertank_cls)
        baseline, info = train_model(baseline, wt_train, wt_val, wt_test, device,
                                      epochs=args.epochs_baseline, lr=1e-3, tag=f"baseline_cnn{suffix}")
        torch.save(baseline.state_dict(), RESULTS_DIR / f"cls_baseline_cnn{suffix}.pt")
        runs[f"baseline_cnn{suffix}"] = info
        json.dump(runs, open(runs_path, "w"), indent=2)

    # 2. Main model, from scratch: ResNet-50 on Watertank-Cropped
    if stage in ("scratch", "all"):
        set_seed(SEED)
        scratch = make_resnet50(n_watertank_cls)
        scratch, info = train_model(scratch, wt_train, wt_val, wt_test, device,
                                     epochs=args.epochs_scratch, lr=1e-4, tag=f"resnet50_scratch{suffix}")
        torch.save(scratch.state_dict(), RESULTS_DIR / f"cls_watertank_resnet50_scratch{suffix}.pt")
        runs[f"resnet50_scratch{suffix}"] = info
        json.dump(runs, open(runs_path, "w"), indent=2)

    # 3. Task C pt.1 — pretrain ResNet-50 on Turntable-Cropped
    if stage in ("pretrain", "all"):
        tt_train = FLSClassificationDataset(TURNTABLE, split="train", seed=SEED, split_strategy=split_strategy)
        tt_val = FLSClassificationDataset(TURNTABLE, split="val", seed=SEED, split_strategy=split_strategy)
        tt_test = FLSClassificationDataset(TURNTABLE, split="test", seed=SEED, split_strategy=split_strategy)
        n_turntable_cls = len(tt_train.classes)
        print(f"Turntable-Cropped ({split_strategy} split): {n_turntable_cls} classes | "
              f"train={len(tt_train)} val={len(tt_val)} test={len(tt_test)}", flush=True)

        set_seed(SEED)
        pretrain_model = make_resnet50(n_turntable_cls)
        pretrain_model, pretrain_info = train_model(
            pretrain_model, tt_train, tt_val, tt_test, device,
            epochs=args.epochs_pretrain, lr=1e-4, tag=f"resnet50_turntable_pretrain{suffix}")
        torch.save(pretrain_model.state_dict(), RESULTS_DIR / f"cls_turntable_resnet50_pretrain{suffix}.pt")
        runs[f"resnet50_turntable_pretrain{suffix}"] = pretrain_info
        json.dump(runs, open(runs_path, "w"), indent=2)

    # 3. Task C pt.2 — transfer pretrained backbone, finetune on Watertank-Cropped
    if stage in ("transfer", "all"):
        n_turntable_cls = 18  # fixed by dataset (see Dataset section)
        pretrain_model = make_resnet50(n_turntable_cls)
        pretrain_model.load_state_dict(
            torch.load(RESULTS_DIR / f"cls_turntable_resnet50_pretrain{suffix}.pt", map_location="cpu"))

        set_seed(SEED)
        transfer_model = make_resnet50(n_watertank_cls)
        # Copy every layer except the final classification head (class counts differ: 18 vs 10)
        pretrain_state = pretrain_model.state_dict()
        transfer_state = transfer_model.state_dict()
        backbone_state = {k: v for k, v in pretrain_state.items() if not k.startswith("fc.")}
        transfer_state.update(backbone_state)
        transfer_model.load_state_dict(transfer_state)

        transfer_model, transfer_info = train_model(
            transfer_model, wt_train, wt_val, wt_test, device,
            epochs=args.epochs_transfer, lr=3e-5, tag=f"resnet50_transfer{suffix}")
        torch.save(transfer_model.state_dict(), RESULTS_DIR / f"cls_watertank_resnet50_transfer{suffix}.pt")
        runs[f"resnet50_transfer{suffix}"] = transfer_info
        json.dump(runs, open(runs_path, "w"), indent=2)

    if args.leakage_safe:
        print("\nLeakage-safe (blocked-split) run complete. Skipping the random-split "
              "comparison chart — see classification_runs.json for the new "
              f"*{suffix} entries.")
        return

    if not all(k in runs for k in ("baseline_cnn", "resnet50_scratch", "resnet50_transfer")):
        print("\nNot all stages complete yet — skipping comparison chart. "
              "Re-run with --stage for the remaining stage(s).")
        return

    # Comparison chart: baseline vs scratch vs transfer (Watertank test accuracy)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["Baseline CNN\n(scratch)", "ResNet-50\n(scratch)", "ResNet-50\n(Turntable→Watertank transfer)"]
    accs = [runs["baseline_cnn"]["test_acc"], runs["resnet50_scratch"]["test_acc"],
            runs["resnet50_transfer"]["test_acc"]]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, [a * 100 for a in accs], color=["#90A4AE", "#42A5F5", "#66BB6A"])
    ax.bar_label(bars, fmt="%.2f%%")
    ax.set_ylabel("Watertank-Cropped Test Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Classification: Baseline vs. Scratch vs. Cross-Domain Transfer")
    plt.tight_layout()
    fig_path = FIGURES_DIR / "classification_baseline_vs_transfer.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"\nSaved comparison chart: {fig_path}")
    print(f"Saved run log: {RESULTS_DIR / 'classification_runs.json'}")
    print("\nSummary:")
    for name, info in runs.items():
        print(f"  {name:35s} test_acc={info['test_acc']*100:.2f}%  ({info['train_minutes']:.1f} min)")


if __name__ == "__main__":
    main()
