"""
Dataset download helper for the Marine Debris project.

Run this script AFTER accepting the dataset licenses on the respective sites.
It will guide you through placing files in the correct directories.

Usage:
    python download_datasets.py --check     # check what's already present
    python download_datasets.py --guide     # print download instructions
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"


def check_status():
    print("=" * 60)
    print("Dataset status check")
    print("=" * 60)

    # TrashCan
    trashcan = DATA_DIR / "trashcan"
    tc_ann   = trashcan / "annotations"
    tc_imgs  = trashcan / "images"

    splits_found = []
    for split in ["train", "val", "test"]:
        ann_ok = (tc_ann / f"instances_{split}.json").exists()
        img_ok = (tc_imgs / split).exists() and any((tc_imgs / split).iterdir())
        if ann_ok and img_ok:
            n_imgs = len(list((tc_imgs / split).glob("*")))
            splits_found.append(split)
            print(f"  TrashCan [{split}]: OK  ({n_imgs} images)")
        else:
            print(f"  TrashCan [{split}]: MISSING  (need annotations + images/{split}/)")

    # FLS sonar
    fls_dir = DATA_DIR / "fls"
    hdf5_files = list(fls_dir.glob("*.hdf5")) + list(fls_dir.glob("*.h5"))
    if hdf5_files:
        for f in hdf5_files:
            size_mb = f.stat().st_size / 1024**2
            print(f"  FLS sonar: OK  ({f.name}, {size_mb:.0f} MB)")
    else:
        print("  FLS sonar: MISSING  (.hdf5 file not found in data/fls/)")

    print()
    if len(splits_found) == 3 and hdf5_files:
        print("All datasets present. You can run the notebooks.")
    else:
        print("Some datasets are missing. Run:  python download_datasets.py --guide")


def print_guide():
    print("=" * 60)
    print("DOWNLOAD GUIDE")
    print("=" * 60)
    print()
    print("─── Dataset 1: TrashCan 1.0 (optical) ───────────────────")
    print()
    print("1. Visit: https://conservancy.umn.edu/handle/11299/214366")
    print("   (or search 'TrashCan 1.0 underwater' on Google)")
    print()
    print("2. Download the dataset ZIP/TAR (contains COCO JSON annotations")
    print("   and images).")
    print()
    print("3. Extract and place files so your structure looks like:")
    print(f"   {DATA_DIR}/trashcan/")
    print(f"   {DATA_DIR}/trashcan/annotations/instances_train.json")
    print(f"   {DATA_DIR}/trashcan/annotations/instances_val.json")
    print(f"   {DATA_DIR}/trashcan/annotations/instances_test.json")
    print(f"   {DATA_DIR}/trashcan/images/train/  (images here)")
    print(f"   {DATA_DIR}/trashcan/images/val/")
    print(f"   {DATA_DIR}/trashcan/images/test/")
    print()
    print("─── Dataset 2: FLS Sonar (watertank segmentation) ────────")
    print()
    print("1. Visit: https://zenodo.org/doi/10.5281/zenodo.15101686")
    print()
    print("2. Download ONLY the watertank-segmentation file")
    print("   (do NOT download the full 1.3 TB release — just the segmentation subset).")
    print()
    print("3. Place the .hdf5 file here:")
    print(f"   {DATA_DIR}/fls/watertank_segmentation.hdf5")
    print()
    print("4. After placing the file, inspect its internal structure:")
    print("   python -c \"import h5py; f=h5py.File('data/fls/watertank_segmentation.hdf5','r'); print(list(f.keys()))\"")
    print()
    print("   Then update the HDF5 key paths in 03_sonar.ipynb if needed.")
    print()
    print("─── After downloading ────────────────────────────────────")
    print()
    print("  python download_datasets.py --check")
    print("  conda activate marine-debris")
    print("  jupyter notebook notebooks/01_eda.ipynb")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check",  action="store_true", help="Check dataset status")
    parser.add_argument("--guide",  action="store_true", help="Print download instructions")
    args = parser.parse_args()

    if args.check:
        check_status()
    elif args.guide:
        print_guide()
    else:
        # Default: show both
        check_status()
        print()
        print_guide()
