# 🌊 Marine Debris Detection in Sonar and Optical Imagery

## 🎯 End Goal

> **Build a machine learning system that can automatically detect, localize, and classify marine debris objects from sonar (and optionally optical) underwater imagery.**

---

### In practical terms, your system should be able to:

1. **Take an underwater sonar image as input**
2. **Identify whether debris is present**
3. **Locate where it is** in the image (bounding box or pixel-level mask)
4. **Classify what type of debris it is** (bottle, tire, pipe, hook, etc.)

---

### 🌍 Real-World Impact

The ultimate application is to equip **Autonomous Underwater Vehicles (AUVs)** with this detection capability so they can:
- 🤖 Autonomously scan the ocean floor
- 🗑️ Identify and map marine debris
- 🌊 Aid in **underwater cleanup missions** without human divers
- 📊 Monitor pollution levels in water bodies over time

---

### 📚 From a University Project Perspective

Your deliverable will likely be:
- A **trained ML model** (segmentation/detection) evaluated on the dataset
- A **comparison of approaches** (e.g., baseline CNN vs. U-Net vs. Attention U-Net)
- A **report/presentation** showing your methodology, results (mIoU, accuracy, etc.), and conclusions
- Ideally, **beating or matching the published baseline** of ~0.748 mIoU with U-Net + ResNet34

---

## 1. Problem Statement

Marine debris (plastic bottles, tires, hooks, pipes, etc.) pollutes underwater environments and poses serious ecological and navigational hazards. The goal of this project is to **automatically detect and classify marine debris** from two types of imagery:

- **Sonar imagery** — specifically Forward-Looking Sonar (FLS) images (grayscale, acoustic)
- **Optical imagery** — standard camera-based underwater images

This is a **computer vision + deep learning** problem involving tasks like classification, object detection, and/or semantic segmentation.

---

## 2. The Dataset

### 📦 FLS Dataset (Primary — from GitHub & Zenodo)

Captured using an **ARIS Explorer 3000 FLS sensor** at the Ocean Systems Lab Water Tank (Heriot-Watt University).

The dataset has **three scenarios**:

| Scenario | Description |
|---|---|
| **Watertank** | AUV-mounted sonar, objects placed on tank floor |
| **Turntable** | Sonar fixed, object rotates 360° for full yaw capture |
| **Flooded Quarry** | Real-world outdoor setting for diversity |

**~2000+ FLS images**, **10 object classes + background** (bottles, tires, hooks, valves, pipes, propellers, etc.)

**Supported ML tasks** from this dataset:
- Image Classification
- Object Detection
- Semantic Segmentation
- Patch Matching
- Unsupervised Feature Learning

---

## 3. Recommended Solution Approach

Based on published research, here's what works and what your team should consider:

### 🥇 Primary Recommendation: Semantic Segmentation

> Best performing baseline: **U-Net with ResNet34 backbone → 0.7481 mIoU**

Segmentation tells you **what** the debris is *and* **where** it is — the most informative output.

### 🔁 Pipeline Overview

```
Raw FLS/Optical Images
        ↓
  Preprocessing (resize, normalize, augmentation)
        ↓
  Deep Learning Model (U-Net / YOLO / CNN)
        ↓
  Output: Bounding Box / Segmentation Mask / Class Label
        ↓
  Evaluation (mIoU / mAP / Accuracy)
```

### Model Options by Task

| Task | Recommended Model | Notes |
|---|---|---|
| Classification | CNN, ResNet, EfficientNet | Simplest starting point |
| Object Detection | YOLOv8, Faster R-CNN | Good for localization |
| Segmentation | **U-Net + ResNet34** | State-of-the-art on this dataset |
| Attention-based | Attention U-Net | Best performance reported |

> 💡 Since images are **grayscale sonar** images, **pretrained ImageNet weights won't directly apply** — you'll train from scratch or use grayscale-adapted backbones.

---

## 4. Team of 3 — Work Division & Step-by-Step Plan

### 👥 Suggested Role Split

| Member | Role | Responsibilities |
|---|---|---|
| **Member 1** | Data & EDA Lead | Dataset download, exploration, preprocessing, augmentation |
| **Member 2** | Modeling Lead | Model architecture, training, hyperparameter tuning |
| **Member 3** | Evaluation & Reporting Lead | Metrics, visualization, paper writing, comparisons |

---

### 📅 Phase-by-Phase Plan

#### **Phase 1 — Understanding & Setup (Week 1)**
- [ ] Read the dataset paper: arXiv 2503.22880 and arXiv 2108.06800
- [ ] Download the dataset from GitHub and Zenodo
- [ ] Set up shared environment: Python, PyTorch or TensorFlow, Jupyter/Colab/Kaggle
- [ ] Explore dataset structure — image sizes, class distribution, annotation format

#### **Phase 2 — EDA & Preprocessing (Week 1–2)**
- [ ] Visualize sample sonar images and segmentation masks
- [ ] Analyze class imbalance (background vs. debris)
- [ ] Implement preprocessing: resize to 256×256, normalize pixel values
- [ ] Apply data augmentation: flips, rotations, brightness jitter
- [ ] Create train/val/test splits

#### **Phase 3 — Baseline Model (Week 2–3)**
- [ ] Start with **image classification** as a warm-up (simpler, faster feedback)
- [ ] Then implement **U-Net** for segmentation as the main task
- [ ] Use ResNet34 as the encoder backbone
- [ ] Train on watertank scenario first

#### **Phase 4 — Improve & Experiment (Week 3–4)**
- [ ] Try Attention U-Net or transformer-based segmentation (SegFormer)
- [ ] Experiment with loss functions: BCE, Dice Loss, Focal Loss (handles class imbalance)
- [ ] Try cross-scenario generalization (train on watertank, test on quarry)
- [ ] Optionally: add optical imagery and compare/fuse results

#### **Phase 5 — Evaluation & Report (Week 4–5)**
- [ ] Evaluate using **mIoU** (primary metric for segmentation), precision, recall, F1
- [ ] Visualize predictions vs. ground truth masks
- [ ] Compare your results to baseline results from the paper
- [ ] Write the report: problem → data → method → results → conclusion

---

## 5. Tech Stack Recommendations

```python
# Core
Python 3.10+
PyTorch + torchvision  # or TensorFlow/Keras

# Segmentation
segmentation_models_pytorch  # has U-Net + ResNet34 ready

# Data handling
numpy, pandas, OpenCV, PIL, albumentations  # augmentation

# Visualization
matplotlib, seaborn

# Experiment tracking
wandb or TensorBoard
```

---

## 6. Key Papers to Read

1. 📄 [The Marine Debris FLS Datasets (2025)](https://arxiv.org/abs/2503.22880) — most complete dataset paper
2. 📄 [FLS Semantic Segmentation (2021)](https://arxiv.org/abs/2108.06800) — U-Net baseline
3. 📄 [Deep NNs for Marine Debris in Sonar (2019)](https://arxiv.org/abs/1905.05241) — foundational work

---

## 7. Quick Summary

| Aspect | Detail |
|---|---|
| **Input** | Grayscale FLS sonar images (+ optical if available) |
| **Output** | Class label / bounding box / segmentation mask |
| **Best Task to Target** | Semantic Segmentation |
| **Best Known Model** | U-Net + ResNet34 (mIoU: 0.748) |
| **Dataset Size** | ~2000+ images, 10 classes + background |
| **Key Challenge** | Grayscale images, class imbalance, sonar noise/artifacts |

---

> 💬 **Tip for your team:** Start simple — get a classification model working first. Then scale up to segmentation. This builds intuition about the data before tackling the harder problem. Good luck! 🚀
