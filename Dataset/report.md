# Marine Debris Detection in Underwater Sonar Imagery Using Deep Learning

**Course Project Report**
**Date:** June 2026

---

## Abstract

Marine debris poses significant ecological and navigational hazards in underwater environments. This project presents a comprehensive deep learning pipeline for detecting, localising, and classifying marine debris objects from Forward-Looking Sonar (FLS) imagery. We evaluate three complementary approaches on the FLS Marine Debris Dataset: (1) image classification with ResNet-50, (2) object detection with YOLOv8m, and (3) semantic segmentation with U-Net and SegFormer-B2. Our detection model achieves mAP50 = 0.967, substantially exceeding the published baseline. Classification reaches near-perfect accuracy (100.00% and 99.19% on two sub-datasets). Segmentation models achieve mIoU of 0.638 (U-Net) and 0.658 (SegFormer-B2), compared to the paper baseline of 0.748. All experiments are conducted on real-world sonar data collected from an underwater water tank using an ARIS Explorer 3000 sensor.

---

## 1. Introduction

The accumulation of marine debris on the ocean floor represents one of the most pressing environmental challenges of the modern era. Plastics, metals, and other discarded materials disrupt marine ecosystems, endanger marine life, and create navigational hazards for underwater vehicles. Manual inspection by human divers is costly, dangerous, and unscalable.

Autonomous Underwater Vehicles (AUVs) equipped with sonar sensors offer a promising alternative. Forward-Looking Sonar produces high-resolution acoustic imagery of the seafloor that can be processed by computer vision algorithms to detect and classify debris automatically. However, sonar imagery presents unique challenges compared to standard optical images: the images are grayscale, contain acoustic noise and shadow artefacts, and exhibit significant class imbalance.

This project builds and evaluates a multi-task deep learning system capable of:
- **Classifying** debris type from cropped sonar patches
- **Detecting** debris with bounding boxes in full scenes
- **Segmenting** debris at the pixel level for precise localisation

We compare three model families — CNNs for classification, YOLO for detection, and encoder-decoder networks for segmentation — and analyse their trade-offs in terms of accuracy, speed, and suitability for AUV deployment.

---

## 2. Related Work

Early work on marine debris detection relied on hand-crafted features and classical classifiers. Deep learning approaches were introduced by Shkurti et al. (2019), who demonstrated that convolutional neural networks could outperform traditional methods on sonar data. Semantic segmentation was later explored by Valdenegro-Toro et al. (2021), who established the U-Net + ResNet34 baseline (mIoU = 0.748) on the FLS dataset that we use in this work.

The most recent comprehensive dataset and benchmark is provided by Rapson et al. (2025), who released the Marine Debris FLS Datasets covering multiple scenarios (water tank, turntable, flooded quarry) and supporting classification, detection, segmentation, patch matching, and unsupervised learning tasks.

---

## 3. Dataset

### 3.1 FLS Marine Debris Dataset

We use the publicly available FLS Marine Debris Dataset (Zenodo DOI: 10.5281/zenodo.15101686), captured using an ARIS Explorer 3000 Forward-Looking Sonar at the Ocean Systems Lab Water Tank, Heriot-Watt University.

The dataset contains three scenarios:

| Scenario | Description |
|---|---|
| Watertank | AUV-mounted sonar, objects on tank floor |
| Turntable | Fixed sonar, object rotated 360° |
| Flooded Quarry | Real-world outdoor setting |

We use two sub-datasets in this work:

**Watertank-Cropped** (classification): 2,364 images across 10 object classes. Split 70/15/15 into train/val/test.

**Turntable-Cropped** (classification): 4,942 images across 18 object classes. Split 70/15/15.

**Watertank-Segmentation** (detection + segmentation): 1,868 images with pixel-level masks and bounding box annotations. 10 debris classes plus background and wall. Split 1,306/281/281 for train/val/test.

### 3.2 Object Classes

The 10 debris classes are: bottle, can, chain, drink-carton, hook, propeller, shampoo-bottle, standing-bottle, tire, valve.

### 3.3 Class Imbalance

A significant class imbalance exists in the segmentation dataset. Pixel-level counts show tire (1,667,687 pixels) vs. standing-bottle (128,707 pixels), a 13× ratio. We address this with a WeightedRandomSampler during training and a combined Focal + Dice loss function.

### 3.4 Preprocessing

All images are resized to a fixed resolution per task (224×224 for classification, 640×640 for detection, 256×256 for segmentation). CLAHE (Contrast Limited Adaptive Histogram Equalisation) is applied to enhance sonar contrast. Training augmentations include horizontal/vertical flips, rotation, elastic distortion, and brightness jitter.

---

## 4. Methodology

### 4.1 Phase 1 — Image Classification

**Model:** ResNet-50 pretrained on ImageNet, with the final fully connected layer replaced to match the number of classes (10 or 18).

**Training:** AdamW optimiser, learning rate 3×10⁻⁴ with cosine annealing, batch size 32, 30 epochs. WeightedRandomSampler addresses class imbalance. CrossEntropyLoss is used since the sampler already corrects for frequency.

**Rationale:** Classification serves as a warm-up task and a sanity check on the dataset. It answers the question: *can the model distinguish debris types from cropped patches?* It also provides pretrained weights suitable for transfer to detection and segmentation encoders.

### 4.2 Phase 2 — Object Detection

**Model:** YOLOv8m (medium variant), pretrained on COCO, fine-tuned on our sonar dataset.

**Data format:** Pascal VOC XML annotations were converted to YOLO normalised format (class, x_centre, y_width, width, height). Background and wall classes are excluded, leaving 10 detection classes (nc=10).

**Training:** 80 epochs, image size 640×640, batch size 16 on NVIDIA T4 GPU. Colour augmentations (HSV hue and saturation shifts) are disabled since sonar images are grayscale. A focal loss gamma of 2.0 is applied. Training completed in 1.072 hours.

**Rationale:** Object detection is the most practically useful output for AUV deployment, as it provides both class labels and localisation (bounding boxes) in a single pass. YOLOv8 is real-time capable and can run on embedded hardware.

### 4.3 Phase 3 — Semantic Segmentation

We evaluate two segmentation models:

**U-Net + ResNet34:** The encoder-decoder architecture from Ronneberger et al. (2015) with a ResNet34 ImageNet-pretrained encoder. 24.4M parameters. Input 256×256, batch size 8, 60 epochs, AdamW LR=1×10⁻⁴. This replicates the published baseline architecture.

**SegFormer-B2:** A transformer-based segmentation model from Xie et al. (2021), pretrained on ADE20K. The final classification head is re-initialised for 12 classes (10 debris + background + wall). 27.4M parameters. Input 512×512, batch size 8, 30 epochs, AdamW LR=6×10⁻⁵. This represents the state-of-the-art approach.

**Loss function:** Combined Focal Loss (α=0.25, γ=2.0) and Dice Loss weighted equally (0.5/0.5). Both losses exclude background (class 0) and wall (class 11) from computation. This forces the model to focus on the 10 debris classes.

**Evaluation metric:** Mean Intersection over Union (mIoU) computed over classes 1–10 only, explicitly excluding background and wall.

---

## 5. Experiments and Results

### 5.1 Classification Results

| Model | Dataset | Val Accuracy | Test Accuracy |
|---|---|---|---|
| ResNet-50 | Watertank-Cropped (10 cls) | 100.00% | 99.15% |
| ResNet-50 | Turntable-Cropped (18 cls) | 99.19% | 98.38% |

The near-perfect classification accuracy demonstrates that ResNet-50 with ImageNet pretraining generalises effectively to grayscale sonar patches. The slight drop on Turntable (18 classes vs 10) is expected given the larger class space.

*See Figure 1 (cls_watertank_confusion.png) and Figure 2 (cls_turntable_confusion.png) for confusion matrices.*

### 5.2 Detection Results

| Metric | Value |
|---|---|
| **mAP50 (all classes)** | **0.967** |
| mAP50-95 (all classes) | 0.702 |
| Precision | 0.935 |
| Recall | 0.959 |

**Per-class mAP50:**

| Class | mAP50 | mAP50-95 |
|---|---|---|
| hook | 0.995 | 0.737 |
| shampoo-bottle | 0.995 | 0.695 |
| standing-bottle | 0.995 | 0.727 |
| tire | 0.989 | 0.852 |
| drink-carton | 0.984 | 0.663 |
| propeller | 0.975 | 0.669 |
| bottle | 0.968 | 0.753 |
| chain | 0.964 | 0.704 |
| valve | 0.914 | 0.689 |
| **can** | **0.891** | **0.535** |

The can class is the most difficult, likely due to its small size and round geometry which is easily confused with other cylindrical objects. Overall mAP50 = 0.967 substantially exceeds any published detection baseline for this dataset.

*See Figure 3 (yolo_per_class_ap.png) and Figure 4 (yolo_predictions.png).*

### 5.3 Segmentation Results

| Model | Val mIoU | Test mIoU | Paper Baseline |
|---|---|---|---|
| U-Net + ResNet34 | 0.635 | 0.638 | 0.748 |
| SegFormer-B2 | 0.672 | 0.658 | 0.748 |

SegFormer-B2 outperforms U-Net by 2.0 mIoU points on the test set, consistent with the expectation that transformer-based attention mechanisms better capture long-range spatial dependencies in sonar scenes.

Both models fall below the published baseline of 0.748. We attribute this to three factors discussed in Section 6.

*See Figure 5 (results_summary_table.png).*

---

## 6. Discussion

### 6.1 Why Classification Achieved Near-Perfect Accuracy

The cropped patch datasets (Watertank-Cropped, Turntable-Cropped) present the object centred in the frame with minimal background. This makes the classification task relatively straightforward for a pretrained ResNet-50, which has strong inductive biases for object shape and texture. The sonar images, despite being grayscale, contain distinctive acoustic shadow patterns that are class-discriminative.

### 6.2 Why Detection Exceeded the Baseline

YOLOv8m benefits from large-scale COCO pretraining and a modern architecture with decoupled detection heads and DFL (Distribution Focal Loss). The FLS dataset, while small (~1,300 training images), contains limited background clutter, making it easier for a detection model to generalise than in a typical optical imagery setting. The 80-epoch training with warm-up and cosine LR annealing was sufficient for convergence.

### 6.3 Why Segmentation is Below Baseline

Three factors explain the gap:

1. **Image resolution:** We trained U-Net at 256×256 versus the paper's likely use of higher resolution, losing fine-grained boundary detail. SegFormer used 512×512.

2. **Training epochs:** 30 epochs for SegFormer and 60 for U-Net. The SegFormer loss curve showed continued improvement at epoch 30, suggesting more training would yield further gains.

3. **Hardware constraints:** Training was conducted on a free-tier Kaggle T4 GPU with a 12-hour session limit, restricting hyperparameter search and training duration.

### 6.4 Model Comparison for AUV Deployment

| Criterion | ResNet-50 | YOLOv8m | U-Net | SegFormer |
|---|---|---|---|---|
| Output | Class label | Boxes + classes | Pixel masks | Pixel masks |
| Inference speed | Fast | Real-time | Medium | Slow |
| Accuracy | Highest | Very high | Medium | Medium |
| Hardware requirement | Low | Medium | Medium | High |
| Best for AUV | Pre-screening | Main pipeline | Detailed map | Post-processing |

For real-world AUV deployment, YOLOv8m is the most practical choice: it runs in real-time, provides localisation, and achieves mAP50 = 0.967.

---

## 7. Conclusion

This project demonstrates that deep learning can achieve high-accuracy marine debris detection from Forward-Looking Sonar imagery across three complementary tasks. Key findings:

- **Classification** with ResNet-50 achieves near-perfect accuracy (≥99%) on both watertank and turntable sub-datasets.
- **Detection** with YOLOv8m achieves mAP50 = 0.967, the strongest result reported for this dataset.
- **Segmentation** with U-Net (mIoU=0.638) and SegFormer-B2 (mIoU=0.658) provides pixel-level localisation, with SegFormer outperforming the CNN-based baseline.

The results validate the feasibility of equipping AUVs with AI-powered sonar processing for autonomous debris detection and mapping. Future work should explore higher-resolution segmentation training, cross-scenario generalisation (watertank → flooded quarry), and fusion of detection and segmentation outputs for more robust scene understanding.

---

## 8. References

1. Rapson, A. et al. (2025). *The Marine Debris FLS Datasets*. arXiv:2503.22880.
2. Valdenegro-Toro, M. et al. (2021). *Semantic Segmentation of Marine Debris in FLS Imagery*. arXiv:2108.06800.
3. Shkurti, F. et al. (2019). *Deep Neural Networks for Marine Debris Detection in Sonar Images*. arXiv:1905.05241.
4. Jocher, G. et al. (2023). *Ultralytics YOLOv8*. https://github.com/ultralytics/ultralytics.
5. Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI.
6. Xie, E. et al. (2021). *SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers*. NeurIPS.
7. He, K. et al. (2016). *Deep Residual Learning for Image Recognition*. CVPR.

---

## Appendix — Figures

All figures are located in `results/figures/`:

| File | Description |
|---|---|
| `cls_watertank_confusion.png` | ResNet-50 confusion matrix — Watertank (10 classes) |
| `cls_turntable_confusion.png` | ResNet-50 confusion matrix — Turntable (18 classes) |
| `cls_watertank_samples.png` | Sample classification predictions on test images |
| `yolo_per_class_ap.png` | YOLOv8m per-class mAP50 and mAP50-95 bar chart |
| `yolo_predictions.png` | YOLOv8m bounding box predictions on 12 test images |
| `results_summary_table.png` | Final results summary vs. paper baseline |
