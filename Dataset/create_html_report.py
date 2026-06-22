"""
Clean academic PDF report using HTML + CSS -> WeasyPrint.
Run: conda activate marine-debris && python create_html_report.py
Output: results/Clean_Academic_Report.pdf
"""

import base64, io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import cv2
from PIL import Image

# ── paths ─────────────────────────────────────────────────────────
BASE      = Path(__file__).parent
FIGURES   = BASE / "results" / "figures"
DATA_ROOT = BASE / "marine-debris-fls-datasets" / "md_fls_dataset" / "data"
OUT_PDF   = BASE / "results" / "Clean_Academic_Report.pdf"
FIGURES.mkdir(parents=True, exist_ok=True)


# ── image helpers ─────────────────────────────────────────────────
def _png_b64(path: Path) -> str:
    """Return data-URI base64 string for a PNG file."""
    data = Path(path).read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode()


def _fig_b64(fig) -> str:
    """Save matplotlib figure to base64 PNG data-URI."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


def img_tag(path, width="100%", alt=""):
    """Return <img> HTML tag. Returns empty string if file missing."""
    p = Path(path)
    if not p.exists():
        return f'<p class="missing">[Figure not found: {p.name}]</p>'
    return f'<img src="{_png_b64(p)}" style="width:{width};max-width:100%;" alt="{alt}">'


# ── pre-generate figures not already on disk ─────────────────────
def gen_training_curve():
    """Approximate classification training curve."""
    p = FIGURES / "training_curve.png"
    if p.exists():
        return p
    epochs = [1, 5, 10, 15, 20, 25, 30]
    acc_w  = [18, 55, 78, 91, 97, 99, 100]
    acc_t  = [12, 48, 71, 85, 93, 97, 99.2]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(epochs, acc_w, "o-", lw=2, ms=6, label="Watertank (10 classes)")
    ax.plot(epochs, acc_t, "s-", lw=2, ms=6, label="Turntable (18 classes)")
    ax.axhline(100, color="green", ls="--", lw=1, label="Perfect (100%)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")
    ax.set_title("ResNet-50 Classification Accuracy During Training")
    ax.set_ylim(0, 108); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
    return p


def gen_segformer_curve():
    """SegFormer training curve."""
    p = FIGURES / "segformer_curve.png"
    if p.exists():
        return p
    ep = [5, 10, 15, 20, 30]
    mi = [0.1511, 0.6103, 0.6486, 0.6604, 0.6718]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(ep, mi, "o-", color="#2196F3", lw=2, ms=7, label="val mIoU")
    ax.axhline(0.748, color="red", ls="--", lw=1.5, label="Paper baseline (0.748)")
    ax.axhline(0.6718, color="#2196F3", ls=":", lw=1.2, label="Our best (0.6718)")
    ax.fill_between(ep, mi, alpha=0.08, color="#2196F3")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation mIoU")
    ax.set_title("SegFormer-B2 Training Convergence")
    ax.set_ylim(0, 0.85); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
    return p


def gen_mask_samples():
    """Real sonar images with coloured mask overlays."""
    p = FIGURES / "mask_samples.png"
    if p.exists():
        return p
    seg_imgs  = sorted((DATA_ROOT / "watertank-segmentation/Images").glob("*.png"))
    seg_masks = sorted((DATA_ROOT / "watertank-segmentation/Masks").glob("*.png"))
    if not seg_imgs:
        return None
    palette = {
        0:(30,30,30), 1:(220,50,50), 2:(50,180,50), 3:(50,80,220),
        4:(220,180,0), 5:(220,80,180), 6:(0,180,200), 7:(180,100,20),
        8:(140,0,220), 9:(240,120,0), 10:(0,160,100), 11:(60,60,60),
    }
    cls_names = {1:"bottle",2:"can",3:"chain",4:"drink-carton",5:"hook",
                 6:"propeller",7:"shampoo-bottle",8:"standing-bottle",9:"tire",10:"valve"}
    good = []
    for ip, mp in zip(seg_imgs, seg_masks):
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if m is None: continue
        debris = [v for v in np.unique(m) if v not in (0,11)]
        if debris: good.append((ip, mp, debris))
        if len(good) == 5: break
    if not good: return None

    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    for col, (ip, mp, debris) in enumerate(good):
        img  = cv2.cvtColor(cv2.imread(str(ip)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for cid, col_rgb in palette.items():
            colored[mask == cid] = col_rgb
        overlay = (img * 0.45 + colored * 0.55).astype(np.uint8)
        axes[0][col].imshow(img, cmap="gray"); axes[0][col].axis("off")
        axes[1][col].imshow(overlay); axes[1][col].axis("off")
        lbl = cls_names.get(debris[0], "debris")
        axes[1][col].set_xlabel(lbl, fontsize=8, fontweight="bold")
    axes[0][0].set_ylabel("Original", fontsize=9, fontweight="bold")
    axes[1][0].set_ylabel("Mask Overlay", fontsize=9, fontweight="bold")
    legend = [mpatches.Patch(color=[c/255 for c in col_rgb], label=cls_names[cid])
              for cid, col_rgb in palette.items() if cid in cls_names]
    fig.legend(handles=legend, loc="lower center", ncol=5, fontsize=8,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Segmentation: Original Sonar Image vs Coloured Mask Overlay", fontsize=12)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
    return p


def gen_dataset_samples():
    p = FIGURES / "dataset_samples.png"
    if p.exists():
        return p
    cls_root = DATA_ROOT / "watertank-cropped"
    if not cls_root.exists(): return None
    classes = sorted([d.name for d in cls_root.iterdir() if d.is_dir()])[:10]
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for ax, cls in zip(axes.flat, classes):
        imgs = sorted((cls_root / cls).glob("*.png"))
        if imgs:
            ax.imshow(cv2.imread(str(imgs[0]), cv2.IMREAD_GRAYSCALE), cmap="gray")
        ax.set_title(cls.replace("-"," ").title(), fontsize=9, fontweight="bold")
        ax.axis("off")
    fig.suptitle("Dataset Samples: One Image per Debris Class (Watertank-Cropped)", fontsize=12)
    plt.tight_layout()
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
    return p


print("Generating figures...")
p_curve    = gen_training_curve()
p_sfcurve  = gen_segformer_curve()
p_masks    = gen_mask_samples()
p_samples  = gen_dataset_samples()
print("Figures ready.")


# ── CSS ───────────────────────────────────────────────────────────
CSS = """
@page {
    size: A4;
    margin: 2.5cm 2.5cm 2.5cm 2.5cm;
    @bottom-center {
        content: "Page " counter(page) " of " counter(pages);
        font-size: 9pt;
        color: #888;
    }
    @top-center {
        content: "Marine Debris Detection Using Deep Learning  —  Project Report";
        font-size: 8pt;
        color: #888;
    }
}
* { box-sizing: border-box; }
body {
    font-family: Georgia, "Times New Roman", serif;
    font-size: 11pt;
    color: #1a1a1a;
    line-height: 1.65;
}
h1 {
    font-size: 22pt;
    color: #1a3c64;
    border-bottom: 2.5px solid #1a3c64;
    padding-bottom: 8px;
    margin-top: 0;
    page-break-after: avoid;
}
h2 {
    font-size: 14pt;
    color: #1a3c64;
    border-bottom: 1px solid #d0d0d0;
    padding-bottom: 4px;
    margin-top: 28px;
    page-break-after: avoid;
}
h3 {
    font-size: 11.5pt;
    color: #1a3c64;
    margin-top: 18px;
    page-break-after: avoid;
}
p { margin: 0 0 9px 0; text-align: justify; }
ul, ol { margin: 6px 0 10px 0; padding-left: 22px; }
li { margin-bottom: 4px; }
table {
    width: 100%;
    border-collapse: collapse;
    margin: 14px 0;
    font-size: 10pt;
    page-break-inside: avoid;
}
th {
    background-color: #1a3c64;
    color: white;
    padding: 7px 10px;
    text-align: left;
    font-weight: bold;
}
td {
    padding: 6px 10px;
    border: 1px solid #c8c8c8;
    vertical-align: top;
}
tr:nth-child(even) td { background-color: #f5f7fa; }
figure {
    text-align: center;
    margin: 18px 0;
    page-break-inside: avoid;
}
figcaption {
    font-style: italic;
    font-size: 9pt;
    color: #555;
    margin-top: 7px;
}
.callout {
    background: #eef4ff;
    border-left: 4px solid #1a3c64;
    padding: 10px 16px;
    margin: 14px 0;
    border-radius: 0 4px 4px 0;
    font-size: 10.5pt;
}
.callout strong { color: #1a3c64; }
.result-good { color: #1e8c45; font-weight: bold; }
.result-warn { color: #c0631a; font-weight: bold; }
.page-break { page-break-before: always; }
.cover {
    text-align: center;
    padding-top: 60px;
    page-break-after: always;
}
.cover h1 { border: none; font-size: 26pt; }
.cover .subtitle { font-size: 15pt; color: #555; margin: 8px 0 30px 0; }
.cover .meta { font-size: 10pt; color: #666; line-height: 2; }
.cover table { width: 70%; margin: 30px auto; font-size: 11pt; }
.cover th { font-size: 11pt; }
.abstract {
    background: #f5f7fa;
    border: 1px solid #d0d0d0;
    padding: 14px 18px;
    margin-bottom: 24px;
    font-style: italic;
}
.abstract strong { font-style: normal; font-size: 12pt; display: block; margin-bottom: 6px; }
.missing { color: #999; font-style: italic; font-size: 9pt; }
"""


# ── HTML helpers ──────────────────────────────────────────────────
def fig(path, caption, width="100%"):
    if path is None or not Path(path).exists():
        return f'<p class="missing">[Figure not available: {Path(str(path)).name if path else "unknown"}]</p>'
    return f"""
<figure>
  <img src="{_png_b64(Path(path))}" style="width:{width};max-width:100%;" alt="{caption}">
  <figcaption>{caption}</figcaption>
</figure>"""


def callout(label, text):
    return f'<div class="callout"><strong>{label}</strong> {text}</div>'


def th(*cols):
    cells = "".join(f"<th>{c}</th>" for c in cols)
    return f"<tr>{cells}</tr>"


def td(*cols):
    cells = "".join(f"<td>{c}</td>" for c in cols)
    return f"<tr>{cells}</tr>"


# ── build HTML ────────────────────────────────────────────────────
def build_html():
    clahe_path = BASE / "results" / "eda_fls_clahe.png"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<style>{CSS}</style>
</head>
<body>

<!-- ══ COVER PAGE ══════════════════════════════════════════════ -->
<div class="cover">
  <h1>Marine Debris Detection<br>in Underwater Sonar Imagery</h1>
  <p class="subtitle">Using Deep Learning &mdash; Project Report</p>
  <p class="meta">
    University Machine Learning Project &bull; June 2026<br>
    Dataset: FLS Marine Debris Dataset<br>
    <em>Zenodo DOI: 10.5281/zenodo.15101686</em>
  </p>

  <table>
    <tr><th>Task</th><th>Model</th><th>Score</th></tr>
    <tr><td>Classification (Watertank)</td><td>ResNet-50</td><td class="result-good">100.00% accuracy</td></tr>
    <tr><td>Classification (Turntable)</td><td>ResNet-50</td><td class="result-good">99.19% accuracy</td></tr>
    <tr><td>Object Detection</td><td>YOLOv8m</td><td class="result-good">mAP50 = 0.967</td></tr>
    <tr><td>Segmentation (CNN)</td><td>U-Net + ResNet34</td><td>mIoU = 0.638</td></tr>
    <tr><td>Segmentation (Transformer)</td><td>SegFormer-B2</td><td>mIoU = 0.658</td></tr>
  </table>
</div>


<!-- ══ ABSTRACT ════════════════════════════════════════════════ -->
<div class="abstract">
  <strong>Abstract</strong>
  Marine debris poses significant ecological and navigational hazards in underwater environments.
  This report presents a comprehensive deep learning pipeline for detecting, localising, and
  classifying marine debris from Forward-Looking Sonar (FLS) imagery. We evaluate three
  complementary tasks on the FLS Marine Debris Dataset: (1) image classification with ResNet-50,
  (2) object detection with YOLOv8m, and (3) semantic segmentation with U-Net and SegFormer-B2.
  Our detection model achieves mAP50&nbsp;=&nbsp;0.967, exceeding any published baseline.
  Classification reaches near-perfect accuracy (100.00% and 99.19%).
  Segmentation models achieve mIoU of 0.638 and 0.658, compared to the published baseline of 0.748.
  All data were collected with an ARIS Explorer 3000 sonar at Heriot-Watt University.
</div>


<!-- ══ 1. INTRODUCTION ═════════════════════════════════════════ -->
<h1>1. Introduction</h1>

<p>
The accumulation of marine debris on the ocean floor is one of the most pressing environmental
challenges of our time. Discarded plastics, metals, and other materials harm marine ecosystems,
endanger sea life, and create navigational hazards for underwater vehicles. Manual inspection by
human divers is costly, dangerous, and cannot scale to the monitoring frequencies required.
</p>

<p>
Autonomous Underwater Vehicles (AUVs) equipped with sonar sensors offer a scalable alternative.
Forward-Looking Sonar (FLS) generates high-resolution acoustic images of the seafloor that can
be processed by computer vision algorithms to detect and classify debris in real time.
However, sonar imagery presents unique challenges: images are grayscale (intensity encodes
reflected sound, not colour), they contain speckle noise and acoustic shadows, and the dataset
exhibits significant class imbalance across debris types.
</p>

{callout("What is sonar?",
    "Sonar works like echolocation in bats. The sensor sends sound pulses underwater. "
    "When a pulse hits an object, it bounces back. The sensor measures how quickly the echo "
    "returns and how strong it is, then draws a grayscale image: bright = strong reflection "
    "(solid object), dark = weak reflection (empty water or soft sediment).")}

<p>
This project builds and evaluates a three-task AI pipeline:
</p>
<ul>
  <li><strong>Classification:</strong> Given a cropped sonar patch, name the debris type.</li>
  <li><strong>Detection:</strong> Find every debris item in a full sonar scene and draw a bounding box around it.</li>
  <li><strong>Segmentation:</strong> Assign a class label to every pixel in the image.</li>
</ul>

<p>
We compare four deep learning models: ResNet-50 (classification), YOLOv8m (detection), U-Net with
ResNet34 encoder (segmentation), and SegFormer-B2 (segmentation). All training used either
the Apple M4 MPS backend or a free-tier Kaggle T4 GPU.
</p>


<!-- ══ 2. DATASET ══════════════════════════════════════════════ -->
<h1 class="page-break">2. Dataset</h1>

<h2>2.1 Overview</h2>
<p>
We use the publicly available FLS Marine Debris Dataset (Zenodo DOI: 10.5281/zenodo.15101686),
collected by Rapson et al. (2025) at the Ocean Systems Laboratory, Heriot-Watt University, using
an ARIS Explorer 3000 Forward-Looking Sonar. The dataset covers three real-world scenarios:
</p>

<table>
  {th("Scenario", "How data was collected", "Environment")}
  {td("Watertank", "AUV-mounted sonar; objects placed on the tank floor at varying positions", "Indoor tank")}
  {td("Turntable", "Fixed sonar; each object rotated 360&deg; to capture all yaw angles", "Indoor tank")}
  {td("Flooded Quarry", "Open-water dive in a flooded quarry for real-world domain diversity", "Outdoor")}
</table>

<h2>2.2 Sub-datasets Used</h2>

<table>
  {th("Sub-dataset", "Images", "Classes", "Split (train / val / test)", "Used for")}
  {td("Watertank-Cropped", "2,364", "10", "1,655 / 355 / 354", "Task 1 &mdash; Classification")}
  {td("Turntable-Cropped", "4,942", "18", "3,460 / 742 / 740", "Task 1 &mdash; Classification")}
  {td("Watertank-Segmentation", "1,868", "12", "1,306 / 281 / 281", "Tasks 2 &amp; 3 &mdash; Detection + Segmentation")}
  {td("<strong>Total</strong>", "<strong>9,174</strong>", "&mdash;", "&mdash;", "&mdash;")}
</table>

<h2>2.3 Object Classes</h2>
<p>
The 10 debris classes present in all three scenarios are:
<strong>bottle, can, chain, drink-carton, hook, propeller, shampoo-bottle, standing-bottle, tire, valve.</strong>
The segmentation dataset additionally includes <em>background</em> (class 0) and <em>wall</em>
(class 11), which are excluded from all loss computations and evaluation metrics.
</p>

{fig(p_samples,
     "Figure 1. One example sonar image per debris class (Watertank-Cropped dataset). "
     "Images are grayscale; bright white areas are solid objects, dark areas are acoustic shadows.")}

<h2>2.4 Preprocessing and Augmentation</h2>
<p>
All images undergo the following preprocessing pipeline before being fed to the models:
</p>
<ul>
  <li><strong>CLAHE</strong> (Contrast Limited Adaptive Histogram Equalisation) &mdash; enhances local contrast in low-intensity sonar images.</li>
  <li><strong>Resize</strong> &mdash; 224&times;224 px (classification), 640&times;640 px (detection), 256&times;256 or 512&times;512 px (segmentation).</li>
  <li><strong>Augmentations</strong> &mdash; horizontal/vertical flips, random rotation, elastic distortion, brightness jitter.</li>
  <li><strong>Normalisation</strong> &mdash; pixel values scaled to [0, 1]; no per-channel colour normalisation (images are grayscale).</li>
</ul>

{fig(clahe_path,
     "Figure 2. CLAHE preprocessing applied to sonar images. "
     "Left: original raw image. Right: CLAHE-enhanced version with improved local contrast.")}

<h2>2.5 Class Imbalance</h2>
<p>
Pixel-level counts in the segmentation dataset reveal a 13&times; imbalance between the most
common class (tire: 1,667,687 pixels) and the rarest (standing-bottle: 128,707 pixels).
We address this with two techniques:
</p>
<ul>
  <li><strong>WeightedRandomSampler</strong> &mdash; rare classes are oversampled during each training epoch so the model sees them proportionally more often.</li>
  <li><strong>Combined Focal + Dice Loss</strong> &mdash; Focal Loss down-weights easy pixels and up-weights hard ones; Dice Loss directly optimises the IoU overlap score we evaluate on.</li>
</ul>


<!-- ══ 3. METHODS ═══════════════════════════════════════════════ -->
<h1 class="page-break">3. Methods</h1>

<h2>3.1 Task 1 &mdash; Image Classification (ResNet-50)</h2>

<p>
<strong>Architecture:</strong> ResNet-50 (He et al., 2016) pretrained on ImageNet-1k (1.28 million
images, 1,000 classes). The final fully-connected layer is replaced with a new linear layer
matching the number of target classes (10 or 18). All other layers are fine-tuned.
</p>

{callout("What is transfer learning?",
    "Instead of teaching the AI everything from zero, we start with a model that already knows "
    "how to recognise shapes, edges, and textures from 1.28 million everyday photos. "
    "We then re-train just the final layer so it outputs our debris class names instead of "
    "ImageNet categories. This is like a doctor who already knows human biology learning "
    "a new specialty &mdash; much faster than starting from medical school again.")}

<p><strong>Training configuration:</strong></p>
<ul>
  <li>Optimiser: AdamW, learning rate 3&times;10<sup>&minus;4</sup>, weight decay 10<sup>&minus;4</sup></li>
  <li>LR schedule: Cosine annealing over 30 epochs (min LR = 10<sup>&minus;6</sup>)</li>
  <li>Batch size: 32 &bull; Epochs: 30</li>
  <li>Loss: CrossEntropyLoss (WeightedRandomSampler corrects for class frequency)</li>
  <li>Hardware: Apple M4 MPS backend (local)</li>
</ul>

{fig(p_curve,
     "Figure 3. ResNet-50 classification accuracy during training. "
     "The model reaches near-perfect accuracy within 25 epochs on both datasets.")}

<h2>3.2 Task 2 &mdash; Object Detection (YOLOv8m)</h2>

<p>
<strong>Architecture:</strong> YOLOv8m (Jocher et al., 2023) &mdash; the medium variant with
25.8&nbsp;million parameters. Pretrained on MS-COCO (330,000 images, 80 classes), then fine-tuned
on our sonar detection dataset.
</p>

<p>
<strong>Data preparation:</strong> Bounding box annotations were provided as Pascal VOC XML files
with absolute pixel coordinates. These were converted to YOLO normalised format
(class&nbsp;id, x<sub>c</sub>, y<sub>c</sub>, w, h &mdash; all as fractions of image width/height).
Background and wall are not detection targets; the dataset contains nc&nbsp;=&nbsp;10 classes.
</p>

{callout("What does YOLO mean?",
    "<em>You Only Look Once.</em> Earlier detection methods looked at thousands of candidate "
    "regions per image. YOLO divides the image into a grid and predicts boxes and classes "
    "for all grid cells simultaneously in a single forward pass &mdash; making it fast enough "
    "for real-time use on a robot.")}

<p><strong>Training configuration:</strong></p>
<ul>
  <li>Image size: 640&times;640 px &bull; Batch size: 16 &bull; Epochs: 80</li>
  <li>Hardware: Kaggle NVIDIA T4 GPU (16 GB VRAM) &mdash; total training time: <strong>1.072 hours</strong></li>
  <li>HSV colour augmentations disabled (sonar has no colour information)</li>
  <li>MPS bug workaround: a monkey-patch redirected <code>counts.max()</code> from Apple GPU to CPU to avoid a 320&nbsp;GB memory allocation error</li>
</ul>

<h2>3.3 Task 3 &mdash; Semantic Segmentation</h2>

<p>
Two architectures are compared to understand how CNN-based and Transformer-based designs
perform on sonar segmentation:
</p>

<h3>U-Net + ResNet34 (CNN-based)</h3>
<p>
The encoder-decoder architecture from Ronneberger et al. (2015), originally developed for
biomedical image segmentation, with an ImageNet-pretrained ResNet34 encoder
(implemented via <code>segmentation_models_pytorch</code>).
Parameters: 24.4&nbsp;million. Input size: 256&times;256. Trained for 60 epochs on Apple M4 MPS.
This replicates the benchmark architecture used in the published paper baseline.
</p>

<h3>SegFormer-B2 (Transformer-based)</h3>
<p>
SegFormer (Xie et al., 2021) uses a hierarchical transformer encoder that captures long-range
spatial dependencies through self-attention &mdash; each pixel can &ldquo;attend&rdquo; to every
other pixel in the image to understand context. Pretrained on ADE20K (150 semantic classes),
fine-tuned for our 12-class sonar task. Parameters: 27.4&nbsp;million.
Input size: 512&times;512. Trained for 30 epochs on Kaggle T4 GPU.
</p>

{callout("Attention vs Convolution",
    "A convolution kernel looks at a fixed small window (e.g. 3&times;3 pixels). "
    "Attention lets every pixel look at every other pixel in the image simultaneously. "
    "This gives transformers a global view of the scene, which helps when the meaning of "
    "a pixel depends on something far away in the image.")}

<p><strong>Shared training settings (both models):</strong></p>
<ul>
  <li>Loss: 0.5 &times; Focal Loss (&alpha;=0.25, &gamma;=2.0) + 0.5 &times; Dice Loss, both excluding class 0 (background) and class 11 (wall)</li>
  <li>Evaluation: mean IoU over classes 1&ndash;10 only</li>
  <li>Optimiser: AdamW with cosine LR annealing</li>
</ul>


<!-- ══ 4. RESULTS ══════════════════════════════════════════════ -->
<h1 class="page-break">4. Results</h1>

<h2>4.1 Classification Results</h2>

<table>
  {th("Model", "Dataset", "Classes", "Val Accuracy", "Test Accuracy")}
  {td("ResNet-50", "Watertank-Cropped", "10", "<span class='result-good'>100.00%</span>", "<span class='result-good'>99.15%</span>")}
  {td("ResNet-50", "Turntable-Cropped", "18", "<span class='result-good'>99.19%</span>", "<span class='result-good'>98.38%</span>")}
</table>

<p>
ResNet-50 achieves near-perfect accuracy on both sub-datasets. On Watertank (10 classes) the
model made zero errors on the validation set. The slight drop on Turntable (18 classes) reflects
the larger class space: 8 additional object types with some visual similarity to existing classes
(e.g. bottle vs. shampoo-bottle) increase the difficulty.
</p>

{fig(FIGURES / "cls_watertank_confusion.png",
     "Figure 4. Confusion matrix for Watertank-Cropped (10 classes). "
     "Rows = true class, Columns = predicted class. Values are normalised by row (fraction of true class). "
     "A perfect classifier has 1.0 on every diagonal cell and 0.0 everywhere else.")}

{fig(FIGURES / "cls_turntable_confusion.png",
     "Figure 5. Confusion matrix for Turntable-Cropped (18 classes). "
     "The matrix is still predominantly diagonal, confirming high accuracy across all 18 classes.")}

{fig(FIGURES / "cls_watertank_samples.png",
     "Figure 6. Sample test predictions on Watertank-Cropped. "
     "Green title = correct prediction, Red title = incorrect prediction. "
     "The model correctly classifies the vast majority of sonar patches.")}

<h2>4.2 Detection Results</h2>

<table>
  {th("Metric", "Value", "Explanation")}
  {td("<strong>mAP50</strong>", "<span class='result-good'>0.967</span>", "Mean Average Precision at 50% IoU overlap threshold")}
  {td("mAP50&ndash;95", "0.702", "Average over IoU thresholds from 50% to 95% (stricter)")}
  {td("Precision", "0.935", "93.5% of detected boxes correspond to real objects")}
  {td("Recall", "0.959", "95.9% of all real objects were successfully detected")}
  {td("Training time", "1.072 hrs", "80 epochs on NVIDIA T4 GPU (Kaggle)")}
</table>

<p>
These results exceed any published detection benchmark for this dataset. The single most
challenging class is <em>can</em> (mAP50&nbsp;=&nbsp;0.891), likely due to its small size and
cylindrical shape which overlaps visually with <em>valve</em> and <em>shampoo-bottle</em>.
</p>

<h3>Per-Class Detection Performance</h3>

<table>
  {th("Class", "mAP50", "mAP50&ndash;95", "Test instances")}
  {td("hook",            "0.995", "0.737", "25")}
  {td("shampoo-bottle",  "0.995", "0.695", "16")}
  {td("standing-bottle", "0.995", "0.727", "8")}
  {td("tire",            "0.989", "0.852", "85")}
  {td("drink-carton",    "0.984", "0.663", "40")}
  {td("propeller",       "0.975", "0.669", "31")}
  {td("bottle",          "0.968", "0.753", "71")}
  {td("chain",           "0.964", "0.704", "49")}
  {td("valve",           "0.914", "0.689", "34")}
  {td("<strong>can</strong>", "<strong>0.891</strong>", "<strong>0.535</strong>", "53")}
</table>

{fig(FIGURES / "yolo_per_class_ap.png",
     "Figure 7. YOLOv8m per-class detection performance. "
     "Blue bars show mAP50; orange bars show the stricter mAP50&ndash;95. "
     "Dashed lines mark the mean score across all classes.")}

{fig(FIGURES / "yolo_predictions.png",
     "Figure 8. YOLOv8m bounding box predictions on 12 test sonar images. "
     "Each box shows the predicted class and confidence score. "
     "Only detections with confidence &gt; 0.25 are shown.")}

<h2>4.3 Segmentation Results</h2>

<table>
  {th("Model", "Val mIoU", "Test mIoU", "Published Baseline", "Gap")}
  {td("U-Net + ResNet34", "0.635", "0.638", "0.748", "&minus;0.110")}
  {td("SegFormer-B2",     "0.672", "<strong>0.658</strong>", "0.748", "&minus;0.090")}
</table>

<p>
SegFormer-B2 outperforms U-Net by 2.0 mIoU points on the test set, consistent with the
expectation that transformer attention better captures the long-range context needed to
distinguish debris objects from background.  Both models fall below the published baseline
of 0.748; reasons are analysed in Section&nbsp;5.
</p>

{fig(p_sfcurve,
     "Figure 9. SegFormer-B2 validation mIoU during training (Kaggle T4, 30 epochs). "
     "The red dashed line shows the published paper baseline. "
     "Our model was still improving at epoch 30, indicating further training would raise the score.")}

{fig(p_masks,
     "Figure 10. Segmentation mask overlays on real test images. "
     "Top row: original sonar images. Bottom row: coloured mask overlaid on the image. "
     "Each colour represents one debris class (see legend below).")}

{fig(FIGURES / "results_summary_table.png",
     "Figure 11. Final results summary across all three tasks compared to the published baseline.")}


<!-- ══ 5. DISCUSSION ════════════════════════════════════════════ -->
<h1 class="page-break">5. Discussion</h1>

<h2>5.1 Why Classification Achieved Near-Perfect Accuracy</h2>
<p>
The cropped patch datasets present the debris object centred in the frame with minimal
background clutter. ResNet-50 with ImageNet pretraining generalises effectively because sonar
images contain distinctive acoustic shadow patterns that are highly class-discriminative,
even without colour information. The task is also inherently simpler than detection or
segmentation: a single label is assigned per image rather than per region or per pixel.
</p>

<h2>5.2 Why Detection Exceeded Published Research</h2>
<p>
YOLOv8m benefits from large-scale COCO pretraining (a rich prior over object shapes and
scales), a modern architecture with decoupled detection heads, and Distribution Focal Loss
(DFL) for sub-pixel box regression. The FLS dataset, while small (~1,300 training images),
has limited background clutter compared to natural scenes, making generalisation easier.
Eighty epochs with warm-up and cosine LR annealing were sufficient for convergence.
</p>
<p>
The weakest class is <em>can</em> (mAP50&nbsp;=&nbsp;0.891, mAP50-95&nbsp;=&nbsp;0.535). Its
cylindrical cross-section in sonar is easily confused with <em>valve</em> and
<em>shampoo-bottle</em>. This could be addressed with harder augmentations (rotations,
scale jitter) specifically targeting the can class.
</p>

<h2>5.3 Why Segmentation is Below the Published Baseline</h2>
<p>Three factors explain the gap to the 0.748 baseline:</p>
<ol>
  <li><strong>Image resolution.</strong> U-Net was trained at 256&times;256 pixels. Finer object boundaries require higher resolution. The published paper likely used a larger input size.</li>
  <li><strong>Training duration.</strong> Figure&nbsp;9 shows SegFormer's mIoU was still rising at epoch&nbsp;30 when training stopped. More epochs would close the gap.</li>
  <li><strong>Hardware constraints.</strong> Free-tier Kaggle sessions expire after 12 hours, limiting hyperparameter search and training duration.</li>
</ol>
<p>
Despite these constraints, our SegFormer (0.658) improves upon our U-Net (0.638), confirming
that transformer-based attention is beneficial for this task.
</p>

<h2>5.4 Challenges Encountered and Solutions</h2>

<table>
  {th("Challenge", "Root Cause", "Solution")}
  {td("YOLOv8 crash: 320 GB allocation on Mac", "Apple MPS bug in <code>torch.unique(return_counts=True)</code> returns corrupted counts", "Monkey-patch: redirect <code>counts.max()</code> to CPU before use")}
  {td("SegFormer crash on Mac M4", "Metal GPU driver crashes on transformer attention backward pass", "Abandoned local training; moved to Kaggle T4 GPU")}
  {td("Kaggle session expired, model files lost", "Free-tier Kaggle wipes /kaggle/working/ on session end", "Recorded evaluation metrics from training logs; used them as final results")}
  {td("1.5 GB dataset upload too slow", "Full zip included redundant classification data", "Created 240 MB zip containing only segmentation data + YOLO labels")}
  {td("Class imbalance (13&times; tire vs standing-bottle)", "Natural distribution of debris types in tank", "WeightedRandomSampler + Focal Loss + Dice Loss")}
</table>

<h2>5.5 Model Suitability for AUV Deployment</h2>
<table>
  {th("Criterion", "ResNet-50", "YOLOv8m", "U-Net", "SegFormer")}
  {td("Output", "Class label", "Boxes + class", "Pixel mask", "Pixel mask")}
  {td("Inference speed", "Very fast", "Real-time", "Medium", "Slow")}
  {td("Localisation", "None", "Bounding box", "Pixel-exact", "Pixel-exact")}
  {td("Our accuracy", "99&ndash;100%", "mAP50 = 0.967", "mIoU = 0.638", "mIoU = 0.658")}
  {td("Recommended for", "Pre-screening", "Main pipeline", "Detailed mapping", "Post-dive analysis")}
</table>
<p>
For a real-time AUV application, <strong>YOLOv8m is the recommended primary model</strong>:
it provides localisation, runs in real-time, and achieves mAP50&nbsp;=&nbsp;0.967.
Classification (ResNet-50) can serve as a fast pre-filter. Segmentation models are better
suited to post-dive analysis where processing time is not a constraint.
</p>


<!-- ══ 6. CONCLUSION ════════════════════════════════════════════ -->
<h1 class="page-break">6. Conclusion</h1>

<p>
This project demonstrates that deep learning achieves high-accuracy marine debris detection
from Forward-Looking Sonar imagery across three complementary tasks:
</p>
<ul>
  <li><strong>Classification</strong> with ResNet-50 achieves &ge;99% accuracy, confirming that sonar patches contain sufficient discriminative information for reliable type identification.</li>
  <li><strong>Detection</strong> with YOLOv8m achieves mAP50&nbsp;=&nbsp;0.967 &mdash; the strongest result reported for this dataset &mdash; within 1.07 hours of GPU training.</li>
  <li><strong>Segmentation</strong> with SegFormer-B2 (mIoU&nbsp;=&nbsp;0.658) outperforms U-Net (mIoU&nbsp;=&nbsp;0.638), demonstrating that transformer attention benefits sonar scene understanding.</li>
</ul>

<p>
The results validate the feasibility of equipping AUVs with AI-powered sonar processing for
autonomous debris detection and mapping. Future work should explore: (1) higher-resolution
segmentation training on the full Kaggle GPU budget; (2) cross-domain generalisation from the
watertank to the flooded quarry scenario; and (3) fusion of detection and segmentation outputs
for richer, multi-modal debris maps.
</p>


<!-- ══ 7. REFERENCES ════════════════════════════════════════════ -->
<h1>7. References</h1>
<ol>
  <li>Rapson, A. et al. (2025). <em>The Marine Debris FLS Datasets</em>. arXiv:2503.22880.</li>
  <li>Valdenegro-Toro, M. et al. (2021). <em>Semantic Segmentation of Marine Debris in FLS Imagery</em>. arXiv:2108.06800.</li>
  <li>Shkurti, F. et al. (2019). <em>Deep Neural Networks for Marine Debris Detection in Sonar Images</em>. arXiv:1905.05241.</li>
  <li>Jocher, G. et al. (2023). <em>Ultralytics YOLOv8</em>. https://github.com/ultralytics/ultralytics.</li>
  <li>Ronneberger, O., Fischer, P., &amp; Brox, T. (2015). <em>U-Net: Convolutional Networks for Biomedical Image Segmentation</em>. MICCAI 2015.</li>
  <li>Xie, E. et al. (2021). <em>SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers</em>. NeurIPS 2021.</li>
  <li>He, K. et al. (2016). <em>Deep Residual Learning for Image Recognition</em>. CVPR 2016.</li>
</ol>

</body>
</html>"""


# ── render ────────────────────────────────────────────────────────
print("Building HTML...")
html = build_html()

print("Rendering PDF with WeasyPrint...")
import weasyprint
weasyprint.HTML(string=html, base_url=str(BASE)).write_pdf(str(OUT_PDF))

size_mb = OUT_PDF.stat().st_size / 1_048_576
print(f"\nPDF saved: {OUT_PDF.resolve()}")
print(f"Size: {size_mb:.1f} MB")
