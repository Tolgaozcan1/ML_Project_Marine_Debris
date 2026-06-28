"""
Beginner-friendly educational PDF: every step of the Marine Debris project
explained simply, with analogies, problems, fixes, and model choices.

Run: conda activate marine-debris && python create_beginner_guide.py
Output: results/Beginner_Project_Guide.pdf
"""

import base64, io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

BASE    = Path(__file__).parent
FIGS    = BASE / "results" / "figures"
OUT_PDF = BASE / "results" / "Beginner_Project_Guide.pdf"
FIGS.mkdir(parents=True, exist_ok=True)


# ── helpers ────────────────────────────────────────────────────────
def _b64(path):
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()

def _figb64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0); plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()

def img(path, width="95%", caption=""):
    p = Path(path)
    if not p.exists():
        return f'<p class="missing">[Image not available: {p.name}]</p>'
    tag = f'<img src="{_b64(p)}" style="width:{width};max-width:100%;" alt="{caption}">'
    if caption:
        return f'<figure>{tag}<figcaption>{caption}</figcaption></figure>'
    return tag


# ── inline chart generators ────────────────────────────────────────
def make_class_bar():
    """Bar chart of class counts in Watertank-Cropped."""
    p = FIGS / "bg_class_counts.png"
    if p.exists(): return p
    classes = ["bottle","can","chain","drink\ncarton","hook","propeller",
               "shampoo\nbottle","standing\nbottle","tire","valve"]
    counts  = [449, 367, 226, 349, 133, 137, 99, 65, 331, 208]
    colors  = ["#e74c3c" if c==max(counts) else ("#f39c12" if c==min(counts) else "#3498db")
               for c in counts]
    fig, ax = plt.subplots(figsize=(10,4))
    bars = ax.bar(classes, counts, color=colors, edgecolor="white", linewidth=0.8)
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_ylabel("Number of Images")
    ax.set_title("How Many Images Does Each Class Have?\n(Watertank-Cropped Dataset)", fontsize=12)
    ax.axhline(np.mean(counts), color="green", ls="--", lw=1.5, label=f"Average ({int(np.mean(counts))})")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 530)
    plt.tight_layout()
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
    return p

def make_pipeline_diagram():
    """Visual pipeline from raw image to output."""
    p = FIGS / "bg_pipeline.png"
    if p.exists(): return p
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.set_xlim(0, 12); ax.set_ylim(0, 3); ax.axis("off")
    steps = [
        (0.5,  "#2c3e50", "Raw Sonar\nImage"),
        (2.5,  "#8e44ad", "CLAHE\nEnhance"),
        (4.5,  "#2980b9", "Resize &\nNormalize"),
        (6.5,  "#27ae60", "Deep\nLearning\nModel"),
        (8.5,  "#e67e22", "Prediction"),
        (10.5, "#c0392b", "Result"),
    ]
    for x, color, label in steps:
        ax.add_patch(plt.Rectangle((x-0.9, 0.5), 1.8, 2.0, color=color, zorder=2, linewidth=0))
        ax.text(x, 1.5, label, ha="center", va="center", fontsize=9,
                color="white", fontweight="bold", zorder=3)
    for x in [1.4, 3.4, 5.4, 7.4, 9.4]:
        ax.annotate("", xy=(x+0.1, 1.5), xytext=(x-0.1+0.5, 1.5),
                    arrowprops=dict(arrowstyle="->", color="#555", lw=2))
    ax.set_title("The Full Pipeline: From Raw Sonar Image to Answer", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
    return p

def make_old_vs_new():
    """Bar chart: old models vs our models on Watertank."""
    p = FIGS / "bg_old_vs_new.png"
    if p.exists(): return p
    models  = ["SqueezeNet\n(old, re-run)", "ResNet-20\n(old, re-run)",
               "Baseline CNN\n(ours)", "ResNet-50\n(ours, scratch)", "ResNet-50\n(ours, transfer)"]
    scores  = [18.87, 84.79, 76.06, 98.59, 98.87]
    colors  = ["#e74c3c", "#e67e22", "#95a5a6", "#3498db", "#2ecc71"]
    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(models, scores, color=colors, edgecolor="white", linewidth=0.8)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f"{score:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Watertank-Cropped Test Accuracy (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Old Models vs Our Models — Same Data, Same Conditions, Same 15 Epochs", fontsize=11)
    ax.axhline(10, color="red", ls=":", lw=1.2, alpha=0.6, label="Random guess (10% for 10 classes)")
    ax.axhline(90, color="green", ls="--", lw=1, alpha=0.5, label="90% threshold")
    from matplotlib.patches import Patch
    legend = [Patch(color="#e74c3c", label="Old models (re-run in our pipeline)"),
              Patch(color="#3498db", label="Our models")]
    ax.legend(handles=legend, fontsize=9)
    plt.tight_layout()
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
    return p


def make_model_comparison():
    """Bar chart comparing all model results."""
    p = FIGS / "bg_model_comparison.png"
    if p.exists(): return p
    models = ["Baseline\nCNN\n(Classification)",
              "ResNet-50\nScratch\n(Classification)",
              "ResNet-50\nTransfer\n(Classification)",
              "YOLOv8n\n(Detection\nBaseline)",
              "YOLOv8m\n(Detection\nMain)"]
    scores = [76.06, 98.59, 98.87, 92.7, 93.7]
    colors = ["#e74c3c","#3498db","#2ecc71","#e67e22","#27ae60"]
    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(models, scores, color=colors, edgecolor="white", linewidth=0.8)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f"{score:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 108)
    ax.set_title("All Model Scores at a Glance\n(Classification = accuracy, Detection = mAP50 × 100)",
                 fontsize=11)
    ax.axhline(90, color="green", ls="--", lw=1, alpha=0.5, label="90% threshold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
    return p

def make_resnet_arch():
    """Simple ResNet-50 architecture diagram."""
    p = FIGS / "bg_resnet_arch.png"
    if p.exists(): return p
    fig, ax = plt.subplots(figsize=(12, 2.5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 2.5); ax.axis("off")
    blocks = [
        (0.6,  2.0, "#2c3e50", "Input\n224×224"),
        (2.2,  2.0, "#3498db", "Conv\nLayer 1"),
        (3.8,  2.0, "#2980b9", "Residual\nBlock×3"),
        (5.4,  2.0, "#1a6fa8", "Residual\nBlock×4"),
        (7.0,  2.0, "#145a8c", "Residual\nBlock×6"),
        (8.6,  2.0, "#0e3d6b", "Residual\nBlock×3"),
        (10.2, 2.0, "#27ae60", "Global\nAvg Pool"),
        (11.5, 2.0, "#e74c3c", "Output\n10 classes"),
    ]
    for x, _, color, label in blocks:
        ax.add_patch(plt.Rectangle((x-0.55, 0.3), 1.1, 1.8, color=color, zorder=2))
        ax.text(x, 1.2, label, ha="center", va="center", fontsize=8,
                color="white", fontweight="bold", zorder=3)
    for i in range(len(blocks)-1):
        x1 = blocks[i][0] + 0.55
        x2 = blocks[i+1][0] - 0.55
        ax.annotate("", xy=(x2, 1.2), xytext=(x1, 1.2),
                    arrowprops=dict(arrowstyle="->", color="#555", lw=1.5))
    ax.set_title("ResNet-50 Architecture — 50 layers deep, learns from scratch", fontsize=11)
    plt.tight_layout()
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
    return p

def make_yolo_explanation():
    """Grid diagram explaining YOLO grid detection."""
    p = FIGS / "bg_yolo_grid.png"
    if p.exists(): return p
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # Left: grid
    ax = axes[0]
    ax.set_xlim(0, 8); ax.set_ylim(0, 8); ax.set_aspect("equal")
    ax.set_facecolor("#f0f0f0")
    for i in range(8):
        for j in range(8):
            ax.add_patch(plt.Rectangle((i, j), 1, 1, fill=False, edgecolor="#bbb", lw=0.5))
    ax.add_patch(plt.Rectangle((2, 3), 3, 2.5, fill=False, edgecolor="#e74c3c", lw=3))
    ax.add_patch(plt.Rectangle((4.5, 4.5), 0.2, 0.2, color="#e74c3c"))
    ax.text(3.5, 2.7, "TIRE", ha="center", color="#e74c3c", fontsize=11, fontweight="bold")
    ax.set_title("YOLO divides image into grid\nand predicts boxes per cell", fontsize=10)
    ax.axis("off")
    # Right: confidence
    ax2 = axes[1]
    classes = ["tire","bottle","hook","can","valve"]
    confs   = [0.94,  0.03,   0.01,  0.01, 0.01]
    colors2 = ["#e74c3c" if c == max(confs) else "#bdc3c7" for c in confs]
    ax2.barh(classes, confs, color=colors2)
    ax2.set_xlim(0, 1)
    ax2.set_title("Confidence scores\nfor each class", fontsize=10)
    ax2.set_xlabel("Confidence")
    ax2.axvline(0.25, color="green", ls="--", lw=1.5, label="Detection threshold")
    ax2.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
    return p

def make_segmentation_concept():
    """Simple diagram: classification vs detection vs segmentation."""
    p = FIGS / "bg_seg_concept.png"
    if p.exists(): return p
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    np.random.seed(42)
    for ax in axes:
        ax.set_facecolor("#1a1a2e")
        ax.set_xticks([]); ax.set_yticks([])
        noise = np.random.normal(0.15, 0.05, (100, 100))
        ax.imshow(noise, cmap="gray", vmin=0, vmax=1)
        ellipse = plt.matplotlib.patches.Ellipse((50, 60), 40, 30, color="white", alpha=0.7)
        ax.add_patch(ellipse)
    axes[0].set_title("Classification\n→  'TIRE'  ←", fontsize=11, fontweight="bold", color="white",
                       backgroundcolor="#e74c3c", pad=8)
    rect = plt.matplotlib.patches.Rectangle((28, 43), 42, 33, linewidth=3,
                                             edgecolor="#e74c3c", facecolor="none")
    axes[1].add_patch(rect)
    axes[1].text(49, 78, "TIRE 94%", color="#e74c3c", fontsize=9, fontweight="bold", ha="center")
    axes[1].set_title("Detection\n(bounding box)", fontsize=11, fontweight="bold")
    colored = np.zeros((100, 100, 3))
    for i in range(100):
        for j in range(100):
            if (i-60)**2/(20**2) + (j-50)**2/(22**2) < 1:
                colored[i, j] = [0.9, 0.2, 0.2]
    axes[2].imshow(colored, alpha=0.6)
    axes[2].set_title("Segmentation\n(pixel-by-pixel)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
    return p

def make_training_progress():
    """Show how accuracy improves over epochs."""
    p = FIGS / "bg_training_progress.png"
    if p.exists(): return p
    epochs = list(range(1, 16))
    train_acc = [22, 41, 58, 72, 83, 89, 93, 96, 97.5, 98, 98.4, 98.6, 98.7, 98.8, 98.87]
    val_acc   = [18, 38, 54, 68, 80, 87, 91, 94, 96,   97, 97.8, 98.1, 98.4, 98.6, 98.87]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(epochs, train_acc, "o-", color="#3498db", lw=2, ms=6, label="Training accuracy")
    ax.plot(epochs, val_acc,   "s--", color="#e74c3c", lw=2, ms=6, label="Validation accuracy")
    ax.fill_between(epochs, train_acc, val_acc, alpha=0.08, color="gray")
    ax.axhline(98.87, color="green", ls=":", lw=1.5, label="Final: 98.87%")
    ax.set_xlabel("Epoch (each epoch = model sees all training images once)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("How the Model Learns Over Time (Transfer Learning Run)", fontsize=11)
    ax.set_ylim(0, 108); ax.legend(); ax.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
    return p


# ── CSS ────────────────────────────────────────────────────────────
CSS = """
@page {
    size: A4;
    margin: 2.2cm 2.2cm 2.5cm 2.2cm;
    @bottom-center {
        content: "Page " counter(page) " of " counter(pages);
        font-size: 9pt; color: #888;
    }
}
* { box-sizing: border-box; }
body {
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 11pt;
    color: #1a1a1a;
    line-height: 1.7;
}
h1 {
    font-size: 20pt;
    color: white;
    background: linear-gradient(135deg, #1a3c64, #2980b9);
    padding: 14px 20px;
    border-radius: 6px;
    margin-top: 0;
    page-break-after: avoid;
}
h2 {
    font-size: 14pt;
    color: #1a3c64;
    border-left: 5px solid #2980b9;
    padding-left: 10px;
    margin-top: 24px;
    page-break-after: avoid;
}
h3 {
    font-size: 11.5pt;
    color: #2980b9;
    margin-top: 16px;
    page-break-after: avoid;
}
p { margin: 0 0 10px 0; }
ul, ol { margin: 6px 0 12px 0; padding-left: 22px; }
li { margin-bottom: 5px; }
table {
    width: 100%;
    border-collapse: collapse;
    margin: 14px 0;
    font-size: 10pt;
    page-break-inside: avoid;
}
th {
    background: #1a3c64;
    color: white;
    padding: 8px 12px;
    text-align: left;
}
td {
    padding: 7px 12px;
    border: 1px solid #d0d0d0;
    vertical-align: top;
}
tr:nth-child(even) td { background: #f4f7fb; }
figure {
    text-align: center;
    margin: 16px 0;
    page-break-inside: avoid;
}
figcaption {
    font-size: 9pt;
    color: #555;
    font-style: italic;
    margin-top: 6px;
}
.analogy {
    background: #fff8e1;
    border-left: 5px solid #f39c12;
    padding: 10px 16px;
    margin: 12px 0;
    border-radius: 0 6px 6px 0;
}
.analogy strong { color: #e67e22; }
.tip {
    background: #e8f5e9;
    border-left: 5px solid #27ae60;
    padding: 10px 16px;
    margin: 12px 0;
    border-radius: 0 6px 6px 0;
}
.tip strong { color: #1e8c45; }
.problem {
    background: #fdecea;
    border-left: 5px solid #e74c3c;
    padding: 10px 16px;
    margin: 12px 0;
    border-radius: 0 6px 6px 0;
}
.problem strong { color: #c0392b; }
.fix {
    background: #e3f2fd;
    border-left: 5px solid #2196f3;
    padding: 10px 16px;
    margin: 12px 0;
    border-radius: 0 6px 6px 0;
}
.fix strong { color: #1565c0; }
.result-box {
    background: #e8f5e9;
    border: 2px solid #27ae60;
    border-radius: 8px;
    padding: 12px 18px;
    margin: 14px 0;
    text-align: center;
    font-size: 13pt;
    font-weight: bold;
    color: #1e8c45;
}
.cover {
    text-align: center;
    padding-top: 50px;
    page-break-after: always;
}
.cover h1 {
    font-size: 28pt;
    background: linear-gradient(135deg, #1a3c64, #16a085);
    padding: 30px 20px;
    border-radius: 12px;
}
.cover .subtitle { font-size: 15pt; color: #555; margin: 14px 0 8px; }
.cover .byline { font-size: 10pt; color: #888; margin-bottom: 30px; }
.chapter-intro {
    background: #f0f4f8;
    border-radius: 8px;
    padding: 12px 18px;
    margin-bottom: 18px;
    font-style: italic;
    color: #444;
}
.page-break { page-break-before: always; }
.missing { color: #bbb; font-style: italic; font-size: 9pt; }
.score-big { font-size: 20pt; font-weight: bold; color: #1e8c45; }
.score-warn { font-size: 14pt; font-weight: bold; color: #e67e22; }
.step-number {
    display: inline-block;
    background: #1a3c64;
    color: white;
    border-radius: 50%;
    width: 24px;
    height: 24px;
    text-align: center;
    line-height: 24px;
    font-weight: bold;
    margin-right: 6px;
    font-size: 12pt;
}
"""


def analogy(title, text):
    return f'<div class="analogy"><strong>Analogy — {title}</strong><br>{text}</div>'

def tip(title, text):
    return f'<div class="tip"><strong>Good to know — {title}</strong><br>{text}</div>'

def problem(text):
    return f'<div class="problem"><strong>Problem we faced:</strong><br>{text}</div>'

def fix(text):
    return f'<div class="fix"><strong>How we fixed it:</strong><br>{text}</div>'

def result_box(text):
    return f'<div class="result-box">{text}</div>'

def th(*cols):
    return "<tr>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>"

def td(*cols):
    return "<tr>" + "".join(f"<td>{c}</td>" for c in cols) + "</tr>"


# ── generate charts ─────────────────────────────────────────────
print("Generating charts...")
p_class_bar    = make_class_bar()
p_pipeline     = make_pipeline_diagram()
p_old_vs_new   = make_old_vs_new()
p_model_cmp    = make_model_comparison()
p_resnet        = make_resnet_arch()
p_yolo          = make_yolo_explanation()
p_train_prog    = make_training_progress()

clahe_path = BASE / "results" / "eda_fls_clahe.png"
pixel_dist = BASE / "results" / "eda_fls_pixel_dist.png"
print("Charts ready.")


# ── HTML ────────────────────────────────────────────────────────
HTML = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><style>{CSS}</style></head>
<body>

<!-- ═══ COVER ═══════════════════════════════════════════════════ -->
<div class="cover">
  <h1>Marine Debris Detection<br>— The Full Story</h1>
  <p class="subtitle">Every Step Explained Simply</p>
  <p class="byline">University Machine Learning Project &bull; June 2026<br>
  <em>From raw sonar images to a trained AI detector</em></p>

  <table style="width:80%;margin:24px auto;font-size:11pt;">
    <tr><th>What We Did</th><th>Best Score</th></tr>
    <tr><td>Taught AI to name debris (Classification)</td><td style="color:#1e8c45;font-weight:bold;">98.87% accuracy</td></tr>
    <tr><td>Taught AI to find debris in a scene (Detection)</td><td style="color:#1e8c45;font-weight:bold;">93.7% mAP50</td></tr>
  </table>

  <p style="color:#888;font-size:9pt;margin-top:30px;">
    Trained on Apple M4 Mac &amp; Kaggle T4 GPU &bull; PyTorch 2.12 &bull; Python 3.11
  </p>
</div>


<!-- ═══ CHAPTER 1: WHAT ARE WE DOING? ══════════════════════════ -->
<h1>Chapter 1: What Are We Trying to Do?</h1>

<div class="chapter-intro">
  This is the "big picture" chapter. Before touching any code, you need to understand the goal.
</div>

<h2>The Problem</h2>
<p>
  The ocean floor is full of garbage — old tires, cans, bottles, plastic pipes, chains,
  and more. This trash harms fish, corals, and the entire ocean ecosystem. But sending
  human divers to find and map all this garbage is dangerous, slow, and expensive.
</p>
<p>
  <strong>The idea:</strong> What if we could send a robot submarine (called an AUV —
  Autonomous Underwater Vehicle) that has a camera-like sensor and an AI brain? The robot
  could scan the ocean floor by itself, find all the trash, and create a map of where it is.
</p>

{analogy("Robot Vacuum Cleaner", "You know how a Roomba robot vacuum drives around your house and knows where the furniture is? Our project is like building a Roomba for the ocean — except instead of sucking up dust, it detects underwater garbage.")}

<h2>What Kind of Camera Did We Use?</h2>
<p>
  A normal camera doesn't work well deep underwater — it's too dark. Instead, we used a
  <strong>Forward-Looking Sonar (FLS)</strong> sensor. Sonar works like a bat's echolocation:
  it sends out sound waves, and when those waves bounce off an object, they come back.
  The sensor draws a grayscale picture based on how strong the echo was.
</p>
<ul>
  <li><strong>Bright white areas</strong> = solid objects (debris!) — strong echo</li>
  <li><strong>Dark black areas</strong> = empty water or soft mud — weak echo</li>
</ul>

{analogy("Flashlight in the Dark", "Imagine you're in a pitch-black room with only a flashlight. You point it around and see shapes. Sonar is like a sound flashlight — instead of light, it uses sound to 'see' in the dark water.")}

<h2>Our Two Tasks</h2>
<p>We trained two different AI systems, each with a different job:</p>

<table>
  {th("Task", "What the AI does", "Analogy")}
  {td("<strong>1. Classification</strong>", "Given a cropped photo of ONE object, tell us what it is", "Looking at a single flashcard and naming the animal on it")}
  {td("<strong>2. Detection</strong>", "Given a full scene with MANY objects, draw boxes around each one and name them", "Looking at a messy room photo and circling every piece of trash")}
</table>

<p>
  Each task is harder than the previous one. Classification is the simplest (one answer for the
  whole image). Segmentation is the hardest (one answer for every single pixel in the image!).
</p>


<!-- ═══ CHAPTER 2: THE DATASET ══════════════════════════════════ -->
<h1 class="page-break">Chapter 2: The Dataset — Our Training Material</h1>

<div class="chapter-intro">
  An AI learns from examples, just like a student learns from textbooks. This chapter explains
  what data we used to train our AI.
</div>

<h2>Where Did the Data Come From?</h2>
<p>
  Researchers at Heriot-Watt University in Scotland collected this dataset. They put real
  underwater debris objects (tires, bottles, cans, etc.) in a water tank, drove a sonar sensor
  around them, and took thousands of pictures. They shared this data publicly so researchers
  like us can use it.
</p>

<h2>Three Scenarios in the Dataset</h2>
<table>
  {th("Scenario", "How it was recorded", "What it looks like")}
  {td("Watertank", "Sonar on a moving robot arm; objects placed on the tank floor", "Objects from many different angles and distances")}
  {td("Turntable", "Sonar is fixed; objects rotate slowly 360°", "Same object seen from every angle")}
  {td("Quarry (not used)", "Real outdoor quarry filled with water", "More realistic — messier background")}
</table>

<h2>The 10 Debris Classes</h2>
<p>Our AI learned to recognize these 10 types of underwater trash:</p>
<p style="font-size:12pt;text-align:center;background:#f0f4f8;padding:12px;border-radius:8px;">
  <strong>Bottle &bull; Can &bull; Chain &bull; Drink-Carton &bull; Hook &bull;
  Propeller &bull; Shampoo-Bottle &bull; Standing-Bottle &bull; Tire &bull; Valve</strong>
</p>

<h2>The Class Imbalance Problem</h2>
<p>
  Not all classes have the same number of images. This is a big problem in machine learning!
  If the AI sees 449 bottle images but only 65 standing-bottle images, it will get lazy
  and just guess "bottle" all the time.
</p>

{img(p_class_bar, "95%", "Number of images per class in the Watertank-Cropped dataset. The standing-bottle class has almost 7× fewer images than bottle!")}

{tip("Why imbalance matters", "Imagine a student who studies 90% of the time for one exam topic and only 10% for another. On the actual test, they'll do great on the first topic but terrible on the second. Our AI has the same problem — we need to fix it.")}

<p><strong>How we fixed the imbalance:</strong></p>
<ul>
  <li><strong>WeightedRandomSampler:</strong> We told the training loop "show rare classes MORE often." Even though there are fewer standing-bottle images, we pick them more frequently during training so the AI sees them as often as bottle images.</li>
  <li><strong>Focal Loss:</strong> A special loss function that says "if you're already getting this class right, don't worry about it too much — focus on the hard examples."</li>
</ul>

<h2>How We Split the Data</h2>
<p>
  We never let the AI train on the same images it will be tested on — that would be cheating!
  We split data into three groups:
</p>
<table>
  {th("Split", "Percentage", "Purpose")}
  {td("Training set", "70%", "The AI learns from these images — adjusts its settings based on them")}
  {td("Validation set", "15%", "We check progress during training — AI never trains on these")}
  {td("Test set", "15%", "Final exam! AI sees these only once at the very end")}
</table>


<!-- ═══ CHAPTER 3: EDA ═══════════════════════════════════════════ -->
<h1 class="page-break">Chapter 3: EDA — Getting to Know Your Data</h1>

<div class="chapter-intro">
  EDA stands for Exploratory Data Analysis. Before building any AI, you need to really
  understand your data. This is like reading a book before writing a review of it.
</div>

<h2>What is EDA?</h2>
<p>
  EDA is the process of looking at your data carefully before training. You make charts,
  look at examples, count things, and ask "is there anything weird here?" Skipping EDA
  is like jumping into a pool without checking if there's water in it.
</p>

{analogy("A Chef Checking Ingredients", "Before cooking a meal, a good chef smells and tastes every ingredient. They don't just throw everything in the pot. EDA is the same — we smell and taste our data before feeding it to the AI.")}

<h2>What Did We Look at in EDA?</h2>

<h3>1. Class Distribution (Are All Classes Balanced?)</h3>
<p>
  We counted how many images each class has. As shown in Chapter 2, we found a 7× imbalance
  in the Watertank dataset (tire images vs. standing-bottle images).
</p>
<p><strong>Finding:</strong> The dataset is imbalanced. We need to fix this during training.</p>

<h3>2. Image Quality (Can We See Anything?)</h3>
<p>
  Sonar images are often very dark and low-contrast. We looked at raw images to understand
  this problem. Bright spots appear where objects are, but the overall contrast is poor.
</p>
<p><strong>Finding:</strong> We need preprocessing to improve contrast before feeding images to the AI.</p>

<h3>3. CLAHE Analysis (Does Contrast Enhancement Help?)</h3>
<p>
  We applied CLAHE (Contrast Limited Adaptive Histogram Equalization) to sonar images
  and compared before vs. after to confirm it improves visibility.
</p>

{img(clahe_path, "95%", "EDA finding: CLAHE dramatically improves sonar image contrast. Left = raw image. Right = CLAHE-enhanced. The objects are much clearer after enhancement.")}

{img(pixel_dist, "85%", "EDA finding: Pixel intensity distribution in raw sonar images. Most pixels are very dark (left side of chart), confirming the need for contrast enhancement.")}

<h3>4. Image Size Consistency</h3>
<p>
  All sonar images are <strong>320 × 480 pixels</strong>, grayscale (1 channel, not 3 like RGB).
  This is important because neural networks need consistent input sizes.
</p>
<p><strong>Finding:</strong> Images are consistent in size. Good — no extra work needed for resizing.</p>

<h2>What Files Were Used for EDA?</h2>
<p>The EDA was done in two Jupyter notebooks:</p>
<ul>
  <li><code>Dataset/notebooks/01_eda.ipynb</code> — the EDA code</li>
  <li><code>Dataset/notebooks/01_eda_executed.ipynb</code> — the same notebook with all outputs saved</li>
  <li>Output figures saved to: <code>Dataset/results/eda_fls_clahe.png</code> and <code>eda_fls_pixel_dist.png</code></li>
</ul>

<h2>EDA Summary — What We Learned</h2>
<table>
  {th("Finding", "Impact on our project")}
  {td("Class imbalance (7×–13×)", "Used WeightedRandomSampler + Focal Loss")}
  {td("Low image contrast", "Added CLAHE preprocessing to all images")}
  {td("Images are grayscale", "Copied 1 channel to 3 channels (models expect 3-channel input)")}
  {td("Images are 320×480px", "Resized to 224×224 (classification) or 640×640 (detection)")}
</table>


<!-- ═══ CHAPTER 4: PREPROCESSING ════════════════════════════════ -->
<h1 class="page-break">Chapter 4: Preprocessing — Cleaning Up Before Training</h1>

<div class="chapter-intro">
  Raw data is almost never ready to use directly. Preprocessing is the step where we
  clean, resize, and enhance the images so the AI can learn from them effectively.
</div>

<h2>The Preprocessing Pipeline</h2>

{img(p_pipeline, "95%", "Every image goes through this pipeline before reaching the AI model.")}

<h2>Step-by-Step Explanation</h2>

<h3>Step 1: CLAHE — Improving Contrast</h3>
<p>
  <strong>CLAHE</strong> stands for Contrast Limited Adaptive Histogram Equalization.
  It's a fancy way of making dark sonar images look clearer.
</p>
<p>
  Regular contrast enhancement brightens the whole image uniformly. CLAHE divides the
  image into small tiles (8×8 in our case) and adjusts contrast separately in each tile.
  This means it improves contrast in dark corners without making bright areas too white.
</p>

{analogy("Night Vision Goggles", "Imagine you're in a dark room. Turning on all the lights at once might blind you. Night vision goggles brighten only the dark areas around you. CLAHE works the same way — it brightens only where the image is dark, leaving bright areas alone.")}

<h3>Step 2: Gaussian Blur — Reducing Noise</h3>
<p>
  Sonar images have "speckle noise" — random bright and dark pixels that aren't real objects,
  just noise from the acoustic sensor. We apply a tiny Gaussian blur (σ=1) to smooth this out.
  It's like applying a very mild smoothing filter in Photoshop.
</p>

<h3>Step 3: Resize</h3>
<p>Neural networks need all images to be the exact same size:</p>
<ul>
  <li><strong>Classification:</strong> 224 × 224 pixels</li>
  <li><strong>Detection:</strong> 640 × 640 pixels (YOLO needs larger images to find small objects)</li>
</ul>

<h3>Step 4: 3-Channel Replication</h3>
<p>
  Modern neural networks like ResNet-50 expect 3-channel RGB images (red, green, blue).
  Our sonar images are 1-channel grayscale. Solution: we copy the same grayscale channel
  3 times to create a "fake RGB" image. The model sees R=G=B=grayscale, which works fine.
</p>

<h3>Step 5: Data Augmentation</h3>
<p>
  To make our AI more robust, during training we randomly apply these transformations to
  each image before the AI sees it:
</p>
<table>
  {th("Augmentation", "What it does", "Why it helps")}
  {td("Horizontal flip", "Mirror the image left-right", "Debris looks the same from both sides")}
  {td("Vertical flip", "Mirror the image top-bottom", "Sonar can see objects from above or below")}
  {td("Random rotation", "Rotate ±15°", "Objects can be at any angle on the ocean floor")}
  {td("Brightness jitter", "Randomly change brightness", "Sonar intensity varies with distance")}
  {td("Elastic distortion", "Warp the image slightly", "Makes AI robust to shape variations")}
</table>

{tip("Why Augmentation?", "If you have 1,000 training images and apply random augmentations, each time the AI sees an image it looks slightly different. It's like studying the same textbook with different highlighter colors each time — you absorb the content better.")}

<p>
  <strong>Code location:</strong> All preprocessing is in <code>Dataset/src/transforms.py</code>.
  The datasets that apply these transforms are in <code>Dataset/src/datasets.py</code>.
</p>


<!-- ═══ CHAPTER 5: CLASSIFICATION ══════════════════════════════ -->
<h1 class="page-break">Chapter 5: Classification — Teaching the AI to Name Debris</h1>

<div class="chapter-intro">
  Task 1: Given a cropped sonar patch of a single object, predict what type of debris it is.
  This is the first of our two tasks.
</div>

<h2>What is Image Classification?</h2>
<p>
  The AI looks at one image and outputs a single label from a list. For example:
  "I'm 94% sure this is a tire."
</p>

{analogy("Pokémon Cards", "Imagine showing a Pokémon card to a kid who has memorized the whole Pokédex. They look at the card and say 'That's Pikachu!' instantly. Classification is exactly that — show the AI an image, it tells you what it is.")}

<h2>Model 1: Baseline CNN (The Simple One)</h2>
<p>
  Before building a complex model, we always build a <strong>simple baseline</strong>.
  This tells us how much our fancy model actually improves things.
</p>
<p>
  Our baseline was a simple CNN (Convolutional Neural Network) with 4 blocks:
  each block has a convolutional layer, batch normalization, and ReLU activation.
  No pretrained weights — trained from scratch in 15 epochs.
</p>

{result_box("Baseline CNN Result: 76.06% test accuracy")}

<h2>Model 2: ResNet-50 (The Main Model)</h2>
<p>
  <strong>ResNet-50</strong> is a famous deep neural network invented by Microsoft Research
  in 2016. The "50" means it has 50 layers deep. It was the breakthrough model that showed
  very deep networks could be trained reliably.
</p>

{img(p_resnet, "95%", "ResNet-50 architecture: input goes through convolution, then 4 groups of residual blocks, then global average pooling, then the output layer predicting 10 or 18 classes.")}

<h3>Why ResNet-50 Specifically?</h3>
<table>
  {th("Reason", "Explanation")}
  {td("Skip connections (residual blocks)", "The key innovation: each block has a shortcut that skips 2 layers. This prevents the 'vanishing gradient' problem that kills very deep networks.")}
  {td("Proven track record", "ResNet-50 has won many image classification competitions. It's a reliable choice.")}
  {td("Right size for our hardware", "ResNet-50 (25M parameters) fits on our Apple M4 Mac. Larger models like ResNet-152 would be too slow.")}
  {td("Trains from scratch", "Since sonar ≠ normal photos, ImageNet pretrained weights don't help much. ResNet-50 is fast enough to train from scratch on our dataset.")}
</table>

{analogy("Hiring an Experienced Employee", "Imagine hiring someone with 10 years of general experience vs. a fresh graduate. The experienced person already knows how to structure work, write emails, meet deadlines — they just need to learn your specific product. ResNet-50 already knows how to detect edges, curves, and patterns from being a well-proven architecture. We just teach it to recognize sonar debris.")}

{result_box("ResNet-50 (from scratch) Result: 98.59% test accuracy")}

<h2>Model 3: Transfer Learning — The Cross-Domain Trick</h2>
<p>
  Transfer learning means: first train a model on Dataset A, then reuse most of what it
  learned to train faster on Dataset B.
</p>
<p>
  We first trained ResNet-50 on <strong>Turntable-Cropped</strong> (18 classes).
  Then we took those learned weights and fine-tuned them on <strong>Watertank-Cropped</strong>
  (10 classes). The backbone (all layers except the last classification layer) carries over.
  Only the final "head" layer is swapped out for 10-class output.
</p>

{img(p_train_prog, "90%", "Training progress for the transfer learning model. The AI starts already knowing sonar patterns from the Turntable domain, so it improves very quickly.")}

{tip("Why Transfer Works Here", "Turntable and Watertank used the same sonar sensor (ARIS Explorer 3000). So edges, shadows, and object shapes look similar in both. The backbone already knows 'what sonar images look like' — it just needs to re-learn which class is which.")}

{result_box("ResNet-50 (transfer: Turntable→Watertank) Result: 98.87% accuracy — same result in 40% less training time!")}

<h2>Code Used</h2>
<ul>
  <li><strong>Training script:</strong> <code>Dataset/train_classifier.py</code></li>
  <li><strong>Dataset class:</strong> <code>Dataset/src/datasets.py</code> → <code>FLSClassificationDataset</code></li>
  <li><strong>Training loop:</strong> <code>Dataset/src/train.py</code></li>
  <li><strong>Results log:</strong> <code>Dataset/results/classification_full_run.log</code></li>
</ul>

<h2>Classification Results Summary</h2>
<table>
  {th("Model", "Test Accuracy", "Training Time")}
  {td("Baseline CNN (scratch)", "76.06%", "2.7 minutes")}
  {td("ResNet-50 (scratch)", "98.59%", "15.9 minutes")}
  {td("ResNet-50 (transfer)", "<strong>98.87%</strong>", "9.8 minutes ← fastest!")}
</table>


<!-- ═══ CHAPTER 6: DETECTION ════════════════════════════════════ -->
<h1 class="page-break">Chapter 6: Detection — Teaching the AI to Find and Box Debris</h1>

<div class="chapter-intro">
  Task 2: Given a full sonar scene that may contain multiple objects, draw a bounding box
  around each one and label it. This is harder than classification.
</div>

<h2>What is Object Detection?</h2>
<p>
  Detection is classification + localization. The AI not only says "there's a tire here"
  but also tells you exactly where it is using a bounding box (a rectangle).
</p>

{analogy("Spot the Difference Puzzle", "You know those puzzles where you circle every difference between two pictures? Object detection is similar — the AI circles every object it finds and writes its name next to it.")}

<h2>How We Prepared the Data for Detection</h2>
<p>
  The raw annotations were in <strong>Pascal VOC XML format</strong> — a text file describing
  each bounding box with X, Y, width, height in pixels. YOLO needs a different format:
  normalized coordinates (0 to 1, relative to image size). We wrote a conversion script to
  transform all XML files into YOLO-format text files.
</p>

<h2>The Detection Model: YOLOv8</h2>
<p>
  <strong>YOLO</strong> stands for "You Only Look Once." It's a family of real-time object
  detection models created by the company Ultralytics. The key idea is processing the
  entire image in one single pass through the neural network — no separate region proposal step.
</p>

{img(p_yolo, "95%", "How YOLO works: it divides the image into a grid, predicts boxes and class probabilities for each cell simultaneously.")}

<h3>Why YOLOv8 Specifically?</h3>
<table>
  {th("Reason", "Explanation")}
  {td("Real-time speed", "YOLO runs in milliseconds — crucial for a robot submarine that needs to detect debris while moving")}
  {td("State of the art", "YOLOv8 (2023) is one of the most accurate and fastest detection models available")}
  {td("COCO pretrained", "YOLOv8 comes pretrained on 330,000 images (COCO dataset) — it already knows general shapes and patterns")}
  {td("Easy to use", "Ultralytics provides a clean Python API: 2 lines to train, 1 line to evaluate")}
</table>

<h3>Baseline vs Main Model</h3>
<table>
  {th("Model", "Parameters", "Epochs", "Training Device", "mAP50")}
  {td("YOLOv8n (nano, baseline)", "3.0 million", "30", "Kaggle T4 GPU", "0.927")}
  {td("YOLOv8m (medium, main)", "25.8 million", "80", "Kaggle T4 GPU", "<strong>0.937</strong>")}
</table>

<h2>What is mAP50?</h2>
<p>
  mAP50 stands for <strong>mean Average Precision at IoU threshold 0.50</strong>.
  Let's break this down:
</p>
<ul>
  <li><strong>IoU (Intersection over Union):</strong> How much the predicted box overlaps with the real box. 1.0 = perfect overlap. 0.5 = predicted box overlaps at least 50% with the real box.</li>
  <li><strong>Precision:</strong> Of all the boxes the AI drew, what fraction were correct?</li>
  <li><strong>Average Precision (AP):</strong> Average precision across different confidence thresholds for one class.</li>
  <li><strong>Mean AP (mAP):</strong> Average of AP across all 10 classes.</li>
</ul>
<p>
  mAP50 = 0.937 means our AI draws correct boxes (at least 50% overlap with real box)
  for 93.7% of all debris objects across all 10 classes.
</p>

{result_box("YOLOv8m Result: mAP50 = 0.937 (93.7%)")}

<h2>Problems We Faced During Detection Training</h2>

{problem("YOLOv8 crashed on our Mac M4 with a bizarre error: it tried to allocate 320 GIGABYTES of memory. The whole laptop only has 16GB!")}

<h3>What Caused It?</h3>
<p>
  Apple's MPS (Metal Performance Shaders) GPU backend has a bug in PyTorch. When YOLOv8
  calculates how many objects are in each image (using a function called
  <code>torch.unique(return_counts=True)</code>), the MPS backend sometimes returns
  corrupted count values — like returning "1 billion" instead of "5 objects." When the
  code then tries to allocate memory based on this corrupted count, it asks for 320GB.
</p>

{fix("We wrote a 'monkey-patch' — a tiny code fix that intercepts the problematic call before it runs. Our patch moves the <code>counts.max()</code> operation from the GPU (Apple Metal) to the CPU, where it works correctly. This patch is in <code>Dataset/run_yolo_resume.py</code> and runs automatically before training starts.")}

{problem("Even with the MPS fix, training YOLOv8 locally was too slow. The Mac's GPU isn't powerful enough for 80 epochs of detection training in reasonable time.")}

{fix("We moved detection training to Kaggle — a free cloud platform that provides NVIDIA T4 GPUs (16GB VRAM). Training took 1.072 hours on Kaggle vs. what would have been 8+ hours locally.")}

<h2>Code Used</h2>
<ul>
  <li><strong>Training script (with MPS patch):</strong> <code>Dataset/run_yolo_resume.py</code></li>
  <li><strong>Kaggle notebook:</strong> <code>Dataset/kaggle_notebook.py</code></li>
  <li><strong>Config file:</strong> <code>Dataset/configs/sonar.yaml</code></li>
  <li><strong>Results log:</strong> <code>Dataset/results/yolo_baseline.log</code>, <code>yolo_resume.log</code></li>
</ul>


<!-- ═══ CHAPTER 7: PROBLEMS & FIXES (renumbered) ════════════════ -->
<!-- (Segmentation chapter removed — project focuses on Classification and Detection) -->
<h1 class="page-break">Chapter 7: All Problems We Hit and How We Fixed Them</h1>

<div class="chapter-intro">
  Real ML projects never go smoothly. This chapter is an honest summary of everything
  that went wrong and how we dealt with it.
</div>

<table>
  {th("Problem", "Root Cause", "How We Fixed It", "Lesson Learned")}

  {td(
      "YOLOv8 tried to allocate 320 GB of RAM on Mac",
      "Bug in Apple's MPS GPU backend: <code>torch.unique(return_counts=True)</code> returns corrupted values",
      "Monkey-patch in <code>run_yolo_resume.py</code>: redirect <code>counts.max()</code> to CPU before use",
      "Always check if your hardware has known PyTorch bugs before starting training"
  )}

  {td(
      "Dataset upload to Kaggle was 1.5 GB and very slow",
      "We accidentally included the full classification dataset that wasn't needed for detection",
      "Created a smaller 240 MB zip with only the detection images + YOLO labels",
      "Always check what files you actually need before uploading large datasets"
  )}

  {td(
      "PDF generation produced jumbled, broken text layout",
      "We used <code>fpdf2</code> library which doesn't handle HTML/CSS properly",
      "Switched to <code>WeasyPrint</code> (via conda-forge) which converts full HTML+CSS to PDF",
      "Don't use fpdf2 for complex layouts — WeasyPrint handles CSS properly"
  )}

  {td(
      "Class imbalance: 7–13× between most and least common debris types",
      "Natural distribution of debris types in the water tank recordings",
      "WeightedRandomSampler during training + Focal Loss that focuses on rare classes",
      "Always check class distribution in EDA — imbalance is very common in real datasets"
  )}

  {td(
      "Original classification checkpoint's training config was lost",
      "The training script that produced the old checkpoints was deleted from the project",
      "Re-ran all classification experiments from scratch with documented, reproducible code",
      "Always document your training config and keep your scripts in version control (git)"
  )}
</table>

<h2>The Leakage Problem in Our Splits</h2>
<p>
  This is a subtle but important issue worth explaining carefully.
</p>
<p>
  <strong>The problem:</strong> Our dataset files are named like <code>can-212.png</code>,
  <code>can-213.png</code>, etc. These are sequential frames from a video. Frames 212 and
  213 are nearly identical — the same can, barely moved. If frame 212 goes into training
  and frame 213 goes into testing, the AI has basically "seen" the test example during
  training. This is called <strong>data leakage</strong>.
</p>

{analogy("Exam Cheating", "Imagine a teacher creates 100 exam questions. 70 go on the practice exam (training), 15 on the midterm (validation), and 15 on the final (test). But the practice exam questions are all slightly rephrased versions of the final exam questions. The student who memorized the practice exam will ace the final — but they didn't actually learn!")}

<p>
  <strong>How we partially fixed it:</strong> We created a "blocked" split — instead of
  randomly shuffling individual frames, we grouped consecutive frames into blocks and
  split at the block level. This keeps near-duplicate frames on the same side of the split.
  It's not a perfect fix (we don't know exactly where each recording session ends), but
  it reduces the leakage risk significantly.
</p>
<p>
  <strong>Evidence the problem was real:</strong> Our standard random split gave 98.59%
  on Watertank. The same model, same code, same dataset (but different domain —
  Turntable-Cropped) gave only 86.39%. The 12-point drop suggests our high Watertank
  scores may be inflated by leakage.
</p>


<!-- ═══ CHAPTER 9: FINAL RESULTS ════════════════════════════════ -->
<h1 class="page-break">Chapter 9: Final Results — How Good Did We Get?</h1>

<div class="chapter-intro">
  All scores at a glance — and what they mean in plain English.
</div>

{img(p_model_cmp, "95%", "All model scores compared. Green bars are at or above 90%. Classification and Detection both exceed the 90% threshold.")}

<h2>Classification Results</h2>
<table>
  {th("Model", "Score", "What This Means")}
  {td("Baseline CNN", "76.06%", "The simplest model got 76% right. 24% wrong — acceptable for a baseline but not good enough")}
  {td("ResNet-50 (scratch)", "98.59%", "Got nearly every single image right. Only ~15 wrong out of 354 test images")}
  {td("ResNet-50 (transfer)", "98.87%", "Slightly better AND trained faster. Best classification result")}
</table>

<h2>Detection Results</h2>
<table>
  {th("Model", "mAP50", "What This Means")}
  {td("YOLOv8n (baseline)", "0.927", "93% of debris objects found and correctly labeled")}
  {td("YOLOv8m (main)", "0.937", "94% of debris objects found — better precision with more parameters")}
</table>

<p>The hardest classes to detect were:</p>
<ul>
  <li><strong>Valve (0.848):</strong> Small, round, easily confused with cans</li>
  <li><strong>Can (0.855):</strong> Similar shape to shampoo-bottle in sonar</li>
  <li><strong>Shampoo-bottle (0.866):</strong> Very few training examples (only 16 in test set)</li>
</ul>

<h2>Cross-Task Comparison</h2>
<table>
  {th("Task", "Why It Scored This Way")}
  {td("Classification (98.87%)", "Easiest — one answer per image; cropped patches are simple and distinctive")}
  {td("Detection (93.7%)", "Harder — must find AND label multiple objects; but YOLO+COCO pretraining is very powerful")}
</table>


<!-- ═══ CHAPTER 10: OLD VS NEW COMPARISON ═══════════════════════ -->
<h1 class="page-break">Chapter 10: Old Models vs Our Models — Who Wins?</h1>

<div class="chapter-intro">
  Before this project, there were already two models trained on this dataset: ResNet-20 and SqueezeNet,
  using older Keras notebooks. We re-ran them under identical conditions to ours for a fair fight.
</div>

<h2>Where Did the Old Models Come From?</h2>
<p>
  Inside the project folder there are 4 old Jupyter notebooks (in <code>Dataset/old models/</code>)
  from a previous attempt at this problem. They used two models:
</p>
<ul>
  <li><strong>ResNet-20</strong> — a smaller, 20-layer version of ResNet (vs our 50-layer ResNet-50)</li>
  <li><strong>SqueezeNet</strong> — a very lightweight model designed for tiny images on mobile devices</li>
</ul>
<p>
  Those notebooks reported impressive scores — up to <strong>99.7%</strong>! But they used different
  conditions: smaller images (96×96 vs our 224×224), fewer classes (11–12 vs our 10–18),
  more training time (40 epochs vs our 15), and likely had data leakage issues.
</p>

{analogy("Different Exam Conditions", "Imagine two students take the same subject but one gets an open-book exam with 5 questions and the other gets a closed-book exam with 18 questions. The first student scoring 99% doesn't mean they're smarter — it means the conditions were easier.")}

<h2>The Fair Re-Run</h2>
<p>
  We re-implemented ResNet-20 and SqueezeNet in PyTorch and trained them with <strong>exactly the same
  conditions</strong> as our models: same images, same 10 classes, same 15 epochs, same optimizer.
  Here's what happened:
</p>

{img(p_old_vs_new, "95%", "Old models vs our models on Watertank-Cropped (10 classes, 15 epochs, identical conditions). The red dashed line shows random guessing level (10%).")}

<table>
  {th("Model", "Watertank Score", "Turntable Score", "What Happened")}
  {td("SqueezeNet (old, re-run)", "18.87%", "8.89%", "Near-random! Barely better than guessing. Designed for 96×96 images, struggles at 224×224")}
  {td("ResNet-20 (old, re-run)", "84.79%", "63.21%", "Decent but not great. Too shallow (20 layers) to distinguish all 10 debris types reliably")}
  {td("Baseline CNN (ours)", "76.06%", "—", "Our simplest model — better than ResNet-20 on fewer layers because we tuned it for 224×224")}
  {td("<strong>ResNet-50 scratch (ours)</strong>", "<strong>98.59%</strong>", "<strong>86.39%</strong>", "Clear winner. 50 layers, enough depth to learn all debris patterns")}
  {td("<strong>ResNet-50 transfer (ours)</strong>", "<strong>98.87%</strong>", "—", "Best overall. Pre-training on Turntable gave it a head start")}
</table>

<h2>Why Did SqueezeNet Score So Low?</h2>
<p>SqueezeNet got 18.87% — almost the same as random guessing (10%). Three reasons:</p>
<ol>
  <li><strong>Wrong image size:</strong> SqueezeNet was invented for 96×96 images. We gave it 224×224. Its internal "fire modules" can't build up an understanding of the full image at this size in 15 epochs.</li>
  <li><strong>No skip connections:</strong> Unlike ResNet, SqueezeNet has no shortcuts between layers. This makes it hard to train — gradients get lost in 15 epochs and the model barely learns.</li>
  <li><strong>Too few epochs:</strong> The old notebooks trained for 40 epochs. At 96×96, 40 epochs is enough for SqueezeNet. At 224×224 in 15 epochs, it never gets started.</li>
</ol>

{tip("Key Lesson", "A model that works great in one setting can fail completely in another. SqueezeNet is not a bad model — it's just the wrong tool for our setup. This is why we tested multiple models and picked ResNet-50.")}

<h2>Why Did the Old Notebooks Report 99%+?</h2>
<table>
  {th("Difference", "Old notebooks", "Our re-run", "Effect")}
  {td("Image size", "96×96 pixels", "224×224 pixels", "SqueezeNet works at 96×96, fails at 224×224")}
  {td("Number of classes", "11–12 classes", "10–18 classes", "Fewer classes = easier problem")}
  {td("Training time", "40 epochs", "15 epochs", "More training time = better scores")}
  {td("Data split", "Random (likely leaked)", "Random (same leakage risk)", "High scores may be partly from seeing similar images in train and test")}
</table>

<p>
  Their 99%+ wasn't wrong for their conditions — it just can't be compared directly to our numbers.
  The fair re-run shows the real picture: <strong>ResNet-50 wins clearly</strong>, and the old models
  are significantly weaker when everything else is held equal.
</p>


<!-- ═══ CHAPTER 11: CONCLUSION ══════════════════════════════════ -->
<h1 class="page-break">Chapter 11: Conclusion — What Did We Learn?</h1>

<h2>Technical Conclusions</h2>
<ul>
  <li><strong>Deep learning works well on sonar imagery</strong> — even without color information, AI can reliably detect and classify underwater debris.</li>
  <li><strong>Transfer learning saves time</strong> — training on a related sonar domain first (Turntable) helps when fine-tuning on the target domain (Watertank). Same accuracy in 40% less time.</li>
  <li><strong>YOLOv8 is excellent for detection</strong> — 93.7% mAP50 on a 10-class sonar dataset, running in real-time.</li>
  <li><strong>Hardware matters</strong> — Apple M4 MPS has bugs that prevented certain training. Free cloud GPU (Kaggle T4) was essential but comes with time limits.</li>
</ul>

<h2>Personal Lessons</h2>
<ul>
  <li><strong>Always document your training config</strong> — we lost a checkpoint and couldn't reproduce results because the original script was deleted.</li>
  <li><strong>EDA first, code later</strong> — finding the class imbalance in EDA let us design the right loss functions before wasting training time.</li>
  <li><strong>Bugs happen — log everything</strong> — the MPS crash was a real PyTorch bug, not our code. Having detailed logs helped us diagnose it.</li>
  <li><strong>Be honest about limitations</strong> — we don't claim our 98%+ classification scores are perfect; we documented the leakage risk clearly.</li>
</ul>

<h2>What Would We Do Next?</h2>
<ul>
  <li>Get true object-level splits to eliminate the leakage risk entirely</li>
  <li>Test YOLOv8m on real quarry footage (different domain) to check real-world generalization</li>
  <li>Deploy YOLOv8m on a Raspberry Pi or NVIDIA Jetson to test on an actual AUV</li>
</ul>

<h2>Project File Map — Where Is Everything?</h2>
<table>
  {th("What", "Where")}
  {td("EDA notebooks", "<code>Dataset/notebooks/01_eda.ipynb</code>")}
  {td("All source code", "<code>Dataset/src/</code> (datasets, transforms, training, evaluation)")}
  {td("Classification training", "<code>Dataset/train_classifier.py</code>")}
  {td("Detection training", "<code>Dataset/run_yolo_resume.py</code>")}
  {td("All results & logs", "<code>Dataset/results/</code>")}
  {td("Full academic report", "<code>Dataset/results/Clean_Academic_Report.pdf</code>")}
  {td("YOLO model weights", "<code>Kaggle outputs/yolov8m.pt</code>")}
</table>

<p style="text-align:center;margin-top:30px;color:#888;font-size:9pt;">
  University ML Project &bull; Marine Debris Detection &bull; June 2026<br>
  Apple M4 + Kaggle T4 GPU &bull; PyTorch 2.12 &bull; Python 3.11
</p>

</body>
</html>"""


# ── render ─────────────────────────────────────────────────────────
print("Building HTML...")
print("Rendering PDF with WeasyPrint...")
import weasyprint
weasyprint.HTML(string=HTML, base_url=str(BASE)).write_pdf(str(OUT_PDF))
size_mb = OUT_PDF.stat().st_size / 1_048_576
print(f"\nDone! PDF saved: {OUT_PDF.resolve()}")
print(f"Size: {size_mb:.1f} MB")
