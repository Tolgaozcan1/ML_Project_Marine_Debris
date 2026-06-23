"""
Project Update Report — documents the compliance remediation pass done against
the course "Machine Learning Project Instructions" (audit, fixes, new results).
HTML + CSS -> WeasyPrint (see CLAUDE.md: fpdf2 breaks layout, use WeasyPrint).

Run: conda activate marine-debris && python create_update_report_pdf.py
Output: results/Project_Update_Report.pdf
"""
import base64
from pathlib import Path

BASE = Path(__file__).parent
FIGURES = BASE / "results" / "figures"
OUT_PDF = BASE / "results" / "Project_Update_Report.pdf"


def _png_b64(path):
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def fig(path, caption, width="100%"):
    p = Path(path)
    if not p.exists():
        return f'<p class="missing">[Figure not found: {p.name}]</p>'
    return f'<figure><img src="{_png_b64(p)}" style="width:{width};max-width:100%;" alt="{caption}"><figcaption>{caption}</figcaption></figure>'


def th(*cols):
    return "<tr>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>"


def td(*cols):
    return "<tr>" + "".join(f"<td>{c}</td>" for c in cols) + "</tr>"


def callout(title, text):
    return f'<div class="callout"><div class="callout-title">{title}</div><p>{text}</p></div>'


CSS = """
@page {
    size: A4;
    margin: 2.5cm 2.5cm 2.5cm 2.5cm;
    @bottom-center { content: "Page " counter(page) " of " counter(pages); font-size: 9pt; color: #888; }
    @top-center { content: "Marine Debris Project — Update Report"; font-size: 8pt; color: #888; }
}
* { box-sizing: border-box; }
body { font-family: Georgia, "Times New Roman", serif; font-size: 11pt; color: #1a1a1a; line-height: 1.7; }
h1.chapter { font-size: 16pt; color: #1a3c64; border-left: 6px solid #1a3c64; padding: 6px 0 6px 14px; margin-top: 0; page-break-before: always; page-break-after: avoid; }
h1.chapter:first-of-type { page-break-before: avoid; }
h2 { font-size: 13pt; color: #1a3c64; border-bottom: 1px solid #ddd; padding-bottom: 4px; margin-top: 22px; page-break-after: avoid; }
h3 { font-size: 11.5pt; color: #333; margin-top: 16px; page-break-after: avoid; }
p { margin: 0 0 9px 0; text-align: justify; }
ul, ol { margin: 6px 0 10px 0; padding-left: 24px; }
li { margin-bottom: 5px; }
table { width: 100%; border-collapse: collapse; margin: 14px 0; font-size: 10pt; page-break-inside: avoid; }
th { background: #1a3c64; color: white; padding: 7px 10px; text-align: left; }
td { padding: 6px 10px; border: 1px solid #ccc; vertical-align: top; }
tr:nth-child(even) td { background: #f5f7fa; }
figure { text-align: center; margin: 18px 0; page-break-inside: avoid; }
figcaption { font-style: italic; font-size: 9pt; color: #555; margin-top: 7px; }
.callout { background: #eef4ff; border-left: 5px solid #1a3c64; padding: 12px 16px; margin: 14px 0; border-radius: 0 4px 4px 0; }
.callout-title { font-weight: bold; color: #1a3c64; font-size: 11pt; margin-bottom: 4px; }
.callout p { margin: 0; font-size: 10.5pt; }
.warn { background: #fff4e5; border-left: 5px solid #c0631a; padding: 12px 16px; margin: 14px 0; border-radius: 0 4px 4px 0; }
.warn-title { font-weight: bold; color: #c0631a; font-size: 11pt; margin-bottom: 4px; }
.warn p { margin: 0; font-size: 10.5pt; }
.result-good { color: #1e8c45; font-weight: bold; }
.result-warn { color: #c0631a; font-weight: bold; }
.cover { text-align: center; padding-top: 60px; page-break-after: always; }
.cover h1 { font-size: 26pt; color: #1a3c64; border: none; padding: 0; page-break-before: avoid; }
.cover .subtitle { font-size: 13pt; color: #555; margin: 10px 0 8px 0; }
.cover .tagline { font-size: 11pt; color: #888; font-style: italic; margin-bottom: 30px; }
.cover table { width: 80%; margin: 28px auto; }
.missing { color: #aaa; font-style: italic; font-size: 9pt; }
.done::before { content: "✓ "; color: #1e8c45; font-weight: bold; }
.todo::before { content: "☐ "; color: #c0631a; font-weight: bold; }
"""

html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><style>{CSS}</style></head>
<body>

<div class="cover">
<h1>Project Update Report</h1>
<div class="subtitle">Marine Debris Detection in Forward-Looking Sonar Imagery</div>
<div class="tagline">Compliance remediation pass against the Machine Learning Project Instructions (Summer 2026)</div>
<table>
{th("Field", "Detail")}
{td("Course", "Machine Learning — Summer 2026")}
{td("Group", "8")}
{td("Team members", "Tolga Ozcan (26576920), Venkatesh Ajay Vijaya Kumar (36379442), Deekshith Hunsur Shekar (62246101)")}
{td("Report date", "23 June 2026")}
{td("Deadline", "Sunday, 28 June 2026")}
</table>
</div>

<h1 class="chapter">1. Why This Update Was Needed</h1>
<p>
The project's existing results (classification, detection, segmentation) were strong, but an
audit against the course's "Machine Learning Project Instructions" found several gaps between
what had been built and what the instructions explicitly require: no baseline model for any
task, no single consolidated presentation notebook, no fixed random seeds, minimal qualitative
error analysis, and segmentation results that could not be independently re-verified. This report
documents what was found, what was fixed, and what remains open.
</p>

{callout("Scope of this pass", "Classification and detection were fully remediated (baselines, "
"transfer experiment, fixed seeds, real error analysis). Segmentation and the leakage-prone "
"random splits were intentionally left as documented limitations rather than retrained, given "
"the time remaining before the deadline.")}

<h1 class="chapter">2. Gaps Found in the Audit</h1>
<table>
{th("Gap", "Why it mattered")}
{td("No baseline model for any of the 3 tasks", "Instructions Section 4 require a justified baseline before the main model")}
{td("No single presentation notebook", "Instructions Section 7 require one consolidated .ipynb with mandated sections")}
{td("Random image-level splits, no leakage check", "Same physical object can appear across frames in this dataset")}
{td("No global random seeds in training scripts", "Reproducibility requirement (Section 8)")}
{td("Classification training script missing entirely", "Only checkpoints survived; the original code could not be re-run or explained")}
{td("Segmentation weights lost, training log incomplete", "unet_train.log stopped at epoch 4 of a claimed 60; mIoU numbers were undocumented beyond a log file")}
{td("02_optical.ipynb / 04_fusion.ipynb unexecuted", "Zero real results behind the multimodal/fusion claim")}
{td("No git/online backup", "Instructions Section 9 require a backup beyond one laptop")}
</table>

<h1 class="chapter">3. What Was Fixed</h1>

<h2>3.1 Classification — Baseline and Cross-Domain Transfer</h2>
<p>
A new training script (<code>train_classifier.py</code>) regenerates the missing classification
pipeline with fixed seeds and produces three fresh, independently verifiable models on
Watertank-Cropped: a simple from-scratch CNN baseline, a ResNet-50 trained from scratch, and a
ResNet-50 pretrained on Turntable-Cropped then fine-tuned on Watertank-Cropped — the cross-domain
transfer comparison the project brief calls for.
</p>
<table>
{th("Model", "Watertank Test Accuracy", "Training Time")}
{td("Baseline CNN (scratch)", "76.06%", "2.7 min")}
{td("ResNet-50 (scratch)", "98.59%", "15.9 min")}
{td('<span class="result-good">ResNet-50 (Turntable&rarr;Watertank transfer)</span>', '<span class="result-good">98.87%</span>', "9.8 min")}
</table>
<p>
The transfer model matches or slightly exceeds the from-scratch model while converging in roughly
40% less training time — a clean, defensible answer to the cross-domain transfer question.
</p>
{fig(FIGURES / "classification_baseline_vs_transfer.png", "Baseline CNN vs. scratch ResNet-50 vs. cross-domain transfer (live-recomputed test accuracy).")}

<h2>3.2 Detection — Baseline Added, Main-Model Bug Fixed</h2>
<p>
A new <code>train_yolo_baseline.py</code> fine-tunes YOLOv8n (nano) on the same dataset and
config as the existing YOLOv8m (medium) main model, for fewer epochs — a genuine "simple but
correct" baseline at a fixed seed.
</p>
<p>
While wiring up live evaluation in the notebook, a pre-existing bug was also found and fixed:
the visualization and evaluation scripts were pointing at a generic, never-fine-tuned
COCO-pretrained <code>yolov8m.pt</code> checkpoint instead of the actual fine-tuned sonar model
(<code>results/yolo_sonar/yolov8m_watertank/weights/best.pt</code>). This produced a near-zero
mAP the first time the corrected live-evaluation cell was run, which is how the bug surfaced.
</p>
<table>
{th("Model", "mAP50", "mAP50-95")}
{td("YOLOv8n (baseline)", "0.927", "0.680")}
{td('<span class="result-good">YOLOv8m (main model, corrected weights)</span>', '<span class="result-good">0.937</span>', '<span class="result-good">0.697</span>')}
</table>
{fig(FIGURES / "yolo_per_class_ap.png", "Per-class detection performance.")}

<h2>3.3 Real Error Analysis</h2>
<p>
Rather than a random sample of predictions, the notebook now specifically surfaces every
misclassified test example for the main classification model, and contrasts it with the
baseline's failure pattern.
</p>
<table>
{th("Model", "Error rate", "Most common confusion")}
{td("ResNet-50 (Turntable, main model)", "1.6% (12/742)", "glass-bottle &rarr; plastic-pipe")}
{td("Baseline CNN (Watertank)", "23.9% (85/355)", "chain &rarr; drink-carton")}
</table>
{fig(FIGURES / "cls_baseline_misclassified.png", "Baseline CNN misclassifications — concentrated on visually similar, lower-frequency classes.", width="85%")}

<h2>3.4 Consolidated Presentation Notebook</h2>
<p>
<code>Marine_Debris_Presentation.ipynb</code> is the single notebook the course instructions
require, with every mandated section (Title Slide, Problem Statement, Dataset, Methodology,
Selected Models, Evaluation Criterion, Results, Error Analysis and Limitations, Conclusion, Code
Demonstration). Classification and detection results are computed <em>live</em> inside the
notebook from saved checkpoints — nothing is hardcoded — and it executes top-to-bottom with zero
errors.
</p>

<h2>3.5 Local Backup</h2>
<p>
A git repository was initialized at the project root with the full codebase, checkpoints, figures,
and logs committed locally (no remote push performed in this pass).
</p>

<h1 class="chapter">4. What Remains Open (Documented Limitations)</h1>
<table>
{th("Limitation", "Why it was not fixed in this pass")}
{td("Random, leakage-risk train/val/test splits", "The released dataset has no session/object ID in its filenames, so a true leakage-safe split is not cheaply derivable; fixing it would require re-running all three tasks")}
{td("Segmentation weights not independently reproducible", "U-Net and SegFormer-B2 weights were lost when a Kaggle session expired; retraining was judged out of scope given the time remaining")}
{td("No segmentation baseline", "Same reasoning as above")}
{td("Optical/TrashCan multimodal fusion", "Was a stretch goal in the original project brainstorm, not part of the assigned core tasks; descoped to stay within the deadline")}
</table>

<h1 class="chapter">5. Outstanding Action Items</h1>
<table>
{th("Item", "Owner")}
{td('<span class="todo">Push the local git repo to GitHub (or Drive) for online backup</span>', "Tolga")}
{td('<span class="todo">Copy the project to Venkatesh and Deekshith laptops</span>', "Venkatesh / Deekshith")}
{td('<span class="todo">All 3 members rehearse explaining the full workflow, splits, baselines, and limitations</span>', "All")}
{td('<span class="todo">Bring charger and USB-C adapter; test the notebook opens on more than one laptop</span>', "All")}
{td('<span class="done">Title slide matriculation numbers</span>', "Done — Tolga 26576920, Venkatesh 36379442, Deekshith 62246101")}
</table>

</body>
</html>"""

print("Rendering PDF with WeasyPrint...")
import weasyprint
weasyprint.HTML(string=html, base_url=str(BASE)).write_pdf(str(OUT_PDF))
size_mb = OUT_PDF.stat().st_size / 1_048_576
print(f"\nPDF saved:  {OUT_PDF.resolve()}")
print(f"Size:       {size_mb:.1f} MB")
