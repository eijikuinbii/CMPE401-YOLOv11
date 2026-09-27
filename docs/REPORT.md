# CMPE 401 Project 1 — Technical Report

> Fill this in as you run experiments. It maps 1:1 to the required Parts I–V in
> the project spec and is the narrative the rubric grades. Paste figures from
> `results/figures/` and tables from `results/tables/`.

## 0. Overview
- **Team:** <name(s)>
- **Primary model:** YOLOv11
- **Dataset:** VisDrone-DET (10 classes)
- **Compute:** RTX 3060 laptop (6 GB) + Google Colab (T4/A100)
- **Repo:** <github url>

---

## Part I — Baseline Model
**Settings:** model=`yolo11s`, imgsz=640, batch=8, epochs=100, optimizer=auto.

| Metric | Value |
|---|---|
| mAP@50-95 | |
| mAP@50 | |
| Precision | |
| Recall | |

Training/validation loss curves: `results/figures/loss_baseline_s_640.png`

---

## Part II — Loss Curve & Fitting Analysis
- **Convergence behavior:** <describe where loss plateaus>
- **Overfitting?** <train keeps dropping while val rises / gap widens?>
- **Underfitting?** <both losses high / model too small for the data?>
- **Causes — reference dataset size:** VisDrone-DET train ≈ 6,471 images, very
  dense small objects. <how does this drive your diagnosis?>
- **Causes — reference model capacity:** <n/s/m capacity vs task difficulty>

---

## Part III — Structured Experimental Design (≥1 round)
State settings → report quantitative results → analyze. Suggested variables:
model size (n/s/m), image resolution, augmentation, LR schedule, batch, epochs.

| Run | Model | imgsz | batch | epochs | key change | mAP@50-95 | mAP@50 | P | R |
|---|---|---|---|---|---|---|---|---|---|
| baseline | yolo11s | 640 | 8 | 100 | — | | | | |
| exp1 | | | | | | | | | |

**Analysis:** <what changed and why>

---

## Part IV — Iterative Model Improvement (≥1 cycle)
Follow: Baseline → Experimental Settings → Controlled Modification → Evaluation
→ Analysis → Conclusion. Examples: regularization, normalization, weight init,
LR schedule, overfitting control.

- **Hypothesis / principle:** <e.g. mosaic+higher weight_decay to curb overfit>
- **Controlled modification:** <exact single change>

| Run | Change | mAP@50-95 | mAP@50 | P | R | Δ vs baseline |
|---|---|---|---|---|---|---|
| baseline | — | | | | | — |
| improved | | | | | | |

**Justification & conclusion:** <did it work, and why in DL terms>

---

## Part V — Multi-Version YOLO Comparison (OPTIONAL)
Compare YOLOv11 against ≥1 other version (v8/v9/v10) under a matched recipe.

| Model | mAP@50-95 | mAP@50 | Params (M) | Train time | Inference (ms/img) |
|---|---|---|---|---|---|
| yolo11s | | | | | |
| yolov8s | | | | | |

**Discussion:** <accuracy vs speed vs size trade-offs; confusion matrices>

---

## Reproducibility
Exact commands to reproduce every run above are in the README. Seeds fixed via
`--seed`. Environment: `requirements.txt`.
