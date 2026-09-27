# Advanced Object Detection & Comparative Study using YOLOv11 — VisDrone

**CMPE 401 Instructor-defined Project 1**

Design, optimization, and comparative evaluation of modern YOLO models for
real-world object detection on the **VisDrone-DET** aerial dataset. This repo
contains a complete, reproducible pipeline: data download & conversion, baseline
training, loss/fitting analysis, controlled experiments, an iterative
improvement cycle, and an optional multi-version YOLO comparison.

> **Status:** scaffold ready. Results tables/figures below are filled in as runs
> complete. Full write-up in [`docs/REPORT.md`](docs/REPORT.md).

---

## 1. Objective

Fine-tune a modern YOLO model (YOLOv11) on VisDrone-DET and study its training
dynamics rather than just its final score: diagnose overfitting/underfitting,
run controlled experiments, perform theory-driven improvements, and (optionally)
compare against other YOLO versions — all reproducibly.

## 2. Dataset — VisDrone-DET

Aerial images with dense, small objects across 10 classes: `pedestrian, people,
bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor`. Splits: train
(~1.44 GB), val (~0.07 GB), test-dev (~0.28 GB), test-challenge (~0.28 GB).
Source: <https://github.com/VisDrone/VisDrone-Dataset>.

Raw VisDrone annotations (`x,y,w,h,score,category,trunc,occ`) are converted to
normalized YOLO boxes; category ids `1..10` map to YOLO classes `0..9`, and
`ignored regions (0)` / `others (11)` are dropped. See
[`scripts/convert_visdrone.py`](scripts/convert_visdrone.py).

## 3. Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
# Install a CUDA torch build matching your GPU (RTX 3060 laptop -> cu121):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Verify the GPU is visible:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 4. Get the data

```bash
python scripts/download_visdrone.py --root datasets/VisDrone   # add --challenge for the competition split
python scripts/convert_visdrone.py --root datasets/VisDrone
python scripts/visualize_labels.py --split VisDrone2019-DET-train --n 6   # sanity check
```

## 5. Compute strategy (hybrid)

| Where | Use for | Typical config |
|---|---|---|
| **RTX 3060 laptop (6 GB)** | pipeline dev, baseline, light experiments | `yolo11s`, imgsz 640, batch 4–8, AMP |
| **Google Colab (T4/A100)** | large models, high-res, final/competition runs | `yolo11m`, imgsz 1280, batch 16 |

The same scripts run in both places. Colab workflow:
[`notebooks/colab_train.ipynb`](notebooks/colab_train.ipynb).

## 6. Reproduce the experiments

```bash
# Part I — baseline (local 3060)
python scripts/train.py --model yolo11s.pt --imgsz 640 --batch 8 --epochs 100 --name baseline_s_640

# Part II — loss / fitting analysis
python scripts/plot_curves.py results/runs/baseline_s_640

# Part III — controlled experiments (examples)
python scripts/train.py --model yolo11m.pt --imgsz 1280 --batch 16 --epochs 100 --name exp_m_1280        # resolution + capacity
python scripts/train.py --model yolo11s.pt --imgsz 640  --batch 8  --epochs 100 --name exp_s_cos --cos-lr # LR schedule

# Part IV — iterative improvement (example: overfitting control via augmentation/reg)
python scripts/train.py --model yolo11s.pt --imgsz 640 --batch 8 --epochs 100 --name exp_s_reg \
    --extra weight_decay=0.001 mosaic=1.0 close_mosaic=10

# Part V (optional) — multi-version comparison
python scripts/train.py --model yolov8s.pt --imgsz 640 --batch 8 --epochs 100 --name compare_v8s_640

# Evaluate any run on val / test-dev
python scripts/evaluate.py --weights results/runs/baseline_s_640/weights/best.pt --split test

# Build comparison table across runs
python scripts/summarize_runs.py results/runs/baseline_s_640 results/runs/exp_m_1280 --out results/tables/experiments.md
```

## 7. Repository layout

```
configs/visdrone.yaml        # dataset + class config for Ultralytics
scripts/
  download_visdrone.py       # fetch + extract VisDrone-DET
  convert_visdrone.py        # VisDrone annotations -> YOLO labels
  visualize_labels.py        # draw converted labels (sanity check)
  train.py                   # reproducible training wrapper (all experiments)
  evaluate.py                # val / test-dev metrics
  plot_curves.py             # train-vs-val loss + mAP curves (Part II)
  summarize_runs.py          # comparison tables (Parts III/IV/V)
notebooks/colab_train.ipynb  # heavy runs on Colab
results/                     # runs/ (gitignored), figures/, tables/
docs/REPORT.md               # graded technical write-up (Parts I–V)
```

## 8. Results (fill in)

### Baseline (Part I)
| Metric | Value |
|---|---|
| mAP@50-95 | _tbd_ |
| mAP@50 | _tbd_ |
| Precision | _tbd_ |
| Recall | _tbd_ |

### Experiments summary (Parts III–V)
_Auto-generated into `results/tables/experiments.md`._

## 9. Key findings
_Summarize once runs complete — see [`docs/REPORT.md`](docs/REPORT.md) for the full analysis._

## 10. Reproducibility notes
- All runs use `--seed` (default 0). Weights and `runs/` are gitignored; commit
  `results.csv`, figures, and tables, and document exact commands (above).
- Environment pinned in `requirements.txt`.
