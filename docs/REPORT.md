# CMPE 401 Project 1 — Technical Report

> Parts I, II and IV (cycle 1) are complete; Part III, Part IV cycle 2 and Part V
> are designed and queued in the Colab notebook. Each
> section maps 1:1 to the required Parts in the project spec. Figures come from
> `results/figures/`, tables from `results/tables/`.

## 0. Overview
- **Team:** <name(s)>
- **Primary model:** YOLOv11 (`yolo11s`, 9.4M parameters, 21.7 GFLOPs at 640)
- **Dataset:** VisDrone-DET (10 classes)
- **Compute:** Google Colab T4 (baseline) + RTX 3060 laptop, 6 GB (pipeline dev, light runs)
- **Repo:** <https://github.com/eijikuinbii/CMPE401-YOLOv11>

### Dataset at a glance
Statistics of the converted YOLO labels (ignored regions and "others" removed):

| Split | Images | Boxes | Boxes / image | Boxes < 32×32 px at 640 input |
|---|---|---|---|---|
| train | 6,471 | 343,204 | 53.0 | 85.3% |
| val | 548 | 38,759 | 70.7 | 85.9% |

The median box covers only ~0.05% of the image (about 14×14 px at 640 input).
Classes are heavily imbalanced in train: car 144,866 and pedestrian 79,337 boxes
versus awning-tricycle 3,246, tricycle 4,812 and bus 5,926.

---

## Part I — Baseline Model
**Settings:** model=`yolo11s` (COCO-pretrained), imgsz=640, batch=16, epochs=100,
optimizer=auto (Ultralytics selected MuSGD, lr0=0.01, lrf=0.01, linear decay),
warmup=3 epochs, mosaic on with `close_mosaic=10`, AMP, seed=0. Trained on a Colab
T4 in three sessions (epochs 1–52, 53–86, 87–100) by resuming from `last.pt`;
the learning-rate schedule continues without a jump across resumes. Total
training time ≈ 4.2 h (~150 s/epoch).

Command:
```bash
python scripts/train.py --model yolo11s.pt --imgsz 640 --batch 16 --epochs 100 --name baseline_s_640
```

Results from `best.pt` (epoch 71), evaluated with `scripts/evaluate.py`:

| Split | mAP@50-95 | mAP@50 | Precision | Recall |
|---|---|---|---|---|
| val | 0.225 | 0.385 | 0.516 | 0.396 |
| test-dev | 0.186 | 0.327 | 0.463 | 0.352 |

(The per-epoch validation during training reports 0.227 mAP@50-95 at epoch 71; the
small difference comes from the standalone evaluation using a different batch size
and padding.)

Per-class mAP@50-95:

| Class | val | test-dev | train boxes |
|---|---|---|---|
| car | 0.544 | 0.456 | 144,866 |
| bus | 0.372 | 0.383 | 5,926 |
| van | 0.293 | 0.247 | 24,956 |
| truck | 0.238 | 0.252 | 12,875 |
| motor | 0.200 | 0.122 | 29,647 |
| pedestrian | 0.192 | 0.110 | 79,337 |
| tricycle | 0.152 | 0.104 | 4,812 |
| people | 0.117 | 0.046 | 27,059 |
| awning-tricycle | 0.085 | 0.105 | 3,246 |
| bicycle | 0.056 | 0.040 | 10,480 |

Inference speed on the T4: ~7.8 ms/img model forward pass (+ ~3 ms pre/post-processing).

Ultralytics' own plots for this run (PR curve, confusion matrix, prediction samples)
are in `results/runs/baseline_s_640/`, which is gitignored; reproduce with the command above.

---

## Part II — Loss Curve & Fitting Analysis

![Training vs validation loss](../results/figures/loss_baseline_s_640.png)

![Validation mAP@50-95](../results/figures/map_baseline_s_640.png)

Total loss = box + cls + dfl.

| Epoch | Train loss | Val loss | Gap (val − train) | val mAP@50-95 |
|---|---|---|---|---|
| 1 | 4.172 | 3.722 | −0.450 | 0.129 |
| 40 | 3.152 | 3.184 | 0.032 | 0.217 |
| 71 (best) | 2.937 | **3.139** | 0.202 | **0.227** |
| 90 | 2.798 | 3.164 | 0.366 | 0.222 |
| 100 | 2.694 | 3.188 | 0.493 | 0.217 |

- **Convergence behavior:** In the first epochs, validation loss is *lower* than
  training loss because training images are heavily augmented (mosaic, HSV,
  scale/translate) while validation images are not. The bump at epoch 3 is the end
  of warmup, when the learning rate reaches its peak. From there both losses fall
  together until ~epoch 40. Validation loss then flattens around 3.15, and mAP is
  within 0.005 of its best from epoch 45 on. The model has essentially converged
  on the validation set by the midpoint of the 100-epoch schedule.
- **Overfitting?** Mild, late overfitting. Validation loss reaches its minimum at
  epoch 71 (3.139), which is also the epoch with the best mAP and the source of
  `best.pt`. After that, training loss keeps falling (2.94 → 2.69) while validation
  loss rises slightly (3.14 → 3.19), and the gap grows from 0.20 to 0.49. mAP drops
  only from 0.227 to 0.217, so the overfitting is real but small. The last 10 epochs
  make it clearer: with `close_mosaic=10`, mosaic augmentation is switched off, the
  training images become easier, and training loss drops sharply (2.80 → 2.69). Validation
  loss does not follow; it keeps rising. Without mosaic as a regularizer, the extra
  fit goes into the training set, not into generalization.
- **Underfitting?** Yes, in absolute terms. A validation loss that plateaus at ~3.15
  and a mAP@50-95 of 0.225 show the model cannot represent much of the task: recall
  is only 0.40, so most objects are missed. The fact that training loss keeps
  falling shows the model can still fit the training data. The ceiling is in what
  it can generalize at this resolution and size, not in optimization.
- **Causes — dataset size:** 6,471 training images is small for a 9.4M-parameter
  detector trained for 100 epochs: each image is seen 100 times, which is enough
  for the model to start memorizing image-specific details once the shared
  features are learned (~epoch 70). However, the dataset is dense (53 boxes per
  image, 343k boxes), so the *box* count is large; this is why overfitting stays
  mild rather than severe. The imbalance matters too: car (145k boxes) reaches
  0.54 mAP, while classes with few examples, like awning-tricycle (3.2k) and
  tricycle (4.8k), stay below 0.16. Pedestrian has 79k boxes but still reaches only
  0.19, so object size matters as much as the number of examples (next point).
- **Causes — model capacity:** 85% of boxes are smaller than 32×32 px at 640 input,
  and the median box is ~14×14 px. YOLO11's finest detection head works on a stride-8
  feature map, so a 14-px object covers only about 2×2 cells. `yolo11s` has
  limited width to encode such small, low-detail objects, and downscaling the
  original images (about 1360–2000 px wide) to 640 removes much of the detail before
  the network sees it. Large classes do well (car 0.54, bus 0.37); tiny or similar-looking
  classes do poorly: bicycle 0.06, and people 0.12, which is easily confused with
  pedestrian. This points to underfitting from limited capacity and resolution,
  with mild overfitting layered on top late in training.
- **Implications for the next experiments:**
  - The capacity/resolution limit suggests testing model size (n vs s vs m) and a
    higher `imgsz` (Part III).
  - The late overfitting suggests stronger regularization or keeping augmentation
    on longer (Part IV).

---

## Part III — Structured Experimental Design

> **Status:** designed; run `exp_n_640` is queued next in the Colab notebook.

### Round 1 — Model capacity (yolo11n vs yolo11s)
- **Question:** Part II concluded the baseline is mainly *capacity/resolution-limited*
  (underfitting on small objects), with mild late overfitting. If capacity matters, a
  smaller model should lose accuracy, and the loss should drop most on small and rare
  classes. If the 6.5k-image dataset is the real limit, a smaller model should
  overfit *less* and lose little accuracy.
- **Controlled variable:** model size only: `yolo11n` (≈2.6M params) vs `yolo11s`
  (9.4M). Everything else identical to the baseline: imgsz 640, batch 16, 100
  epochs, optimizer auto, seed 0, default augmentation, Colab T4.
- **Command:**
  ```bash
  python scripts/train.py --model yolo11n.pt --imgsz 640 --batch 16 --epochs 100 --name exp_n_640
  ```

| Run | Model | Params (M) | imgsz | batch | epochs | val mAP@50-95 | val mAP@50 | P | R | test mAP@50-95 |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline_s_640 | yolo11s | 9.4 | 640 | 16 | 100 | 0.225 | 0.385 | 0.516 | 0.396 | 0.186 |
| exp_n_640 | yolo11n | | 640 | 16 | 100 | | | | | |

**Results & analysis:** _pending run._ Compare: (1) overall and per-class mAP,
(2) the train/val gap and the epoch where val loss is lowest (does the smaller model
overfit later or less?), (3) the trade-off between accuracy, training time and speed.

### Round 2 (optional) — Input resolution (640 vs 960)
- **Question:** does giving the network more pixels per object (85% of boxes are
  < 32×32 px at 640) raise mAP, and by how much on small classes?
- **Controlled variable:** `imgsz` 640 → 960, `yolo11s`, everything else identical.
  Costs ~2.25× compute per epoch (~9–10 h on a T4), so it runs only if GPU time allows.

| Run | Model | imgsz | val mAP@50-95 | val mAP@50 | P | R | test mAP@50-95 |
|---|---|---|---|---|---|---|---|
| baseline_s_640 | yolo11s | 640 | 0.225 | 0.385 | 0.516 | 0.396 | 0.186 |
| exp_s_960 | yolo11s | 960 | | | | | |

---

## Part IV — Iterative Model Improvement

### Cycle 1 — L2 regularization (weight decay 0.0005 → 0.001)

**1. Baseline.** `baseline_s_640` (Part I): val mAP@50-95 0.225, test-dev 0.186. Part II
found mild overfitting after epoch 71: validation loss rises from 3.139 to 3.188 while
training loss keeps falling, and the train/val gap grows from 0.20 to 0.49.

**2. Experimental settings.** Identical to the baseline: `yolo11s`, imgsz 640,
batch 16, 100 epochs, optimizer auto (MuSGD), lr0 0.01, seed 0, default augmentation
with `close_mosaic=10`, Colab T4.

**3. Controlled modification.** One change: `weight_decay` 0.0005 → 0.001.
- **Principle:** weight decay adds an L2 penalty λ‖w‖² to the loss, which pulls weights
  toward zero and limits how sharply the network can fit individual training images.
  Larger λ means a smoother, lower-capacity function and should reduce overfitting.
- **Effective strength:** Ultralytics scales weight decay by batch × accumulate / nbs
  = 16 × 4 / 64 = 1, so the effective λ really doubles.
- **Which weights:** it applies to conv weights (including MuSGD's Muon parameter
  group), not to biases or BatchNorm.

```bash
python scripts/train.py --model yolo11s.pt --imgsz 640 --batch 16 --epochs 100 \
    --name exp_s_640_wd001 --extra weight_decay=0.001
```

**4. Evaluation.** `best.pt` of each run, standalone evaluation:

| Run | Change | val mAP@50-95 | val mAP@50 | val P | val R | test mAP@50-95 | test mAP@50 | Train time |
|---|---|---|---|---|---|---|---|---|
| baseline_s_640 | — | 0.2247 | 0.3849 | 0.516 | 0.396 | 0.1863 | 0.3265 | 4.16 h |
| exp_s_640_wd001 | weight_decay 0.001 | 0.2276 | 0.3899 | 0.534 | 0.391 | 0.1854 | 0.3260 | 4.35 h |
| **Δ** | | **+0.003** | +0.005 | +0.019 | −0.005 | **−0.001** | −0.001 | |

![Loss: baseline vs weight decay](../results/figures/loss_baseline_s_640_vs_exp_s_640_wd001.png)

![mAP: baseline vs weight decay](../results/figures/map_baseline_s_640_vs_exp_s_640_wd001.png)

Training dynamics (total loss = box + cls + dfl):

| Epoch | Train loss (base / wd) | Val loss (base / wd) | Gap (base / wd) | val mAP@50-95 (base / wd) |
|---|---|---|---|---|
| 40 | 3.152 / 3.179 | 3.184 / 3.188 | 0.032 / 0.008 | 0.217 / 0.218 |
| 60 | 3.010 / 3.043 | 3.150 / 3.154 | 0.140 / 0.111 | 0.224 / 0.229 |
| 73 | 2.931 / 2.963 | 3.151 / 3.139 | 0.220 / 0.177 | 0.226 / **0.230** |
| 90 | 2.798 / 2.853 | 3.164 / 3.149 | 0.366 / 0.296 | 0.222 / 0.227 |
| 100 | 2.694 / 2.691 | 3.188 / 3.186 | 0.493 / 0.495 | 0.217 / 0.221 |

Per-class val mAP@50-95 changed by at most ±0.015 (largest gain: bus +0.015;
largest drop: tricycle −0.007).

**5. Analysis.**
- **Weight decay did regularize while mosaic was on.** Over epochs 41–90, training
  loss is on average 0.031 higher than the baseline at the same validation loss, so
  the model fits the training set less tightly without generalizing worse. The gap at
  epoch 90 shrinks from 0.37 to 0.30, and late-training mAP holds up better: the best
  epoch moves from 71 to 73, and mAP at epoch 90 is 0.227 vs 0.222.
- **The benefit disappears when mosaic switches off.** In the last 10 epochs
  (`close_mosaic=10`), both runs drop to the same training loss (2.69) and gap (0.49).
  Removing mosaic makes the training images far easier, and that change dominates the
  small L2 penalty. The model then fits the training set equally tightly in both runs.
- **The best model barely changed.** Validation +0.003 and test-dev −0.001 are within
  what you would expect from a different random seed; we ran one seed per setting, so
  this cannot be separated from noise. Precision rose (+0.019) and recall fell
  (−0.005): the model became slightly more conservative, not better at finding objects.
- **Why there is no real gain:** the overfitting was mild and late; best.pt is chosen
  at the val-loss minimum anyway, so it already avoids most of it. The dominant
  problem is underfitting on tiny objects (Part II), which regularization cannot fix,
  because it limits capacity rather than adding it.

**6. Conclusion.** Doubling weight decay is a valid regularizer here: it measurably
narrows the train/val gap during the mosaic phase. But it does **not** improve the
final detector, because (a) its effect is erased once mosaic is switched off, and (b)
overfitting is not the factor limiting accuracy. This motivates two follow-ups:
cycle 2 (keep mosaic on until the end) and the Part III capacity/resolution
experiments, which target the real bottleneck.

### Cycle 2 — Keep mosaic augmentation to the end (`close_mosaic=0`)

> **Status:** queued after Part III in the Colab notebook.

- **Baseline for comparison:** `baseline_s_640` (default `close_mosaic=10`).
- **Hypothesis (from cycle 1):** both runs overfit fastest in the final 10 epochs,
  exactly when mosaic is switched off. Mosaic acts as a strong data-level regularizer:
  it makes new compositions and scales for every batch. Keeping it on to the end should
  stop the late jump in the gap and keep validation loss from rising. The risk is that
  without a final "clean images" phase, the model is less tuned to un-mosaiced images,
  which is the format at test time.
- **Controlled modification:** `close_mosaic` 10 → 0. Everything else identical.
  ```bash
  python scripts/train.py --model yolo11s.pt --imgsz 640 --batch 16 --epochs 100 \
      --name exp_s_640_closemosaic0 --extra close_mosaic=0
  ```

| Run | Change | val mAP@50-95 | val mAP@50 | val P | val R | test mAP@50-95 | Gap @100 |
|---|---|---|---|---|---|---|---|
| baseline_s_640 | — | 0.2247 | 0.3849 | 0.516 | 0.396 | 0.1863 | 0.493 |
| exp_s_640_closemosaic0 | close_mosaic 0 | | | | | | |

**Evaluation / analysis / conclusion:** _pending run._

---

## Part V — Multi-Version YOLO Comparison (optional)

> **Status:** prepared; `compare_v8s_640` is in the Colab run queue.

Matched recipe: same size class ("s"), imgsz 640, batch 16, 100 epochs, optimizer
auto, seed 0, default augmentation, same T4 GPU type, so differences come from the
architecture. Params/GFLOPs and inference time come from each run's `metrics.json`
(written by `scripts/evaluate.py`); training time from `summarize_runs.py`.

| Model | val mAP@50-95 | val mAP@50 | P | R | test mAP@50-95 | Params (M) | GFLOPs | Train time (T4) | Inference (ms/img, T4) |
|---|---|---|---|---|---|---|---|---|---|
| yolo11s | 0.225 | 0.385 | 0.516 | 0.396 | 0.186 | 9.4 | 21.7 | 4.16 h | 7.8 |
| yolov8s | | | | | | | | | |

**Discussion:** _pending run._ Cover accuracy vs speed vs size trade-offs, per-class
differences (especially small classes), and the confusion matrices
(`eval_val/confusion_matrix_normalized.png` in each run folder).

---

## Reproducibility
- Exact commands for every run are in the README and in each section above.
- Every run uses `--seed 0`. Environment: `requirements.txt`.
- All Colab runs use `notebooks/colab_train.ipynb`; only its settings block changes
  between experiments.
- Re-evaluating the Colab-trained baseline `best.pt` on the local RTX 3060 gives the
  same val mAP@50-95 (0.2248 vs 0.2247).
- Limitation: one seed per configuration. Differences smaller than ~0.005 mAP should
  be treated as within run-to-run noise.
