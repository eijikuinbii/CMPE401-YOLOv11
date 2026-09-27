"""Plot training vs validation loss (and mAP) from Ultralytics results.csv.

Supports Part II (loss curve & fitting analysis). Reads one or more run
directories and produces overlay figures under results/figures/.

Ultralytics logs separate train/val components (box_loss, cls_loss, dfl_loss).
This sums them into a single train-vs-val total loss for the classic
overfitting/underfitting view, and also plots mAP@50-95.

Examples:
    python scripts/plot_curves.py results/runs/baseline_s_640
    python scripts/plot_curves.py results/runs/baseline_s_640 results/runs/exp_m_1280 --tag baseline_vs_exp
"""
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

TRAIN_LOSS_COLS = ["train/box_loss", "train/cls_loss", "train/dfl_loss"]
VAL_LOSS_COLS = ["val/box_loss", "val/cls_loss", "val/dfl_loss"]
MAP_COL = "metrics/mAP50-95(B)"

OUT_DIR = Path("results/figures")


def load(run_dir: Path) -> pd.DataFrame:
    csv = run_dir / "results.csv"
    if not csv.exists():
        raise SystemExit(f"No results.csv in {run_dir}")
    df = pd.read_csv(csv)
    df.columns = [c.strip() for c in df.columns]
    df["train_total_loss"] = df[[c for c in TRAIN_LOSS_COLS if c in df]].sum(axis=1)
    df["val_total_loss"] = df[[c for c in VAL_LOSS_COLS if c in df]].sum(axis=1)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="One or more run directories under results/runs/.")
    ap.add_argument("--tag", default=None, help="Filename tag for the output figures.")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = [Path(r) for r in args.runs]
    tag = args.tag or (runs[0].name if len(runs) == 1 else "compare")

    # --- Figure 1: train vs val total loss ---
    plt.figure(figsize=(8, 5))
    for r in runs:
        df = load(r)
        x = df["epoch"] if "epoch" in df else range(len(df))
        plt.plot(x, df["train_total_loss"], label=f"{r.name} train")
        plt.plot(x, df["val_total_loss"], "--", label=f"{r.name} val")
    plt.xlabel("epoch")
    plt.ylabel("total loss (box + cls + dfl)")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    f1 = OUT_DIR / f"loss_{tag}.png"
    plt.tight_layout()
    plt.savefig(f1, dpi=150)
    print(f"saved {f1}")

    # --- Figure 2: mAP@50-95 ---
    plt.figure(figsize=(8, 5))
    for r in runs:
        df = load(r)
        if MAP_COL not in df:
            continue
        x = df["epoch"] if "epoch" in df else range(len(df))
        plt.plot(x, df[MAP_COL], label=r.name)
    plt.xlabel("epoch")
    plt.ylabel("mAP@50-95")
    plt.title("Validation mAP@50-95")
    plt.legend()
    plt.grid(alpha=0.3)
    f2 = OUT_DIR / f"map_{tag}.png"
    plt.tight_layout()
    plt.savefig(f2, dpi=150)
    print(f"saved {f2}")


if __name__ == "__main__":
    main()
