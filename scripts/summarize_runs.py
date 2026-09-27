"""Build a comparison table across runs for Parts III / IV / V.

Reads the final-epoch metrics from each run's results.csv and emits a Markdown
table (and CSV) suitable for pasting into the report / README.

Examples:
    python scripts/summarize_runs.py results/runs/baseline_s_640 results/runs/exp_m_1280
    python scripts/summarize_runs.py results/runs/* --out results/tables/experiments.md
"""
import argparse
from pathlib import Path

import pandas as pd

COLS = {
    "metrics/precision(B)": "precision",
    "metrics/recall(B)": "recall",
    "metrics/mAP50(B)": "mAP@50",
    "metrics/mAP50-95(B)": "mAP@50-95",
}


def best_row(run_dir: Path) -> dict:
    df = pd.read_csv(run_dir / "results.csv")
    df.columns = [c.strip() for c in df.columns]
    # pick the epoch with best mAP@50-95 (matches how best.pt is chosen)
    key = "metrics/mAP50-95(B)"
    row = df.loc[df[key].idxmax()] if key in df else df.iloc[-1]
    out = {"run": run_dir.name, "epochs": int(df["epoch"].max()) + 1 if "epoch" in df else len(df)}
    for raw, pretty in COLS.items():
        out[pretty] = round(float(row[raw]), 4) if raw in df else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="results/tables/summary.md")
    args = ap.parse_args()

    rows = []
    for r in args.runs:
        p = Path(r)
        if (p / "results.csv").exists():
            rows.append(best_row(p))
        else:
            print(f"[skip] no results.csv in {p}")

    if not rows:
        raise SystemExit("No runs with results.csv found.")

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out.with_suffix(".csv"), index=False)
    out.write_text(df.to_markdown(index=False))
    print(df.to_markdown(index=False))
    print(f"\nsaved {out} and {out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
