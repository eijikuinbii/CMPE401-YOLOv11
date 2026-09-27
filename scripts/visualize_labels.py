"""Sanity-check the VisDrone -> YOLO conversion by drawing a few labeled images.

Draws normalized YOLO boxes back onto images so you can confirm the conversion
is correct before training. Saves annotated samples to results/figures/samples/.

Usage:
    python scripts/visualize_labels.py --split VisDrone2019-DET-train --n 6
"""
import argparse
import random
from pathlib import Path

import cv2

CLASS_NAMES = ["pedestrian", "people", "bicycle", "car", "van",
               "truck", "tricycle", "awning-tricycle", "bus", "motor"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/VisDrone")
    ap.add_argument("--split", default="VisDrone2019-DET-train")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    split_dir = Path(args.root) / args.split
    img_dir = split_dir / "images"
    lbl_dir = split_dir / "labels"
    if not lbl_dir.is_dir():
        raise SystemExit(f"No labels/ in {split_dir}. Run convert_visdrone.py first.")

    out_dir = Path("results/figures/samples")
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = sorted(lbl_dir.glob("*.txt"))
    random.seed(args.seed)
    random.shuffle(labels)

    drawn = 0
    for lbl in labels:
        if drawn >= args.n:
            break
        img_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            cand = img_dir / (lbl.stem + ext)
            if cand.exists():
                img_path = cand
                break
        if img_path is None:
            continue

        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        for line in lbl.read_text().splitlines():
            if not line.strip():
                continue
            cls, cx, cy, bw, bh = line.split()
            cls = int(cls)
            cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
            x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
            x2, y2 = int(cx + bw / 2), int(cy + bh / 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(img, CLASS_NAMES[cls], (x1, max(y1 - 2, 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        out = out_dir / f"{lbl.stem}.jpg"
        cv2.imwrite(str(out), img)
        print(f"saved {out}")
        drawn += 1

    print(f"\nDrew {drawn} samples in {out_dir}")


if __name__ == "__main__":
    main()
