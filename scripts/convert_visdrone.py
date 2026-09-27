"""Convert VisDrone-DET annotations to YOLO format.

VisDrone raw annotation lines (one object per line) are:
    <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<category>,<truncation>,<occlusion>

Category ids in the raw files:
    0 = ignored regions   (dropped)
    1 = pedestrian        -> 0
    2 = people            -> 1
    3 = bicycle           -> 2
    4 = car               -> 3
    5 = van               -> 4
    6 = truck             -> 5
    7 = tricycle          -> 6
    8 = awning-tricycle   -> 7
    9 = bus               -> 8
    10 = motor            -> 9
    11 = others           (dropped)

Each VisDrone split directory is expected to contain:
    <split>/images/*.jpg
    <split>/annotations/*.txt
This writes YOLO labels to:
    <split>/labels/*.txt   (normalized cx,cy,w,h)

Usage:
    python scripts/convert_visdrone.py --root datasets/VisDrone
    python scripts/convert_visdrone.py --root datasets/VisDrone --splits VisDrone2019-DET-train VisDrone2019-DET-val
"""
import argparse
from pathlib import Path

from PIL import Image
from tqdm import tqdm

# raw VisDrone category id -> YOLO class id (None = drop)
CATEGORY_MAP = {
    0: None, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4,
    6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: None,
}

DEFAULT_SPLITS = [
    "VisDrone2019-DET-train",
    "VisDrone2019-DET-val",
    "VisDrone2019-DET-test-dev",
]


def convert_split(split_dir: Path) -> tuple[int, int, int]:
    """Convert one split. Returns (images_processed, boxes_kept, boxes_dropped)."""
    img_dir = split_dir / "images"
    ann_dir = split_dir / "annotations"
    lbl_dir = split_dir / "labels"

    if not img_dir.is_dir() or not ann_dir.is_dir():
        print(f"  [skip] {split_dir.name}: missing images/ or annotations/")
        return 0, 0, 0

    lbl_dir.mkdir(parents=True, exist_ok=True)

    ann_files = sorted(ann_dir.glob("*.txt"))
    n_img = kept = dropped = 0

    for ann_path in tqdm(ann_files, desc=f"  {split_dir.name}", unit="img"):
        # locate matching image to read its size
        img_path = None
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            cand = img_dir / (ann_path.stem + ext)
            if cand.exists():
                img_path = cand
                break
        if img_path is None:
            continue

        with Image.open(img_path) as im:
            w_img, h_img = im.size
        if w_img == 0 or h_img == 0:
            continue

        out_lines = []
        for raw in ann_path.read_text().splitlines():
            raw = raw.strip().rstrip(",")
            if not raw:
                continue
            parts = raw.split(",")
            if len(parts) < 6:
                continue
            try:
                x, y, bw, bh = (float(parts[i]) for i in range(4))
                cat = int(parts[5])
            except ValueError:
                continue

            cls = CATEGORY_MAP.get(cat)
            if cls is None or bw <= 0 or bh <= 0:
                dropped += 1
                continue

            # to normalized center-x, center-y, w, h; clamp to [0,1]
            cx = (x + bw / 2) / w_img
            cy = (y + bh / 2) / h_img
            nw = bw / w_img
            nh = bh / h_img
            cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
            nw, nh = min(nw, 1.0), min(nh, 1.0)
            out_lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            kept += 1

        (lbl_dir / f"{ann_path.stem}.txt").write_text("\n".join(out_lines))
        n_img += 1

    return n_img, kept, dropped


def main():
    ap = argparse.ArgumentParser(description="Convert VisDrone-DET to YOLO format.")
    ap.add_argument("--root", default="datasets/VisDrone",
                    help="Dataset root containing the VisDrone2019-DET-* split folders.")
    ap.add_argument("--splits", nargs="*", default=DEFAULT_SPLITS,
                    help="Split folder names to convert.")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"Dataset root not found: {root.resolve()}\n"
                         f"Download VisDrone-DET and extract the split folders there first.")

    print(f"Converting VisDrone-DET under: {root.resolve()}")
    tot_img = tot_kept = tot_drop = 0
    for split in args.splits:
        n, k, d = convert_split(root / split)
        tot_img += n
        tot_kept += k
        tot_drop += d

    print("\nDone.")
    print(f"  images processed : {tot_img}")
    print(f"  boxes kept       : {tot_kept}")
    print(f"  boxes dropped    : {tot_drop}  (ignored-regions / others / invalid)")


if __name__ == "__main__":
    main()
