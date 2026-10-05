"""Evaluate a trained YOLO model on a VisDrone split and print key metrics.

Examples:
    python scripts/evaluate.py --weights results/runs/baseline_s_640/weights/best.pt --split val
    python scripts/evaluate.py --weights results/runs/baseline_s_640/weights/best.pt --split test

Metrics reported: mAP@50-95, mAP@50, precision, recall (and per-class if --verbose).
They are also written to metrics.json in the eval output directory.
"""
import argparse
import json
from pathlib import Path

from ultralytics import YOLO

# Absolute output path: a relative project gets nested under Ultralytics' own runs_dir.
REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Path to trained .pt (e.g. .../weights/best.pt).")
    ap.add_argument("--data", default="configs/visdrone.yaml")
    ap.add_argument("--split", default="val", choices=["val", "test"],
                    help="Which split defined in the data yaml to evaluate.")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--name", default=None, help="Optional run name for saved eval artifacts.")
    ap.add_argument("--project", default=str(REPO_ROOT / "results" / "runs"),
                    help="Parent folder for the eval output directory.")
    ap.add_argument("--verbose", action="store_true", help="Show per-class metrics.")
    args = ap.parse_args()

    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(Path(args.project).resolve()),
        name=args.name or f"eval_{args.split}",
        exist_ok=True,
        plots=True,
        verbose=args.verbose,
    )

    b = metrics.box
    print("\n=== Summary ===")
    print(f"  split        : {args.split}")
    print(f"  mAP@50-95    : {b.map:.4f}")
    print(f"  mAP@50       : {b.map50:.4f}")
    print(f"  mAP@75       : {b.map75:.4f}")
    print(f"  precision    : {b.mp:.4f}")
    print(f"  recall       : {b.mr:.4f}")

    summary = {
        "weights": str(args.weights), "split": args.split, "imgsz": args.imgsz,
        "mAP50-95": b.map, "mAP50": b.map50, "mAP75": b.map75,
        "precision": b.mp, "recall": b.mr,
        "per_class_mAP50-95": dict(zip(metrics.names.values(), map(float, b.maps))),
        "speed_ms_per_img": metrics.speed,
    }
    out = Path(metrics.save_dir) / "metrics.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"  saved        : {out}")


if __name__ == "__main__":
    main()
