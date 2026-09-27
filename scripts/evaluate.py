"""Evaluate a trained YOLO model on a VisDrone split and print key metrics.

Examples:
    python scripts/evaluate.py --weights results/runs/baseline_s_640/weights/best.pt --split val
    python scripts/evaluate.py --weights results/runs/baseline_s_640/weights/best.pt --split test

Metrics reported: mAP@50-95, mAP@50, precision, recall (and per-class if --verbose).
"""
import argparse

from ultralytics import YOLO


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
    ap.add_argument("--verbose", action="store_true", help="Show per-class metrics.")
    args = ap.parse_args()

    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project="results/runs",
        name=args.name or f"eval_{args.split}",
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


if __name__ == "__main__":
    main()
