"""Train a YOLO model on VisDrone-DET.

Thin, reproducible wrapper around Ultralytics so every experiment is a single
named command whose settings are captured in the run directory.

Examples:
    # Part I baseline (local RTX 3060, 6 GB): small model, 640px, small batch
    python scripts/train.py --model yolo11s.pt --imgsz 640 --batch 8 --epochs 100 --name baseline_s_640

    # Part III experiment: larger resolution on Colab
    python scripts/train.py --model yolo11m.pt --imgsz 1280 --batch 16 --epochs 100 --name exp_m_1280

    # Part V comparison: same recipe, different family
    python scripts/train.py --model yolov8s.pt --imgsz 640 --batch 8 --epochs 100 --name compare_v8s_640

Results (weights, results.csv, curves, confusion matrix) are written to
results/runs/<name>/.
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

# Repo root (this file lives in <repo>/scripts/). Used to force an absolute
# output path so Ultralytics writes to <repo>/results/runs/<name> instead of
# nesting the runs under its own default runs_dir (e.g. runs/detect/...).
REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description="Train YOLO on VisDrone-DET.")
    ap.add_argument("--model", default="yolo11s.pt",
                    help="Pretrained weights / model yaml (e.g. yolo11n.pt, yolo11s.pt, yolo11m.pt, yolov8s.pt).")
    ap.add_argument("--data", default="configs/visdrone.yaml")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8,
                    help="Batch size. Use -1 for Ultralytics auto-batch (needs headroom).")
    ap.add_argument("--name", required=True, help="Run name (folder under results/runs/).")
    ap.add_argument("--device", default="0", help="'0' for first GPU, 'cpu', or e.g. '0,1'.")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--patience", type=int, default=50,
                    help="Early-stopping patience (epochs without val improvement).")
    ap.add_argument("--optimizer", default="auto")
    ap.add_argument("--lr0", type=float, default=None, help="Initial learning rate override.")
    ap.add_argument("--cos-lr", action="store_true", help="Use cosine LR schedule.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="Extra key=value overrides passed straight to Ultralytics "
                         "(e.g. mosaic=0.0 close_mosaic=0 weight_decay=0.001).")
    args = ap.parse_args()

    # parse free-form key=value overrides for augmentation / regularization experiments
    overrides = {}
    for kv in args.extra:
        if "=" not in kv:
            raise SystemExit(f"--extra items must be key=value, got: {kv}")
        k, v = kv.split("=", 1)
        # best-effort typing
        if v.lower() in ("true", "false"):
            v = v.lower() == "true"
        else:
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    pass
        overrides[k] = v

    train_kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        optimizer=args.optimizer,
        cos_lr=args.cos_lr,
        seed=args.seed,
        resume=args.resume,
        project=str(REPO_ROOT / "results" / "runs"),
        name=args.name,
        exist_ok=False,
        plots=True,   # saves loss curves, PR curves, confusion matrix
    )
    if args.lr0 is not None:
        train_kwargs["lr0"] = args.lr0
    train_kwargs.update(overrides)

    print("Training with settings:")
    for k, v in train_kwargs.items():
        print(f"  {k}: {v}")

    model = YOLO(args.model)
    model.train(**train_kwargs)
    print(f"\nDone. Results in results/runs/{args.name}/")


if __name__ == "__main__":
    main()
