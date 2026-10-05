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

On Colab, pass --sync-dir <Drive folder> to mirror the run directory into Google
Drive after every epoch and once more when training ends. Training itself stays
on the fast local disk, and a failed Drive copy is logged but never stops
training. With --resume, a run that only survives in --sync-dir is copied back
to local disk first, so a fresh Colab session can continue it.
"""
import argparse
import shutil
import time
from pathlib import Path

import torch
from ultralytics import YOLO

# Repo root (this file lives in <repo>/scripts/). Used to force an absolute
# output path so Ultralytics writes to <repo>/results/runs/<name> instead of
# nesting the runs under its own default runs_dir (e.g. runs/detect/...).
REPO_ROOT = Path(__file__).resolve().parent.parent


def mirror(src: Path, dst: Path):
    """Copy new/changed files from src to dst (one-way, never deletes)."""
    for f in src.rglob("*"):
        if not f.is_file():
            continue
        t = dst / f.relative_to(src)
        s_stat = f.stat()
        if t.exists() and t.stat().st_size == s_stat.st_size and t.stat().st_mtime >= s_stat.st_mtime:
            continue
        t.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, t)


def add_sync_callbacks(model, sync_root: Path):
    """Mirror trainer.save_dir into sync_root/<run name> each epoch and at the end."""
    def sync(trainer, final=False):
        dst = sync_root / Path(trainer.save_dir).name
        try:
            mirror(Path(trainer.save_dir), dst)
            if final:
                print(f"\n[sync] final copy of run saved to {dst}")
        except Exception as e:  # Drive hiccup: keep training, retry next epoch
            print(f"\n[sync] WARNING: copy to {dst} failed ({e}); will retry")

    model.add_callback("on_fit_epoch_end", sync)
    model.add_callback("on_train_end", lambda tr: sync(tr, final=True))


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
    ap.add_argument("--sync-dir", default=None,
                    help="Folder (e.g. on Google Drive) to mirror the run into after every epoch "
                         "and at the end of training.")
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

    sync_root = Path(args.sync_dir) if args.sync_dir else None
    run_dir = Path(train_kwargs["project"]) / args.name
    if sync_root:
        backup = sync_root / args.name
        if args.resume and not (run_dir / "weights" / "last.pt").exists() \
                and (backup / "weights" / "last.pt").exists():
            # New Colab session: the local run is gone, pull it back from Drive.
            print(f"Restoring run from {backup} -> {run_dir}")
            mirror(backup, run_dir)
        elif not args.resume and backup.exists() and any(backup.iterdir()):
            # Don't mix a fresh run into an old run's folder; keep the old one aside.
            old = backup.with_name(f"{backup.name}_prev_{time.strftime('%Y%m%d-%H%M%S')}")
            print(f"Existing {backup} moved to {old}")
            backup.rename(old)

    # Resume: load the run's last checkpoint and allow writing back into the same
    # dir. Ultralytics resumes from the weights the model was built with, so we must
    # point YOLO at last.pt rather than the fresh pretrained model.
    if args.resume:
        ckpt = run_dir / "weights" / "last.pt"
        if not ckpt.exists():
            raise SystemExit(f"--resume set but no checkpoint found at {ckpt}")
        # A finished run's last.pt has its optimizer stripped (epoch == -1); Ultralytics
        # would then silently start a brand-new training run instead of resuming.
        if torch.load(ckpt, map_location="cpu", weights_only=False).get("epoch", -1) < 0:
            raise SystemExit(f"{ckpt} is from a run that already finished; nothing to resume.")
        train_kwargs["exist_ok"] = True
        model_src = str(ckpt)
    else:
        model_src = args.model

    print("Training with settings:")
    for k, v in train_kwargs.items():
        print(f"  {k}: {v}")
    print(f"  model: {model_src}")

    model = YOLO(model_src)
    if sync_root:
        add_sync_callbacks(model, sync_root)
    model.train(**train_kwargs)
    print(f"\nDone. Results in {train_kwargs['project']}/{args.name}/")
    if sync_root:
        print(f"Mirrored to {sync_root / args.name}/")


if __name__ == "__main__":
    main()
