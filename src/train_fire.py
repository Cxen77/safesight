"""Train the two-class public D-Fire smoke/fire detector."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import torch
from ultralytics import YOLO
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--name", default="fire_public_baseline")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1 or args.workers < 0:
        parser.error("Epochs and batch must be positive; workers cannot be negative")
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Complete GPU setup before starting this run.")
    if Path(args.name).name != args.name or args.name in {".", ".."}:
        parser.error("--name must be a single folder name")
    run = ROOT / "results" / args.name
    if run.exists() and not args.resume:
        parser.error(f"Run already exists: {run}; use a new name or --resume its last.pt")
    data_file = ROOT / "data/dataset_fire/train_only.yaml"
    data = yaml.safe_load(data_file.read_text(encoding="utf-8"))
    if "test" in data or data.get("names") != {0: "smoke", 1: "fire"}:
        parser.error("Fire training config must use only train/val and preserve smoke=0, fire=1")
    for split in ("train", "val"):
        if not (Path(data["path"]) / data[split]).is_dir():
            parser.error(f"Missing {split} images")
    status_file = ROOT / "results/fire-training-status.json"
    status = {"run": str(run), "epochs_requested": args.epochs, "state": "starting",
              "machine": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "classes": {"0": "smoke", "1": "fire"},
              "evaluation": "public D-Fire holdout; no test evaluation during training"}

    def update(**values):
        status.update(values, updated_at=datetime.now(timezone.utc).isoformat())
        temp = status_file.with_suffix(".tmp")
        temp.write_text(json.dumps(status, indent=2), encoding="utf-8")
        temp.replace(status_file)

    try:
        subprocess.run([sys.executable, str(ROOT / "scripts/verify_fire_split.py")],
                       cwd=ROOT, check=True)
        weights = args.resume or ROOT / "models/yolov8n.pt"
        model = YOLO(str(weights))
        if not args.resume and len(model.names) != 80:
            raise ValueError("Expected the general 80-class COCO starting checkpoint")
        with Path(weights).open("rb") as stream:
            weight_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        update(initial_weights=str(weights), initial_weights_sha256=weight_hash)

        def on_epoch(trainer):
            update(state="training", epochs_completed=min(trainer.epoch + 1, trainer.epochs),
                   epochs_requested=trainer.epochs,
                   validation_metrics={k: float(v) for k, v in trainer.metrics.items()})

        model.add_callback("on_fit_epoch_end", on_epoch)
        model.add_callback("on_train_start", lambda trainer: update(state="training"))
        options = dict(data=str(data_file), device=0, batch=args.batch, workers=args.workers)
        if args.resume:
            model.train(resume=True, **options)
        else:
            model.train(epochs=args.epochs, imgsz=640, patience=15,
                        project=str(ROOT / "results"), name=args.name, seed=42,
                        deterministic=True, cache=False, amp=True, save=True,
                        plots=True, split="val", **options)
        best = model.trainer.best
        if not best.is_file():
            raise RuntimeError("Training ended without a best checkpoint")
        target = ROOT / "models/fire_best.pt"
        if target.exists():
            raise FileExistsError(f"Refusing to replace an existing trained model: {target}")
        shutil.copy2(best, target)
        update(state="complete", best_weights=str(target),
               note="Training completed; frozen-test evaluation has not run")
    except BaseException as error:
        update(state="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
               error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
