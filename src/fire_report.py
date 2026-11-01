"""Evaluate the completed public fire model once on its untouched test split."""
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "models/fire_best.pt")
    parser.add_argument("--data", type=Path, default=ROOT / "data/dataset_fire/eval_frozen.yaml")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    report = ROOT / "results/fire_eval.md"
    if report.exists():
        parser.error("Fire test report already exists; do not repeatedly evaluate or tune on the test split")
    for path in (args.weights, args.data):
        if not path.is_file():
            parser.error(f"File not found: {path}")
    lock = ROOT / "results/fire-evaluation.lock"
    try:
        with lock.open("x", encoding="utf-8") as stream:
            stream.write("One fire test-evaluation process at a time\n")
    except FileExistsError:
        parser.error("Fire evaluation is already running, or a prior interrupted run left its lock")
    try:
        if args.data.resolve() == (ROOT / "data/dataset_fire/eval_frozen.yaml").resolve():
            subprocess.run([sys.executable, str(ROOT / "scripts/verify_fire_split.py")],
                           cwd=ROOT, check=True)
        import torch
        from ultralytics import YOLO
        model = YOLO(str(args.weights.resolve()))
        names = {int(k): str(v) for k, v in model.names.items()}
        if names != {0: "smoke", 1: "fire"}:
            raise ValueError(f"Unexpected fire model classes: {names}")
        metrics = model.val(data=str(args.data.resolve()), split="test", imgsz=640,
                            batch=8, device=args.device, workers=2, plots=True,
                            project=str(ROOT / "results"), name="fire_test_eval", exist_ok=False)
        def metric(value):
            number = float(value)
            if not math.isfinite(number) or not 0 <= number <= 1:
                raise ValueError(f"Invalid validation metric: {number}")
            return number
        indexed = {int(cls): i for i, cls in enumerate(metrics.box.ap_class_index)}
        per_class = []
        for cls, name in names.items():
            index = indexed.get(cls)
            per_class.append({"class": name,
                              "precision": None if index is None else metric(metrics.box.p[index]),
                              "recall": None if index is None else metric(metrics.box.r[index]),
                              "map50": None if index is None else metric(metrics.box.ap50[index])})
        matrix = Path(metrics.save_dir) / "confusion_matrix.png"
        if not matrix.is_file():
            raise RuntimeError(f"Validation did not create the confusion matrix: {matrix}")
        shutil.copy2(matrix, ROOT / "results/fire_confusion_matrix.png")
        with args.weights.open("rb") as stream:
            checkpoint_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        machine = torch.cuda.get_device_name(int(args.device)) if args.device.isdecimal() else args.device
        measured = {"date": datetime.now(timezone(timedelta(hours=5, minutes=30))).isoformat(),
                    "machine": machine, "weights": str(args.weights.resolve()),
                    "weights_sha256": checkpoint_hash, "data": str(args.data.resolve()),
                    "evaluation_split": "test", "map50": metric(metrics.box.map50),
                    "map50_95": metric(metrics.box.map), "per_class": per_class}
        lines = ["# D-Fire test evaluation", "", f"Measured: {measured['date']}",
                 f"Machine: {machine}", f"Weights: `{measured['weights']}`", "",
                 f"Overall mAP@0.50: **{measured['map50']:.4f}**",
                 f"Overall mAP@0.50:0.95: **{measured['map50_95']:.4f}**", "",
                 "| Class | Precision | Recall | mAP@0.50 |", "| --- | ---: | ---: | ---: |"]
        for row in per_class:
            values = ["N/A" if row[key] is None else f"{row[key]:.4f}"
                      for key in ("precision", "recall", "map50")]
            lines.append("| " + row["class"] + " | " + " | ".join(values) + " |")
        lines += ["", "![Confusion matrix](fire_confusion_matrix.png)", "", "## Scope", "",
                  "This is a public D-Fire holdout evaluation, not validation at the demonstration site.",
                  "The frozen test split was excluded from training and checkpoint selection.",
                  "Near-duplicate or recording-session overlap has not been audited.",
                  "These detector metrics do not measure end-to-end alarm accuracy or response time."]
        (ROOT / "results/fire_eval.json").write_text(json.dumps(measured, indent=2), encoding="utf-8")
        report.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Measured fire test report saved to {report}", flush=True)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
