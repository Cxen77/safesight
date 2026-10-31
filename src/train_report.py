"""Evaluate the completed PPE baseline on test data once and write measured results."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def summarize(metrics, names):
    """Use class indices from the validation API, rather than assuming array order."""
    def number(value):
        value = float(value)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid validation metric: {value}")
        return value
    indexed = {int(cls): i for i, cls in enumerate(metrics.box.ap_class_index)}
    classes, weak = [], []
    for cls, name in names.items():
        index = indexed.get(int(cls))
        if index is None:
            classes.append({"class": name, "precision": None, "recall": None, "map50": None})
            weak.append(f"{name}: no per-class result was returned; check test labels and class coverage.")
            continue
        precision = number(metrics.box.p[index])
        recall = number(metrics.box.r[index])
        ap50 = number(metrics.box.ap50[index])
        classes.append({"class": name, "precision": precision, "recall": recall, "map50": ap50})
        if recall < .6:
            count = 150 if recall < .5 else 75
            weak.append(f"{name}: recall {recall:.3f}; if possible, collect {count} more labelled training photos "
                        "covering this class. Keep test images out of training.")
    return {"map50": number(metrics.box.map50), "map50_95": number(metrics.box.map),
            "per_class": classes, "weak_spots": weak}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "models/ppe_best.pt")
    parser.add_argument("--data", type=Path, default=ROOT / "data/dataset_ppe/eval_frozen.yaml")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    report = ROOT / "results/ppe_eval.md"
    if report.exists():
        parser.error("Test report already exists; do not repeatedly evaluate or tune on frozen test data")
    for path in (args.weights, args.data):
        if not path.is_file():
            parser.error(f"File not found: {path}")
    lock = ROOT / "results/ppe-evaluation.lock"
    try:
        with lock.open("x", encoding="utf-8") as stream:
            stream.write("One test-evaluation process at a time\n")
    except FileExistsError:
        parser.error("Evaluation is already running, or a prior interrupted run left an evaluation lock")
    try:
        if args.data.resolve() == (ROOT / "data/dataset_ppe/eval_frozen.yaml").resolve():
            subprocess.run([sys.executable, str(ROOT / "scripts/verify_ppe_split.py")], cwd=ROOT, check=True)
        import torch
        from ultralytics import YOLO
        model = YOLO(str(args.weights.resolve()))
        metrics = model.val(data=str(args.data.resolve()), split="test", imgsz=640, batch=8,
                            device=args.device, workers=2, plots=True,
                            project=str(ROOT / "results"), name="ppe_test_eval", exist_ok=False)
        measured = summarize(metrics, model.names)
        matrix = Path(metrics.save_dir) / "confusion_matrix.png"
        if not matrix.is_file():
            raise RuntimeError(f"Validation did not create the confusion matrix: {matrix}")
        shutil.copy2(matrix, ROOT / "results/confusion_matrix.png")
        with args.weights.open("rb") as stream:
            checkpoint_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        machine = torch.cuda.get_device_name(int(args.device)) if args.device.isdecimal() else args.device
        measured.update(date=datetime.now(timezone(timedelta(hours=5, minutes=30))).isoformat(),
                        machine=machine, weights=str(args.weights.resolve()), weights_sha256=checkpoint_hash,
                        data=str(args.data.resolve()), evaluation_split="test")
        lines = ["# PPE baseline test evaluation", "", f"Measured: {measured['date']}",
                 f"Machine: {machine}", f"Weights: `{measured['weights']}`", "",
                 f"Overall mAP@0.50: **{measured['map50']:.4f}**",
                 f"Overall mAP@0.50:0.95: **{measured['map50_95']:.4f}**", "",
                 "| Class | Precision | Recall | mAP@0.50 |", "| --- | ---: | ---: | ---: |"]
        for row in measured["per_class"]:
            values = ["N/A" if row[key] is None else f"{row[key]:.4f}" for key in ("precision", "recall", "map50")]
            lines.append("| " + row["class"] + " | " + " | ".join(values) + " |")
        lines += ["", "![Confusion matrix](confusion_matrix.png)", "", "## Weak spots", ""]
        lines += [f"- {item}" for item in measured["weak_spots"]] or ["No class has measured recall below 0.60."]
        lines += ["", "## Scope", "", "This is a public SH17 holdout evaluation, not validation at the demo site.",
                  "The starting checkpoint was general COCO YOLOv8n. Cross-dataset near-duplicate overlap is not audited.",
                  "Test data was excluded from training and checkpoint selection. Do not tune on this report.",
                  "The photo counts above are the guide's collection heuristic, not a measured accuracy improvement."]
        (ROOT / "results/ppe_eval.json").write_text(json.dumps(measured, indent=2), encoding="utf-8")
        report.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Measured test report saved to {report}", flush=True)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
