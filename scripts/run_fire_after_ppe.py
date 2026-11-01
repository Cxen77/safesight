"""Wait for PPE training and its test report, then train and evaluate D-Fire."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "results/fire-queue-status.json"


def update(state, **values):
    temp = STATUS.with_suffix(".tmp")
    temp.write_text(json.dumps({"state": state, **values,
                               "updated_at": datetime.now(timezone.utc).isoformat()}, indent=2),
                    encoding="utf-8")
    temp.replace(STATUS)


def main():
    update("waiting_for_ppe_training")
    while True:
        try:
            training = json.loads((ROOT / "results/ppe-training-status.json").read_text(encoding="utf-8"))
        except (FileNotFoundError, PermissionError, json.JSONDecodeError):
            time.sleep(30)
            continue
        if training.get("state") in {"failed", "interrupted"}:
            update("needs_attention", prerequisite="ppe_training", detail=training.get("error"))
            return 1
        if training.get("state") == "complete":
            break
        time.sleep(30)
    update("waiting_for_ppe_evaluation")
    while True:
        try:
            evaluation = json.loads((ROOT / "results/post-training-status.json").read_text(encoding="utf-8"))
        except (FileNotFoundError, PermissionError, json.JSONDecodeError):
            time.sleep(30)
            continue
        if evaluation.get("state") == "failed":
            update("needs_attention", prerequisite="ppe_evaluation", detail=evaluation)
            return 1
        if evaluation.get("state") == "complete" and (ROOT / "results/ppe_eval.json").is_file():
            break
        time.sleep(30)
    final_model = ROOT / "models/fire_best.pt"
    run = ROOT / "results/fire_public_baseline"
    checkpoint = run / "weights/last.pt"
    if not final_model.is_file():
        command = [sys.executable, "-u", "src/train_fire.py"]
        if checkpoint.is_file():
            command += ["--resume", str(checkpoint)]
        elif run.exists():
            command += ["--name", "fire_public_baseline_retry"]
        update("training_fire", resume=checkpoint.is_file())
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode:
            update("fire_training_failed", return_code=result.returncode)
            return result.returncode
    if not (ROOT / "results/fire_eval.json").is_file():
        update("evaluating_fire")
        result = subprocess.run([sys.executable, "-u", "src/fire_report.py"], cwd=ROOT)
        if result.returncode:
            update("fire_evaluation_failed", return_code=result.returncode)
            return result.returncode
    update("writing_summary")
    result = subprocess.run([sys.executable, "-u", "src/evaluate.py", "--map"], cwd=ROOT)
    update("complete" if result.returncode == 0 else "summary_failed",
           return_code=result.returncode)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
