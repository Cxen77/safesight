"""Wait for this baseline's successful completion, then evaluate its test set once."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    progress = ROOT / "results/post-training-status.json"

    def update(state, **details):
        progress.write_text(json.dumps({"state": state, **details,
            "updated_at": datetime.now(timezone.utc).isoformat()}, indent=2), encoding="utf-8")

    update("waiting_for_training")
    while True:
        try:
            status = json.loads((ROOT / "results/ppe-training-status.json").read_text(encoding="utf-8"))
        except (FileNotFoundError, PermissionError, json.JSONDecodeError):
            time.sleep(30)
            continue
        if status["state"] in {"failed", "interrupted"}:
            update("needs_training_resume", training_error=status.get("error", status["state"]))
            return 1
        if status["state"] == "complete":
            break
        time.sleep(30)
    if (ROOT / "results/ppe_eval.md").exists():
        update("complete", report="results/ppe_eval.md", note="Existing report preserved; no reevaluation")
        return 0
    update("evaluating")
    result = subprocess.run([sys.executable, "-u", "src/train_report.py"], cwd=ROOT)
    update("complete" if result.returncode == 0 else "failed", return_code=result.returncode)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
