"""Back up completed detector weights and measured reports to the GitHub repo."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "results/publish-status.json"
ARTIFACTS = (
    "models/ppe_best.pt", "models/fire_best.pt",
    "results/ppe_eval.md", "results/ppe_eval.json", "results/confusion_matrix.png",
    "results/fire_eval.md", "results/fire_eval.json", "results/fire_confusion_matrix.png",
    "results/final_numbers.md",
)


def update(state, **values):
    temp = STATUS.with_suffix(".tmp")
    temp.write_text(json.dumps({"state": state, **values,
                               "updated_at": datetime.now(timezone.utc).isoformat()}, indent=2),
                    encoding="utf-8")
    temp.replace(STATUS)


def git(*args, check=True):
    return subprocess.run(["git", *args], cwd=ROOT, check=check,
                          capture_output=True, text=True)


def main():
    update("waiting_for_fire_training_and_evaluation")
    while True:
        try:
            status = json.loads((ROOT / "results/fire-queue-status.json").read_text(encoding="utf-8"))
        except (FileNotFoundError, PermissionError, json.JSONDecodeError):
            time.sleep(30)
            continue
        if status.get("state") in {"fire_training_failed", "fire_evaluation_failed", "needs_attention",
                                    "summary_failed"}:
            update("needs_attention", queue_status=status)
            return 1
        if status.get("state") == "complete":
            break
        time.sleep(30)
    if not (ROOT / "results/final_numbers.md").is_file():
        update("writing_summary")
        subprocess.run([sys.executable, "-u", "src/evaluate.py", "--map"], cwd=ROOT, check=True)
    missing = [path for path in ARTIFACTS if not (ROOT / path).is_file()]
    if missing:
        update("needs_attention", missing_artifacts=missing)
        return 1
    too_large = {path: (ROOT / path).stat().st_size for path in ARTIFACTS
                 if (ROOT / path).stat().st_size >= 95 * 1024 * 1024}
    if too_large:
        update("needs_attention", files_over_safe_github_limit=too_large)
        return 1
    update("staging_artifacts", artifacts=list(ARTIFACTS))
    git("add", "-f", *ARTIFACTS)
    if git("diff", "--cached", "--quiet", check=False).returncode:
        git("commit", "-m", "Add trained detector weights and measured results")
    head = git("rev-parse", "HEAD").stdout.strip()
    update("pushing", commit=head)
    git("push", "origin", "main")
    remote = git("ls-remote", "origin", "refs/heads/main").stdout.split()
    if not remote or remote[0] != head:
        raise RuntimeError("Remote main did not verify at the artifact commit")
    update("complete", commit=head, remote_commit=remote[0], artifacts=list(ARTIFACTS))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        update("failed", error=f"{type(error).__name__}: {error}")
        raise
