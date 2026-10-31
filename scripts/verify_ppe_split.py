"""Check frozen test integrity and the evaluation working copy without inference."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dataset_tools import digest


def main():
    data = ROOT / "data/dataset_ppe"
    frozen = ROOT / "data/splits_frozen/ppe"
    splits = json.loads((data / "split_manifest.json").read_text(encoding="utf-8"))
    snapshot = json.loads((frozen / "manifest.json").read_text(encoding="utf-8"))
    if snapshot != splits["test"]:
        raise ValueError("Frozen manifest differs from the planned test split")
    groups = [{row["sha256"] for row in splits[name]} for name in ("train", "val", "test")]
    if sum(map(len, groups)) != sum(map(len, splits.values())):
        raise ValueError("Duplicates exist within a split")
    if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Image overlap exists across splits")
    for index, row in enumerate(snapshot, 1):
        relative = Path(row["image"])
        for root in (frozen, data):
            image = root / "test/images" / relative
            label = root / "test/labels" / relative.with_suffix(".txt")
            if digest(image) != row["sha256"] or digest(label) != row["label_sha256"]:
                raise ValueError(f"Test content changed: {image}")
        if index % 200 == 0:
            print(f"Verified {index}/{len(snapshot)} test pairs", flush=True)
    report = {"state": "passed", "images": {name: len(rows) for name, rows in splits.items()},
              "cross_split_exact_duplicates": 0, "test_pairs_verified": len(snapshot),
              "checks": "SHA-256 images and labels in frozen snapshot and evaluation working copy",
              "inference_run": False}
    (ROOT / "results/ppe-split-verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
