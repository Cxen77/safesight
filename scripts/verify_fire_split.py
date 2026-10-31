"""Verify the cleaned fire split and frozen snapshot without model evaluation."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dataset_tools import digest


def main():
    data, frozen = ROOT / "data/dataset_fire", ROOT / "data/splits_frozen/fire"
    manifest = json.loads((data / "preparation_manifest.json").read_text(encoding="utf-8"))
    snapshot = json.loads((frozen / "manifest.json").read_text(encoding="utf-8"))
    expected = [{k: row[k] for k in ("image", "sha256", "label_sha256")} for row in manifest["splits"]["test"]]
    if snapshot != expected:
        raise ValueError("Frozen fire manifest differs from the prepared test split")
    sets = [{row["sha256"] for row in manifest["splits"][name]} for name in ("train", "val", "test")]
    if sum(map(len, sets)) != sum(len(rows) for rows in manifest["splits"].values()):
        raise ValueError("Duplicate images remain within a split")
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Exact duplicate images occur across splits")
    for row in snapshot:
        for root in (data, frozen):
            image = root / "test/images" / row["image"]
            label = root / "test/labels" / Path(row["image"]).with_suffix(".txt")
            if digest(image) != row["sha256"] or digest(label) != row["label_sha256"]:
                raise ValueError(f"Test image or label changed: {image}")
    report = {"state": "passed", "summary": manifest["summary"], "test_pairs_verified": len(snapshot),
              "quarantined_pairs": len(manifest["quarantined"]),
              "duplicates_excluded": len(manifest["duplicates_excluded"]),
              "cross_split_exact_duplicates": 0, "inference_run": False}
    results = ROOT / "results"
    (results / "fire-split-verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# D-Fire preparation", "", "Original split assignments preserved for retained images.",
             "Invalid image-label pairs excluded without changing source files.",
             "Exact image duplicates keep the test copy first, then validation, then training.", "",
             "| Split | Images | Smoke boxes | Fire boxes |", "| --- | ---: | ---: | ---: |"]
    for name in ("train", "val", "test"):
        row = report["summary"][name]
        lines.append(f"| {name} | {row['images']} | {row['boxes']['smoke']} | {row['boxes']['fire']} |")
    lines += ["", f"Quarantined pairs: {report['quarantined_pairs']}.",
              f"Exact duplicates excluded: {report['duplicates_excluded']}.",
              f"Verified {len(snapshot)} test image/label pairs in both frozen and working copies using SHA-256.",
              "No exact image-byte overlap remains across the three splits.",
              "Class IDs remain 0 smoke, 1 fire, matching the downloaded dataset's declaration.",
              "Training-only config excludes test. Evaluation config uses the working copy so cache writes stay outside the frozen snapshot.",
              "No fire model has been trained or evaluated yet. These are dataset counts, not accuracy measurements.",
              "Near-duplicates and recording-session overlap have not been audited.",
              "Excluded paths and reasons are in data/dataset_fire/preparation_manifest.json."]
    (results / "fire_preparation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
