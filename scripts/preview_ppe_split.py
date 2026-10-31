"""Report a split plan from the validated merge manifest without freezing data."""
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import dataset_tools as dt


def main():
    manifest = json.loads((ROOT / "data/ppe_public/manifest.json").read_text(encoding="utf-8"))
    records = [{**row, "classes": {r[0] for r in row["rows"]}} for row in manifest["images"]]
    groups = dt.assign_splits(records)
    summary = {"status": "preview_only_not_frozen", "seed": 42, "own_lab_data_included": False,
               "evaluation_note": "SH17 is pretrained public data; this is not an unseen-test claim.",
               "splits": {}}
    for name, rows in groups.items():
        boxes = Counter(r[0] for row in rows for r in row["rows"])
        images = Counter(c for row in rows for c in row["classes"])
        summary["splits"][name] = {"images": len(rows),
            "boxes": {label: boxes[i] for i, label in enumerate(dt.NAMES)},
            "images_per_class": {label: images[i] for i, label in enumerate(dt.NAMES)}}
    sets = [{r["sha256"] for r in rows} for rows in groups.values()]
    assert sum(map(len, sets)) == len(records)
    assert not any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3))
    results = ROOT / "results"
    (results / "ppe-public-split-preview.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["# Public PPE preparation report", "", "Source: SH17. Own lab data is not included.",
        f"All {len(records):,} images have matching, validated YOLO labels. No exact image duplicates were found.",
        "Class IDs: 0 person, 1 head, 2 helmet, 3 vest. Other source classes removed.", "",
        "| Dataset / preview | Images | Person boxes | Head boxes | Helmet boxes | Vest boxes |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    total = manifest["box_counts"]
    lines.append(f"| Public corpus | {len(records)} | " + " | ".join(str(total[n]) for n in dt.NAMES) + " |")
    for split, row in summary["splits"].items():
        lines.append(f"| {split} preview | {row['images']} | " +
                     " | ".join(str(row["boxes"][n]) for n in dt.NAMES) + " |")
    lines += ["", "**Preview only:** no split directories or frozen test files were created.",
        "Splits target 70/15/15 proportions of images containing each class. Box counts can vary with crowding.",
        "Final splitting waits for own lab annotations and a review of session/person overlap.",
        "Exact byte duplicates are checked; visual near-duplicates still need review.",
        "Images are hardlinked to conserve disk space. Do not edit source or prepared images in place.",
        "The starting model was pretrained on SH17. A random split of this corpus is not an unseen test set.",
        "Final reporting needs genuinely unseen held-out lab data.", "", "## D-Fire source audit", "", "```text",
        (results / "dfire-inventory.txt").read_text(encoding="utf-8-sig").strip(), "```", "",
        "D-Fire source data and its existing splits are unchanged. Invalid label files must be reviewed before training.",
        "See dfire-label-issues.json for file paths and causes."]
    (results / "dataset_preparation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
