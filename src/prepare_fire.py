"""Prepare D-Fire while preserving source splits and quarantining invalid pairs."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

import yaml
from dataset_tools import (canonical, copy_image, digest, disjoint,
                           empty_destination, images_at, names_from, read_labels)

ROOT = Path(__file__).resolve().parents[1]


def prepare(source, output, frozen, link_images=False):
    disjoint(source, output, frozen)
    empty_destination(output)
    empty_destination(frozen)
    names = names_from(source / "data.yaml")
    names = {cls: canonical(name) for cls, name in names.items()}
    if set(names.values()) != {"smoke", "fire"} or sorted(names) != [0, 1]:
        raise ValueError("Expected exactly two declared classes: smoke and fire")
    config = yaml.safe_load((source / "data.yaml").read_text(encoding="utf-8-sig"))
    kept, seen, quarantine, duplicates = {}, {}, [], []
    # Keep held-out copies first; exclude matching bytes from development data.
    for split in ("test", "val", "train"):
        image_root = (source / config[split]).resolve()
        if not image_root.is_relative_to(source.resolve()):
            raise ValueError(f"Split path escapes source directory: {image_root}")
        images = images_at(image_root)
        if not images:
            raise ValueError(f"No images in {image_root}")
        kept[split] = []
        for number, image in enumerate(images, 1):
            relative = image.relative_to(image_root)
            label = image_root.parent / "labels" / relative.with_suffix(".txt")
            try:
                rows = read_labels(label, names)
            except ValueError as error:
                quarantine.append({"split": split, "image": str(image), "error": str(error)})
                continue
            sha = digest(image)
            if sha in seen:
                duplicates.append({"split": split, "image": str(image),
                                   "kept_split": seen[sha]["split"], "kept_image": seen[sha]["source_image"]})
                continue
            record = {"split": split, "image": str(relative), "sha256": sha,
                      "label_sha256": digest(label), "source_image": str(image),
                      "source_label": str(label), "box_counts": dict(Counter(row[0] for row in rows))}
            seen[sha] = record
            kept[split].append(record)
            if number % 2000 == 0:
                print(f"Validated {split}: {number}/{len(images)}", flush=True)
        if not kept[split]:
            raise ValueError(f"No valid unique images remain in {split}")
    for split, records in kept.items():
        for folder in ("images", "labels"):
            (output / split / folder).mkdir(parents=True, exist_ok=True)
        for record in records:
            relative = Path(record["image"])
            copy_image(Path(record["source_image"]), output / split / "images" / relative, link_images)
            target = output / split / "labels" / relative.with_suffix(".txt")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(record["source_label"], target)
    full = {"path": str(output.resolve()), "train": "train/images", "val": "val/images",
            "test": "test/images", "nc": 2, "names": names}
    (output / "data.yaml").write_text(yaml.safe_dump(full, sort_keys=False), encoding="utf-8")
    training = {k: v for k, v in full.items() if k != "test"}
    (output / "train_only.yaml").write_text(yaml.safe_dump(training, sort_keys=False), encoding="utf-8")
    evaluation = {**full, "train": None, "val": None}
    (output / "eval_frozen.yaml").write_text(yaml.safe_dump(evaluation, sort_keys=False), encoding="utf-8")
    frozen.mkdir(parents=True, exist_ok=True)
    shutil.copytree(output / "test", frozen / "test")
    frozen_manifest = [{k: row[k] for k in ("image", "sha256", "label_sha256")} for row in kept["test"]]
    (frozen / "manifest.json").write_text(json.dumps(frozen_manifest, indent=2), encoding="utf-8")
    snapshot = {**evaluation, "path": str(frozen.resolve())}
    (frozen / "data.yaml").write_text(yaml.safe_dump(snapshot, sort_keys=False), encoding="utf-8")
    summary = {}
    for split, records in kept.items():
        counts = Counter()
        for record in records:
            counts.update(record["box_counts"])
        summary[split] = {"images": len(records), "boxes": {name: counts[cls] for cls, name in names.items()}}
    manifest = {"source": str(source.resolve()), "policy": "source splits preserved; invalid pairs excluded; "
                "exact duplicate priority test > val > train; source files unchanged",
                "classes": names, "summary": summary, "quarantined": quarantine,
                "duplicates_excluded": duplicates, "splits": kept}
    (output / "preparation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "quarantined_pairs": len(quarantine),
                      "exact_duplicates_excluded": len(duplicates)}, indent=2), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/dfire/dataset")
    parser.add_argument("--out", type=Path, default=ROOT / "data/dataset_fire")
    parser.add_argument("--frozen", type=Path, default=ROOT / "data/splits_frozen/fire")
    parser.add_argument("--link-images", action="store_true")
    args = parser.parse_args()
    try:
        prepare(args.source, args.out, args.frozen, args.link_images)
    except (OSError, ValueError, KeyError, yaml.YAMLError) as error:
        parser.exit(2, f"Fire dataset error: {error}\n")


if __name__ == "__main__":
    main()
