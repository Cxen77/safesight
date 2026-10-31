"""YOLO PPE merge, deterministic multilabel split, and dataset reports.

Use merge for public-data preparation first. Freeze only after own lab data is
included. Public data used to pretrain weights is not an unseen evaluation set.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil

import yaml

NAMES = ["person", "head", "helmet", "vest"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")


def canonical(name):
    return " ".join(str(name).lower().replace("-", " ").replace("_", " ").split())


def names_from(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))
    names = config["names"]
    return {int(k): str(v) for k, v in (enumerate(names) if isinstance(names, list) else names.items())}


def images_at(root):
    return sorted(p for p in Path(root).rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)


def read_labels(path, names):
    if not path.is_file():
        raise ValueError(f"Missing label file (use an empty file for negatives): {path}")
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{number}: expected class x y width height")
        try:
            cls = int(parts[0])
            coords = list(map(float, parts[1:]))
        except ValueError as error:
            raise ValueError(f"{path}:{number}: invalid numeric label") from error
        if cls not in names or not all(math.isfinite(x) and 0 <= x <= 1 for x in coords):
            raise ValueError(f"{path}:{number}: unknown class or invalid normalized coordinates")
        if coords[2] <= 0 or coords[3] <= 0:
            raise ValueError(f"{path}:{number}: box width and height must be positive")
        rows.append((cls, *coords))
    return rows


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def empty_destination(path):
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"Refusing to overwrite nonempty destination: {path}")


def disjoint(*paths):
    resolved = [Path(p).resolve() for p in paths]
    for i, a in enumerate(resolved):
        for b in resolved[i + 1:]:
            if a.is_relative_to(b) or b.is_relative_to(a):
                raise ValueError(f"Source/output directories must not overlap: {a}, {b}")


def copy_image(source, target, link=False):
    target.parent.mkdir(parents=True, exist_ok=True)
    if link:
        os.link(source, target)
    else:
        shutil.copy2(source, target)


def write_yaml(root, split=False):
    config = {"path": str(root.resolve()), "names": dict(enumerate(NAMES)), "nc": 4}
    if split:
        config.update({name: f"{name}/images" for name in SPLITS})
    (root / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def merge(args):
    sources = [(args.a, args.a_yaml)] + ([(args.b, args.b_yaml)] if args.b else [])
    disjoint(args.out, *(root for root, _ in sources))
    empty_destination(args.out)
    aliases = {canonical(name): name for name in NAMES}
    for item in args.rename:
        old, separator, new = item.partition(":")
        if not separator or canonical(new) not in NAMES:
            raise ValueError("--rename must be old:new with a supported target class")
        aliases[canonical(old)] = canonical(new)
    records, seen, duplicates, counts = [], {}, [], Counter()
    for index, (root, metadata) in enumerate(sources):
        names = names_from(metadata or root / "data.yaml")
        remap = {cls: NAMES.index(aliases[canonical(name)]) for cls, name in names.items()
                 if canonical(name) in aliases}
        source_images = images_at(root / "images")
        if not source_images:
            raise ValueError(f"No source images in {root / 'images'}")
        for number, image in enumerate(source_images):
            relative = image.relative_to(root / "images")
            label = root / "labels" / relative.with_suffix(".txt")
            original = read_labels(label, names)
            rows = sorted((remap[row[0]], *row[1:]) for row in original if row[0] in remap)
            sha = digest(image)
            if sha in seen:
                previous = seen[sha]
                if rows != previous["rows"]:
                    raise ValueError(f"Identical image bytes have conflicting labels: {image}, {previous['source']}")
                duplicates.append({"source": str(image), "same_as": previous["source"]})
                continue
            stem = f"source{index}_{number:06d}"
            record = {"image": stem + image.suffix.lower(), "source": str(image.resolve()),
                      "sha256": sha, "rows": rows}
            records.append(record)
            seen[sha] = record
            counts.update(row[0] for row in rows)
    # Validate the entire input before creating an output dataset.
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "labels").mkdir()
    for record in records:
        copy_image(Path(record["source"]), args.out / "images" / record["image"], args.link_images)
        content = "".join(" ".join(map(str, row)) + "\n" for row in record["rows"])
        (args.out / "labels" / Path(record["image"]).with_suffix(".txt")).write_text(content, encoding="utf-8")
    write_yaml(args.out)
    manifest = {"images": records, "duplicates_removed": duplicates,
                "box_counts": {name: counts[i] for i, name in enumerate(NAMES)},
                "image_storage": "hardlink; do not edit images in place" if args.link_images else "copy"}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Merged {len(records)} unique images; removed {len(duplicates)} exact duplicates")
    print(json.dumps(manifest["box_counts"], indent=2))


def inventory(root):
    names = names_from(root / "data.yaml")
    if names != dict(enumerate(NAMES)):
        raise ValueError("Expected the remapped person/head/helmet/vest class order")
    records = []
    for image in images_at(root / "images"):
        relative = image.relative_to(root / "images")
        label = root / "labels" / relative.with_suffix(".txt")
        rows = read_labels(label, names)
        records.append({"image": image, "relative": relative, "label": label,
                        "classes": set(row[0] for row in rows), "sha256": digest(image)})
    if not records:
        raise ValueError(f"No images found in {root}")
    return records


def assign_splits(records):
    """Greedy multilabel stratification with exact image counts, seed 42."""
    if len({row["sha256"] for row in records}) != len(records):
        raise ValueError("Duplicate image bytes found; deduplicate with merge before splitting")
    rng = random.Random(42)
    ordered = records.copy()
    rng.shuffle(ordered)
    totals = Counter(cls for row in records for cls in row["classes"])
    ordered.sort(key=lambda row: min((totals[c] for c in row["classes"]), default=math.inf))
    size = len(records)
    capacity = [int(size * .70), int(size * .15), 0]
    capacity[2] = size - sum(capacity)
    targets = [{c: n * count / size for c, n in totals.items()} for count in capacity]
    assigned, used = {name: [] for name in SPLITS}, [Counter(), Counter(), Counter()]
    for row in ordered:
        choices = [i for i, name in enumerate(SPLITS) if len(assigned[name]) < capacity[i]]
        rng.shuffle(choices)
        rarest = min(row["classes"], key=lambda c: (totals[c], c)) if row["classes"] else None
        def priority(i):
            room = (capacity[i] - len(assigned[SPLITS[i]])) / max(capacity[i], 1)
            if rarest is None:
                return room, room, room
            # Match the rarest label's relative quota first. Absolute deficits
            # would let abundant person/head labels overwhelm helmet and vest.
            rare_deficit = (targets[i][rarest] - used[i][rarest]) / max(targets[i][rarest], 1)
            deficit = sum((targets[i][c] - used[i][c]) / max(targets[i][c], 1) for c in row["classes"])
            return rare_deficit, deficit, room
        selected = max(choices, key=priority)
        assigned[SPLITS[selected]].append(row)
        used[selected].update(row["classes"])
    return assigned


def split(args):
    disjoint(args.data, args.out, args.frozen)
    empty_destination(args.out)
    empty_destination(args.frozen)
    assigned = assign_splits(inventory(args.data))
    print("Planned image counts:", {name: len(rows) for name, rows in assigned.items()})
    if args.dry_run:
        print("DRY RUN: no files written and no test split frozen")
        return
    for name, rows in assigned.items():
        for folder in ("images", "labels"):
            (args.out / name / folder).mkdir(parents=True, exist_ok=True)
        for row in rows:
            copy_image(row["image"], args.out / name / "images" / row["relative"], args.link_images)
            target = args.out / name / "labels" / row["relative"].with_suffix(".txt")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(row["label"], target)
    write_yaml(args.out, split=True)
    # Frozen evidence is an independent copy, never hardlinked to mutable data.
    args.frozen.mkdir(parents=True, exist_ok=True)
    shutil.copytree(args.out / "test", args.frozen / "test")
    manifest = {name: [{"image": str(r["relative"]), "sha256": r["sha256"],
                        "label_sha256": digest(r["label"])} for r in rows]
                for name, rows in assigned.items()}
    (args.out / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (args.frozen / "manifest.json").write_text(json.dumps(manifest["test"], indent=2), encoding="utf-8")
    write_yaml(args.frozen, split=True)
    frozen_yaml = yaml.safe_load((args.frozen / "data.yaml").read_text(encoding="utf-8"))
    frozen_yaml.pop("train")
    frozen_yaml.pop("val")
    (args.frozen / "data.yaml").write_text(yaml.safe_dump(frozen_yaml, sort_keys=False), encoding="utf-8")
    print(f"Test split frozen at {args.frozen}; reruns against this folder will be refused")


def report(args):
    names = names_from(args.data / "data.yaml")
    config = yaml.safe_load((args.data / "data.yaml").read_text(encoding="utf-8-sig"))
    split_paths = {}
    for name in SPLITS:
        value = config.get(name)
        if isinstance(value, str) and not Path(value).is_absolute() and (args.data / value).is_dir():
            split_paths[name] = args.data / value
        elif (args.data / name / "images").is_dir():
            split_paths[name] = args.data / name / "images"
    splits = tuple(split_paths) or ("all",)
    issues = []
    print("split\timages\t" + "\t".join(names.values()) + "\tinvalid_label_files")
    for name in splits:
        image_root = args.data / "images" if name == "all" else split_paths[name]
        counts, images = Counter(), images_at(image_root)
        if not images:
            raise ValueError(f"No images found for report: {image_root}")
        invalid = 0
        for image in images:
            label = image_root.parent / "labels" / image.relative_to(image_root).with_suffix(".txt")
            try:
                counts.update(row[0] for row in read_labels(label, names))
            except ValueError as error:
                invalid += 1
                issues.append({"split": name, "image": str(image), "error": str(error)})
        print(f"{name}\t{len(images)}\t" + "\t".join(str(counts[c]) for c in names) + f"\t{invalid}")
        if name == "train":
            for cls, label in names.items():
                if counts[cls] < 50:
                    print(f"WARNING: {label.upper()} HAS FEWER THAN 50 TRAIN BOXES ({counts[cls]})")
    if args.issues:
        args.issues.parent.mkdir(parents=True, exist_ok=True)
        args.issues.write_text(json.dumps(issues, indent=2), encoding="utf-8")
    if issues:
        raise ValueError(f"{len(issues)} invalid label files; box counts exclude those files. "
                         "Inspect --issues output before training.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    merging = commands.add_parser("merge")
    for option in ("a", "b", "a-yaml", "b-yaml", "out"):
        merging.add_argument(f"--{option}", type=Path, required=option in ("a", "out"))
    merging.add_argument("--rename", action="append", default=[])
    merging.add_argument("--link-images", action="store_true", help="Save disk via hardlinks; never edit images in place")
    merging.set_defaults(run=merge)
    splitting = commands.add_parser("split")
    for option in ("data", "out", "frozen"):
        splitting.add_argument(f"--{option}", type=Path, required=True)
    splitting.add_argument("--dry-run", action="store_true")
    splitting.add_argument("--link-images", action="store_true")
    splitting.set_defaults(run=split)
    reporting = commands.add_parser("report")
    reporting.add_argument("--data", type=Path, required=True)
    reporting.add_argument("--issues", type=Path, help="Write invalid-label details to JSON")
    reporting.set_defaults(run=report)
    args = parser.parse_args()
    try:
        args.run(args)
    except (ValueError, OSError, KeyError, yaml.YAMLError) as error:
        parser.exit(2, f"Dataset error: {error}\n")


if __name__ == "__main__":
    main()
