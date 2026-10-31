"""Check dataset isolation, duplicate handling, and frozen-output protection."""
from argparse import Namespace
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import unittest

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import dataset_tools as dt


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def dataset(self, name, names, count=100):
        root = self.root / name
        (root / "images").mkdir(parents=True)
        (root / "labels").mkdir()
        (root / "data.yaml").write_text(yaml.safe_dump({"names": names}))
        for i in range(count):
            (root / "images" / f"{i}.jpg").write_bytes(f"fixture-image-{i}".encode())
            # Multiple labels per image, plus genuine empty-label negatives.
            classes = [i % len(names), (i + 1) % len(names)] if i % 10 else []
            (root / "labels" / f"{i}.txt").write_text("".join(f"{c} 0.5 0.5 0.2 0.2\n" for c in classes))
        return root

    def test_split_is_repeatable_disjoint_and_frozen_is_immutable(self):
        source = self.dataset("source", dt.NAMES)
        args = Namespace(data=source, out=self.root / "split", frozen=self.root / "frozen",
                         dry_run=False, link_images=False)
        with redirect_stdout(io.StringIO()):
            dt.split(args)
        groups = [set(dt.digest(p) for p in dt.images_at(args.out / s / "images")) for s in dt.SPLITS]
        self.assertEqual(list(map(len, groups)), [70, 15, 15])
        self.assertEqual(len(set.union(*groups)), 100)
        for i, group in enumerate(groups):
            for other in groups[i + 1:]:
                self.assertFalse(group & other)
        first = dt.assign_splits(dt.inventory(source))
        second = dt.assign_splits(dt.inventory(source))
        self.assertEqual({s: [r['sha256'] for r in v] for s, v in first.items()},
                         {s: [r['sha256'] for r in v] for s, v in second.items()})
        snapshots = {str(p.relative_to(args.frozen)): p.read_bytes()
                     for p in args.frozen.rglob('*') if p.is_file()}
        args.out = self.root / "new-output"
        with self.assertRaisesRegex(ValueError, "nonempty"):
            dt.split(args)
        self.assertFalse(args.out.exists())
        self.assertEqual(snapshots, {str(p.relative_to(args.frozen)): p.read_bytes()
                                    for p in args.frozen.rglob('*') if p.is_file()})
        for s in dt.SPLITS:
            counts = {c: sum(c in r['classes'] for r in first[s]) for c in range(4)}
            self.assertTrue(all(n > 0 for n in counts.values()), counts)

    def test_merge_remaps_aliases_and_deduplicates(self):
        source = self.dataset("source", ["Helmet", "Safety-vest", "Person", "Head", "tool"], 20)
        other = self.dataset("other", ["Helmet", "Safety-vest", "Person", "Head", "tool"], 20)
        out = self.root / "merged"
        args = Namespace(a=source, b=other, a_yaml=None, b_yaml=None, out=out,
                         rename=["safety-vest:vest"], link_images=False)
        with redirect_stdout(io.StringIO()):
            dt.merge(args)
        self.assertEqual(len(dt.images_at(out / "images")), 20)
        self.assertEqual(dt.names_from(out / "data.yaml"), dict(enumerate(dt.NAMES)))
        labels = (out / "labels" / "source0_000001.txt").read_text()
        # Lexical image ordering puts image 1 second: vest->3 and person->0.
        self.assertEqual({int(line.split()[0]) for line in labels.splitlines()}, {0, 3})
        records = dt.inventory(out)
        self.assertTrue(any(not r['classes'] for r in records))

    def test_invalid_labels_and_duplicate_split_fail_before_output(self):
        source = self.dataset("source", dt.NAMES, 10)
        (source / "labels/0.txt").write_text("0 nan 0.5 0.2 0.2\n")
        with self.assertRaisesRegex(ValueError, "coordinates"):
            dt.inventory(source)
        (source / "labels/0.txt").write_text("")
        (source / "images/1.jpg").write_bytes((source / "images/0.jpg").read_bytes())
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            dt.assign_splits(dt.inventory(source))

    def test_dry_run_does_not_freeze(self):
        source = self.dataset("source", dt.NAMES, 20)
        args = Namespace(data=source, out=self.root / "split", frozen=self.root / "frozen",
                         dry_run=True, link_images=False)
        with redirect_stdout(io.StringIO()):
            dt.split(args)
        self.assertFalse(args.out.exists())
        self.assertFalse(args.frozen.exists())

    def test_rare_classes_keep_their_split_proportions(self):
        records = []
        for i in range(1000):
            classes = {0}
            classes.update(c for c, limit in [(1, 700), (2, 100), (3, 30)] if i < limit)
            records.append({'sha256': str(i), 'classes': classes})
        assigned = dt.assign_splits(records)
        for split, fraction in [('train', .7), ('val', .15), ('test', .15)]:
            for cls, total in [(1, 700), (2, 100), (3, 30)]:
                observed = sum(cls in row['classes'] for row in assigned[split])
                self.assertLessEqual(abs(observed - fraction * total), 2, (split, cls, observed))


if __name__ == '__main__':
    unittest.main()
