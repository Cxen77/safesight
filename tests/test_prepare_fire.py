from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import unittest

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prepare_fire import prepare


class FirePreparationTests(unittest.TestCase):
    def test_invalid_pair_quarantine_duplicate_isolation_and_frozen_protection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, output, frozen = root / 'source', root / 'out', root / 'frozen'
            source.mkdir()
            config = {'names': ['smoke', 'fire']}
            for split in ('train', 'val', 'test'):
                config[split] = f'{split}/images'
                for folder in ('images', 'labels'):
                    (source / split / folder).mkdir(parents=True)
                (source / split / 'images/valid.jpg').write_bytes(split.encode())
                (source / split / 'labels/valid.txt').write_text('1 0.5 0.5 0.2 0.2\n')
                # Repeated bytes should remain only in the held-out split.
                (source / split / 'images/repeated.jpg').write_bytes(b'same-frame')
                (source / split / 'labels/repeated.txt').write_text('')
            bad_image = source / 'train/images/bad.jpg'
            bad_label = source / 'train/labels/bad.txt'
            bad_image.write_bytes(b'bad-annotation')
            bad_label.write_text('0 0.5 0.5 0 0.2\n')
            (source / 'data.yaml').write_text(yaml.safe_dump(config))
            with redirect_stdout(io.StringIO()):
                manifest = prepare(source, output, frozen)
            self.assertEqual(len(manifest['quarantined']), 1)
            self.assertEqual(len(manifest['duplicates_excluded']), 2)
            self.assertFalse((output / 'train/images/bad.jpg').exists())
            self.assertEqual(bad_label.read_text(), '0 0.5 0.5 0 0.2\n')
            self.assertTrue((output / 'test/images/repeated.jpg').exists())
            self.assertFalse((output / 'train/images/repeated.jpg').exists())
            self.assertNotIn('test', yaml.safe_load((output / 'train_only.yaml').read_text()))
            frozen_label = frozen / 'test/labels/valid.txt'
            before = frozen_label.read_bytes()
            (output / 'test/labels/valid.txt').write_text('changed-working-copy')
            self.assertEqual(frozen_label.read_bytes(), before)
            with self.assertRaisesRegex(ValueError, 'nonempty'):
                prepare(source, root / 'new-out', frozen)
            self.assertFalse((root / 'new-out').exists())


if __name__ == '__main__':
    unittest.main()
