from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from train_report import summarize


class ReportTests(unittest.TestCase):
    def metrics(self):
        return SimpleNamespace(box=SimpleNamespace(ap_class_index=[2, 0, 1],
            p=[.7, .9, .8], r=[.49, .8, .5], ap50=[.6, .9, .7], map50=.73, map=.51))

    def test_class_alignment_missing_class_and_recommendation_boundaries(self):
        report = summarize(self.metrics(), {0: 'person', 1: 'head', 2: 'helmet', 3: 'vest'})
        self.assertEqual(report['per_class'][0]['recall'], .8)
        self.assertEqual(report['per_class'][2]['map50'], .6)
        self.assertIsNone(report['per_class'][3]['recall'])
        recommendations = '\n'.join(report['weak_spots'])
        self.assertIn('helmet: recall 0.490; if possible, collect 150', recommendations)
        self.assertIn('head: recall 0.500; if possible, collect 75', recommendations)
        self.assertIn('vest: no per-class result', recommendations)

    def test_invalid_metrics_cannot_be_published(self):
        metrics = self.metrics()
        metrics.box.map50 = float('nan')
        with self.assertRaises(ValueError):
            summarize(metrics, {0: 'person'})


if __name__ == '__main__':
    unittest.main()
