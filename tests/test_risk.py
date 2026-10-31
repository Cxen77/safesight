from itertools import product
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import risk


class RiskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = risk.get_engine()

    def test_guide_examples(self):
        for inputs, bands in [((0,0,0,0,0), {'Low'}), ((.9,.9,.5,0,0), {'High','Critical'}),
                              ((0,0,0,.8,0), {'Critical'}), ((.5,.5,.5,0,4), {'High','Critical'}),
                              ((.2,.1,.1,0,0), {'Low'})]:
            with self.subTest(inputs=inputs):
                score, band = risk.compute_risk(*inputs)
                self.assertIn(band, bands)
                self.assertTrue(0 <= score <= 100)

    def test_overrides_have_exact_boundaries_and_do_not_lower_existing_score(self):
        self.assertEqual(risk.compute_risk(0,0,0,.7,0), (100, 'Critical'))
        self.assertLess(risk.compute_risk(0,0,0,.699999,0)[0], 100)
        self.assertLess(risk.compute_risk(0,0,0,0,2.999999)[0], 65)
        self.assertEqual(risk.compute_risk(0,0,0,0,3), (65, 'High'))
        self.assertEqual(risk.compute_risk(0,0,0,0,8), (85, 'Critical'))
        with patch('risk.get_engine') as get:
            get.return_value.fuzzy_score.return_value = 92
            self.assertEqual(risk.compute_risk(0,0,0,0,3), (92, 'Critical'))
            self.assertEqual(risk.compute_risk(0,0,0,0,8), (92, 'Critical'))
        with patch('risk.get_engine', side_effect=AssertionError('Must short-circuit fire')):
            self.assertEqual(risk.compute_risk(0,0,0,.9,0), (100, 'Critical'))

    def test_full_rule_coverage_and_ordinal_monotonicity(self):
        self.assertEqual(len(list(self.engine.stage1.rules)), 27)
        self.assertEqual(len(list(self.engine.stage2.rules)), 9)
        for inputs in product(range(3), repeat=3):
            conclusion = risk.stage1_conclusion(*inputs)
            for axis in range(3):
                higher = list(inputs)
                higher[axis] = min(2, higher[axis] + 1)
                self.assertGreaterEqual(risk.stage1_conclusion(*higher), conclusion)
        expected = [['Low','Medium','High'], ['Medium','High','High'], ['High','Critical','Critical']]
        for h, f in product(range(3), repeat=2):
            score = self.engine.risk_table[h * 10, f * 10]
            self.assertEqual(risk.risk_band(float(score)), expected[h][f])

    def test_surfaces_and_continuous_outputs_are_monotonic(self):
        for table in (self.engine.hazard_table, self.engine.risk_table):
            for axis in range(table.ndim):
                self.assertTrue(np.all(np.diff(table, axis=axis) >= -1e-10))
        rng = np.random.default_rng(42)
        for _ in range(300):
            values = [*rng.random(4), float(rng.uniform(0, 10))]
            baseline = risk.compute_risk(*values)[0]
            for axis in range(5):
                higher = values.copy()
                ceiling = 10 if axis == 4 else 1
                higher[axis] += (ceiling - higher[axis]) * rng.random()
                self.assertGreaterEqual(risk.compute_risk(*higher)[0] + 1e-9, baseline)
            higher = values.copy()
            higher[:3] = [x + (1-x) * rng.random() for x in higher[:3]]
            self.assertGreaterEqual(self.engine.hazard(*higher[:3]) + 1e-9,
                                    self.engine.hazard(*values[:3]))

    def test_invalid_inputs_are_rejected_even_with_fire_override(self):
        for axis in range(5):
            for bad in (float('nan'), float('inf'), -1):
                inputs = [0,0,0,.9,0]
                inputs[axis] = bad
                with self.assertRaises(ValueError):
                    risk.compute_risk(*inputs)
        with self.assertRaises(ValueError):
            risk.compute_risk(1.01,0,0,0,0)

    def test_band_boundaries_for_continuous_scores(self):
        for score, expected in [(0,'Low'), (25,'Low'), (25.00001,'Medium'), (50,'Medium'),
                                (50.00001,'High'), (75,'High'), (75.00001,'Critical'), (100,'Critical')]:
            self.assertEqual(risk.risk_band(score), expected)


if __name__ == '__main__':
    unittest.main()
