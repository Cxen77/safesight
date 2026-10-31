import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from zones import ZoneMonitor, validate_points


class ZoneTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'zone.json'
        self.path.write_text(json.dumps({'points': [[.25,.25],[.75,.25],[.75,.75],[.25,.75]]}))
        self.now = 0.0
        self.monitor = ZoneMonitor((1000, 1000), self.path, clock=lambda: self.now)

    def update(self, rows, shape=(1000, 1000, 3)):
        return self.monitor.update(rows, shape)

    def test_feet_timer_exit_and_disappearance(self):
        person = (7, 450, 100, 550, 500)
        self.assertEqual(self.update([person])['duration_s'][7], 0)
        self.now = 4.2
        self.assertAlmostEqual(self.update([person])['duration_s'][7], 4.2)
        # Head is inside but feet below the polygon: no intrusion.
        self.assertEqual(self.update([(7,450,500,550,900)])['intruders'], [])
        self.now = 6
        self.assertEqual(self.update([person])['duration_s'][7], 0)
        self.update([])
        self.now = 10
        self.assertEqual(self.update([person])['duration_s'][7], 0)

    def test_crowd_boundary_and_resolution(self):
        rows = [(1, 450, 100, 550, 500), (2, 200, 100, 300, 250)]
        state = self.update(rows)
        self.assertEqual(state['crowd_count'], 2)
        self.assertEqual(state['proximity_margin'], 0)
        scaled = [(i, x1/2, y1/2, x2/2, y2/2) for i,x1,y1,x2,y2 in rows]
        self.assertEqual(self.update(scaled, (500,500,3))['intruders'], [1,2])

    def test_margin_is_safe_far_away_and_empty(self):
        self.assertEqual(self.update([])['proximity_margin'], 1)
        # Feet x=100; boundary x=250; distance=15% of width.
        self.assertAlmostEqual(self.update([(1,50,100,150,500)])['proximity_margin'], 1)
        self.assertAlmostEqual(self.update([(1,125,100,225,500)])['proximity_margin'], .5)
        self.assertEqual(self.update([(1,450,100,550,500)])['proximity_margin'], 0)

    def test_bad_polygons_and_duplicate_ids_are_rejected(self):
        for points in ([[0,0],[1,1]], [[0,0],[.5,.5],[1,1]],
                       [[0,0],[1,1],[0,1],[1,0]], [[0,0],[2,0],[1,1]]):
            with self.assertRaises(ValueError):
                validate_points(points)
        with self.assertRaises(ValueError):
            self.update([(1,0,0,10,10), (1,20,20,30,30)])


if __name__ == '__main__':
    unittest.main()
