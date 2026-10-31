import csv
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from alerts import AlertDriver, AlertPolicy, EventStore, EVENT_COLUMNS


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.policy = AlertPolicy(clock=lambda: self.now)

    def test_flicker_is_suppressed_and_escalation_logs_once(self):
        self.assertTrue(self.policy.update(65, 'High').pending)
        self.now = .9
        self.assertIsNone(self.policy.update(65, 'High').event_band)
        self.now = 1
        self.assertEqual(self.policy.update(65, 'High').event_band, 'High')
        self.now = 2
        self.assertIsNone(self.policy.update(65, 'High').event_band)
        critical = self.policy.update(100, 'Critical')
        self.assertEqual(critical.event_band, 'Critical')
        self.assertTrue(critical.latched)
        self.assertIsNone(self.policy.update(100, 'Critical').event_band)

    def test_critical_requires_confirmation_then_three_clear_seconds(self):
        self.policy.update(100, 'Critical')
        self.now = .5
        self.assertFalse(self.policy.update(12, 'Low').latched)
        self.now = 1
        self.policy.update(100, 'Critical')
        self.now = 2
        self.assertTrue(self.policy.update(100, 'Critical').latched)
        self.now = 3
        self.assertEqual(self.policy.update(12, 'Low').score, 100)
        self.now = 5.99
        self.assertTrue(self.policy.update(38, 'Medium').latched)
        self.now = 6
        self.assertFalse(self.policy.update(38, 'Medium').latched)

    def test_high_and_unknown_frames_interrupt_latch_clearing(self):
        self.policy.update(85, 'Critical')
        self.now = 1
        self.policy.update(85, 'Critical')
        self.now = 2
        self.policy.update(12, 'Low')
        self.now = 4
        self.policy.update(65, 'High')
        self.now = 5
        self.policy.update(12, 'Low')
        self.policy.interrupt_observation()
        self.now = 9
        self.assertTrue(self.policy.update(12, 'Low').latched)
        self.now = 12
        self.assertFalse(self.policy.update(12, 'Low').latched)

    def test_high_to_critical_keeps_shared_debounce_and_invalid_bands_fail(self):
        self.policy.update(65, 'High')
        self.now = .6
        self.policy.update(85, 'Critical')
        self.now = 1
        self.assertEqual(self.policy.update(85, 'Critical').event_band, 'Critical')
        with self.assertRaises(ValueError):
            self.policy.update(90, 'Low')


class StorageAndDriverTests(unittest.TestCase):
    def test_snapshots_csv_and_restart_history_are_consistent(self):
        with tempfile.TemporaryDirectory() as temp:
            store = EventStore(temp)
            inputs = dict(ppe_load=1,zone_pressure=.5,proximity=1,fire_evidence=0)
            frame = np.zeros((20,30,3),dtype=np.uint8)
            first = store.record('High',65,inputs,'bare head',frame)
            second = store.record('Critical',85,inputs,'long intrusion',frame)
            self.assertNotEqual(first['snapshot'],second['snapshot'])
            self.assertIsNotNone(cv2.imread(str(store.snapshots/first['snapshot'])))
            with store.csv_path.open(newline='') as stream:
                rows=list(csv.reader(stream))
            self.assertEqual(tuple(rows[0]),EVENT_COLUMNS)
            self.assertEqual(len(rows),3)
            self.assertEqual(EventStore(temp).recent()[0]['snapshot'],second['snapshot'])
            with patch('alerts.cv2.imwrite',return_value=False):
                with self.assertRaises(RuntimeError):
                    store.record('High',65,inputs,'write failure',frame)
            self.assertEqual(len(store.recent()),2)

    def test_outputs_change_only_and_hardware_cleanup(self):
        sound = SimpleNamespace(PlaySound=Mock(),SND_ALIAS=1,SND_ASYNC=2)
        with patch('alerts.importlib.import_module',return_value=sound):
            driver=AlertDriver()
            driver.set_level('High')
            driver.set_level('High')
            driver.set_level('Critical')
            self.assertEqual(sound.PlaySound.call_count,2)
        gpio=Mock(BOARD=10,OUT=0,LOW=0,HIGH=1)
        with patch('alerts.importlib.import_module',return_value=gpio):
            driver=AlertDriver(hardware=True)
            driver.set_level('Critical')
            driver.set_level('Critical')
            self.assertEqual(gpio.output.call_count,1)
            driver.close()
            driver.close()
            gpio.cleanup.assert_called_once_with((12,16))
            with self.assertRaises(ValueError):
                driver.set_level('High')


if __name__=='__main__':
    unittest.main()
