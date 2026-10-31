import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from app import SafetyPipeline, Settings, create_app


class DashboardTests(unittest.TestCase):
    def test_worker_failure_never_reports_a_healthy_low_score(self):
        with tempfile.TemporaryDirectory() as temp:
            pipeline=SafetyPipeline(Settings(demo=True,silent=True,output_dir=Path(temp)))
            with patch.object(pipeline,'demo_frame',side_effect=RuntimeError('Source unavailable')):
                pipeline.run()
            state=pipeline.snapshot()
            self.assertEqual(state['status'],'error')
            self.assertEqual(state['error'],'Source unavailable')
            self.assertIsNone(state['score'])
            self.assertIsNone(state['band'])
            self.assertTrue(pipeline.stop_event.is_set())

    def test_routes_unknown_initial_state_stream_and_snapshot_isolation(self):
        with tempfile.TemporaryDirectory() as temp:
            pipeline=SafetyPipeline(Settings(demo=True,silent=True,output_dir=Path(temp)))
            client=create_app(pipeline).test_client()
            response=client.get('/')
            self.assertEqual(response.status_code,200)
            self.assertIn(b'Pause display',response.data)
            state=client.get('/api/state').get_json()
            self.assertEqual(state['mode'],'demo')
            self.assertIsNone(state['score'])
            self.assertFalse(state['detectors_active'])
            pipeline.publish(frame=np.zeros((20,30,3),dtype=np.uint8))
            stream=client.get('/video_feed',buffered=False)
            self.assertIn(b'Content-Type: image/jpeg',next(stream.response))
            stream.close()
            self.assertEqual(client.get('/snapshots/../events.csv').status_code,404)

    def test_live_pipeline_time_slices_fire_and_uses_real_module_contracts(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            zone=root/'zones.json'
            zone.write_text(json.dumps({'points':[[0,0],[1,0],[1,1],[0,1]]}))
            frame=np.zeros((360,640,3),dtype=np.uint8)
            def box(cls,coords,track_id=None,conf=.8):
                return SimpleNamespace(cls=np.array(cls),xyxy=np.array([coords]),
                    id=np.array(track_id) if track_id is not None else None,conf=np.array(conf))
            detection=SimpleNamespace(boxes=[box(0,[100,10,200,300],7),box(1,[130,10,170,50],8)],
                                      plot=lambda:frame.copy())
            ppe=Mock(names={0:'person',1:'head',2:'helmet',3:'vest'})
            ppe.track.return_value=[detection]
            fire=Mock(names={0:'smoke',1:'fire'})
            fire.predict.return_value=[SimpleNamespace(boxes=[box(1,[400,20,450,80])])]
            cap=Mock()
            cap.isOpened.return_value=True
            cap.read.side_effect=[(True,frame.copy()) for _ in range(11)]
            modules={'ultralytics':SimpleNamespace(YOLO=lambda path:ppe if path.endswith('ppe.pt') else fire)}
            pipeline=SafetyPipeline(Settings(ppe_weights=Path('ppe.pt'),fire_weights=Path('fire.pt'),
                zones=zone,silent=True,output_dir=root/'events',max_frames=11))
            with patch.dict(sys.modules,modules),patch('app.cv2.VideoCapture',return_value=cap):
                pipeline.run()
            state=pipeline.snapshot()
            self.assertIsNone(state['error'])
            self.assertEqual(state['status'],'stopped')
            self.assertEqual(ppe.track.call_count,11)
            self.assertEqual(fire.predict.call_count,2)
            self.assertEqual(state['crowd_count'],1)
            self.assertEqual(state['raw_band'],'Critical')
            self.assertEqual(state['fire_evidence'],.8)
            cap.release.assert_called_once()


if __name__=='__main__':
    unittest.main()
