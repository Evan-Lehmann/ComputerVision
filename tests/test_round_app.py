import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
import cv2
import numpy as np
from cardvision_webapp.app import create_app
from cardvision_webapp.round_capture import RoundCapture
from round_pipeline import summarize, save_result

class FakeCapture:
    def __init__(self, root, failure=None):
        self.root=root;self.failure=failure;self.error=None;self.frame_count=3;self.thread=None;self.recording=False
    def start(self):
        if self.failure:raise RuntimeError(self.failure)
        self.recording=True;return self.root
    def stop(self):
        self.recording=False;return dict(round_dir=str(self.root),frames=3)
    def preview(self):return np.zeros((10,10,3),np.uint8)

class RoundAppTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        cv2.imwrite(str(self.root/'frame.png'),np.ones((10,10,3),np.uint8))
        self.block=threading.Event();self.block.set()
        self.capture=FakeCapture(self.root)
        def processor(directory,output_dir,regions,progress):
            self.block.wait(2)
            output_dir.mkdir(parents=True,exist_ok=True)
            progress(1,1)
            cv2.imwrite(str(output_dir/'card_first.png'),np.full((10,10,3),240,np.uint8))
            result=summarize(dict(events=[dict(id='1',rank='UNKNOWN',excluded=False,evidence=[dict(image='card_first.png',frame=0,source='frame.png')],region='player',observations=1)],review_confirmed=False,warnings=[]))
            save_result(result,output_dir);return result
        self.processor=processor
        self.app=create_app(capture=self.capture,processor=processor,replays={'sample':self.root},artifact_root=self.root/'artifacts',delay=0)
        self.client=self.app.test_client()
    def tearDown(self):self.block.set();self.temp.cleanup()
    def wait(self,status):
        deadline=time.monotonic()+3
        while time.monotonic()<deadline:
            state=self.client.get('/api/state').json
            if state['session_status']==status:return state
            time.sleep(.01)
        self.fail('State did not reach '+status+': '+str(state))
    def test_record_stop_process_review_and_export(self):
        self.assertEqual(self.client.get('/').status_code,200)
        self.assertEqual(self.client.post('/api/start',json={}).status_code,202)
        self.wait('RECORDING')
        self.assertEqual(self.client.post('/api/start',json={}).status_code,409)
        self.assertEqual(self.client.post('/api/stop',json={}).status_code,202)
        state=self.wait('COMPLETE')
        self.assertIsNone(state['count_change'])
        reviewed=self.client.post('/api/review',json=dict(result_id=state['result_id'],events=[dict(id='1',rank='4')],confirmed=True))
        self.assertEqual(reviewed.status_code,200)
        self.assertEqual(reviewed.json['count_change'],1)
        self.assertTrue(reviewed.json['count_verified'])
        download=self.client.get('/api/result')
        self.assertEqual(download.status_code,200)
        self.assertTrue(json.loads(download.data)['count_verified'])
    def test_replay_and_busy_reset_race(self):
        self.block.clear()
        self.assertEqual(self.client.post('/api/process',json={'source':'sample'}).status_code,202)
        self.wait('PROCESSING')
        self.assertEqual(self.client.post('/api/reset',json={}).status_code,409)
        self.assertEqual(self.client.post('/api/process',json={'source':'sample'}).status_code,409)
        self.block.set();state=self.wait('COMPLETE')
        self.assertEqual(self.client.get('/api/frame/'+state['result_id']+'/0').status_code,200)
        self.assertEqual(self.client.get('/api/frame/wrong/0').status_code,404)
    def test_failed_capture_becomes_visible_error_and_can_reset(self):
        self.capture.failure='Screen permission denied'
        self.client.post('/api/start',json={})
        state=self.wait('ERROR')
        self.assertIn('permission',state['error'])
        self.assertEqual(self.client.post('/api/reset',json={}).json['session_status'],'IDLE')
    def test_invalid_and_cross_origin_requests_are_rejected(self):
        self.assertEqual(self.client.post('/api/process',json={'source':'/etc'}).status_code,400)
        self.assertEqual(self.client.post('/api/process',json={'source':[]}).status_code,400)
        self.assertEqual(self.client.post('/api/process',json=[]).status_code,400)
        self.assertEqual(self.client.post('/api/start',json={},headers={'Origin':'https://example.com'}).status_code,403)
        self.assertEqual(self.client.post('/api/stop',json={}).status_code,409)
    def test_reviewed_crop_download_requires_saved_confirmation_and_can_be_revoked(self):
        import io,zipfile,csv
        self.client.post('/api/process',json={'source':'sample'});state=self.wait('COMPLETE')
        self.assertEqual(self.client.get('/api/reviewed-crops').status_code,409)
        payload=dict(result_id=state['result_id'],events=[dict(id='1',rank='3')],confirmed=True)
        self.assertEqual(self.client.post('/api/review',json=payload).status_code,200)
        response=self.client.get('/api/reviewed-crops')
        self.assertEqual(response.status_code,200)
        with zipfile.ZipFile(io.BytesIO(response.data)) as archive:
            rows=list(csv.DictReader(io.StringIO(archive.read('labels.csv').decode())))
            self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]['rank'],'3')
            self.assertIn('crops/card_first.png',archive.namelist())
        payload['confirmed']=False
        self.client.post('/api/review',json=payload)
        self.assertEqual(self.client.get('/api/reviewed-crops').status_code,409)

    def test_stale_review_does_not_modify_current_result(self):
        self.client.post('/api/process',json={'source':'sample'});self.wait('COMPLETE')
        self.assertEqual(self.client.post('/api/review',json={'result_id':'old','events':[]}).status_code,409)

class CaptureTests(unittest.TestCase):
    def test_real_thread_saves_first_frame_before_ready(self):
        class Screen:
            monitors=[{},{}]
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def grab(self,monitor):return np.zeros((10,10,4),np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            capture=RoundCapture(rounds_dir=directory,screen_factory=Screen,fps=100)
            folder=capture.start()
            self.assertGreaterEqual(capture.frame_count,1)
            result=capture.stop()
            self.assertEqual(len(list(folder.glob('frame_*.png'))),result['frames'])
            self.assertFalse(capture.recording)
    def test_monitor_startup_error_is_propagated(self):
        class Screen:
            monitors=[{}]
            def __enter__(self):return self
            def __exit__(self,*args):pass
        with tempfile.TemporaryDirectory() as directory:
            capture=RoundCapture(rounds_dir=directory,screen_factory=Screen)
            with self.assertRaisesRegex(RuntimeError,'unavailable'):capture.start()
            self.assertFalse(capture.recording)

if __name__=='__main__':unittest.main()
