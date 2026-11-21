"""Live worker regressions: identity continuity, no frame loss, and drain ownership."""
import copy
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from cardvision_webapp.app import create_app
from cardvision_webapp.round_capture import RoundCapture
from round_pipeline import process_frames
from test_round_pipeline import Engine, REGIONS


class Screen:
    monitors=[{},dict(left=0,top=0,width=100,height=100),dict(left=100,top=0,width=100,height=100)]
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def grab(self,monitor):
        image=np.full((100,100,4),255,np.uint8)
        cv2.putText(image,'6',(20,37),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0,255),1)
        return image


class LiveTests(unittest.TestCase):
    def wait(self, client, predicate):
        deadline=time.monotonic()+8
        while time.monotonic()<deadline:
            state=client.get('/api/state').json
            if predicate(state):return state
            time.sleep(.01)
        self.fail(str(state))

    def test_streaming_matches_offline_without_mutating_active_tracks(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);paths=[]
            for i in range(8):
                image=Screen().grab({})[:,:,:3].copy()
                path=root/('frame_%06d.png'%i);cv2.imwrite(str(path),image);paths.append(path)
            snapshots=[]
            offline=process_frames(paths,root/'offline',Engine(),REGIONS,shared_view=False)
            streamed=process_frames([],root/'stream',Engine(),REGIONS,shared_view=False,
                frame_stream=iter(paths),live_progress=lambda r:snapshots.append(copy.deepcopy(r)))
            self.assertEqual(offline,streamed)
            self.assertEqual(len(snapshots),8)
            self.assertEqual(snapshots[0]['cards'],['UNKNOWN'])
            self.assertEqual(snapshots[-1]['cards'],['6'])
            self.assertEqual({r['events'][0]['id'] for r in snapshots},{'1'})
            self.assertEqual([r['known_count_change'] for r in snapshots],[0]+[1]*7)
            self.assertTrue(all(not r['count_verified'] for r in snapshots))

    def test_display_selection_live_updates_and_stop_drains_every_saved_frame(self):
        with tempfile.TemporaryDirectory() as root:
            capture=RoundCapture(fps=30,rounds_dir=root,screen_factory=Screen)
            blocked=threading.Event();release=threading.Event()
            def pipeline(paths,output,**kwargs):
                stream=kwargs.pop('frame_stream')
                def delayed():
                    for index,path in enumerate(stream):
                        if index==2:
                            blocked.set();release.wait(8)
                        yield path
                return process_frames(paths,output,engine=Engine(),frame_stream=delayed(),**kwargs)
            with patch('cardvision_webapp.app.process_frames',side_effect=pipeline):
                app=create_app(capture=capture,artifact_root=Path(root)/'out',delay=0)
                client=app.test_client()
                client.post('/api/regions',json=dict(regions=REGIONS,shared_view=False))
                self.assertEqual(client.get('/api/displays').json['selected'],1)
                self.assertEqual(client.post('/api/display',json={'index':True}).status_code,400)
                self.assertEqual(client.post('/api/display',json={'index':3}).status_code,400)
                self.assertEqual(client.post('/api/display',json={'index':2}).status_code,200)
                try:
                    client.post('/api/start',json={})
                    live=self.wait(client,lambda s:s['processed']>=2)
                    self.assertEqual(live['cards'],['6'])
                    self.assertEqual(live['session_status'],'RECORDING')
                    self.assertTrue(blocked.wait(2))
                    time.sleep(.1)
                    lag=client.get('/api/state').json
                    self.assertGreater(lag['pending_frames'],0)
                    self.assertGreater(lag['processing_lag_seconds'],0)
                    self.assertEqual(client.post('/api/stop',json={}).status_code,202)
                    self.wait(client,lambda s:not capture.recording)
                    self.assertEqual(client.get('/api/state').json['session_status'],'STOPPING')
                    for url,payload in [('/api/start',{}),('/api/reset',{}),('/api/display',{'index':1}),('/api/review',{'result_id':live['result_id']})]:
                        self.assertEqual(client.post(url,json=payload).status_code,409)
                    self.assertEqual(client.get('/api/result').status_code,404)
                    release.set()
                    final=self.wait(client,lambda s:s['session_status']=='COMPLETE')
                    self.assertEqual(final['frames_processed'],capture.frame_count)
                    self.assertEqual(final['cards'],['6'])
                    self.assertEqual(final['known_count_change'],1)
                finally:
                    release.set()
                    if capture.recording:capture.stop()
                    self.wait(client,lambda s:s['session_status'] in {'COMPLETE','ERROR'})

    def test_cancel_countdown_does_not_capture(self):
        with tempfile.TemporaryDirectory() as root:
            capture=RoundCapture(rounds_dir=root,screen_factory=Screen)
            app=create_app(capture=capture,artifact_root=root,delay=3)
            client=app.test_client()
            client.post('/api/start',json={})
            self.assertEqual(client.post('/api/stop',json={}).status_code,202)
            self.wait(client,lambda s:s['session_status']=='IDLE')
            self.assertIsNone(capture.thread)

    def test_inference_failure_stops_capture_and_surfaces_error(self):
        with tempfile.TemporaryDirectory() as root:
            capture=RoundCapture(fps=30,rounds_dir=root,screen_factory=Screen)
            with patch('cardvision_webapp.app.process_frames',side_effect=RuntimeError('inference broke')):
                client=create_app(capture=capture,artifact_root=root,delay=0).test_client()
                client.post('/api/start',json={})
                state=self.wait(client,lambda s:s['session_status']=='ERROR')
                self.assertIn('inference broke',state['error'])
                self.assertFalse(capture.thread.is_alive())
                self.assertGreater(capture.frame_count,0)

    def test_round_clear_retires_identity_but_keeps_prior_count(self):
        class RoundEngine(Engine):
            def detect(self,image,path):
                return [] if 2<=int(path.stem.split('_')[-1])<=5 else super().detect(image,path)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);paths=[]
            for i in range(8):
                image=Screen().grab({})[:,:,:3].copy() if i<2 or i>5 else np.full((100,100,3),(25,90,25),np.uint8)
                path=root/('frame_%06d.png'%i);cv2.imwrite(str(path),image);paths.append(path)
            snapshots=[]
            final=process_frames([],root/'out',RoundEngine(),REGIONS,shared_view=False,
                frame_stream=iter(paths),live_progress=lambda r:snapshots.append(r))
            self.assertEqual(final['cards'],['6','6'])
            self.assertEqual([e['round_number'] for e in final['events']],[1,2])
            self.assertEqual([r['running_known_count_change'] for r in final['rounds']],[1,2])
            self.assertEqual([r['known_count_change'] for r in snapshots],[0,1,1,1,1,1,1,2])
            self.assertNotEqual(final['events'][0]['id'],final['events'][1]['id'])

    def test_capture_error_drains_frames_and_keeps_replay_folder(self):
        class FailingScreen(Screen):
            def __init__(self):self.calls=0
            def grab(self,monitor):
                self.calls+=1
                if self.calls==6:raise RuntimeError('display disconnected')
                return super().grab(monitor)
        with tempfile.TemporaryDirectory() as root:
            capture=RoundCapture(fps=30,rounds_dir=root,screen_factory=FailingScreen)
            def pipeline(paths,output,**kwargs):
                return process_frames(paths,output,engine=Engine(),**kwargs)
            with patch('cardvision_webapp.app.process_frames',side_effect=pipeline):
                client=create_app(capture=capture,artifact_root=Path(root)/'out',delay=0).test_client()
                client.post('/api/regions',json=dict(regions=REGIONS,shared_view=False))
                client.post('/api/start',json={})
                final=self.wait(client,lambda s:s['session_status']=='ERROR')
                self.assertEqual(final['frames_processed'],5)
                self.assertIn('display disconnected',final['error'])
                self.assertEqual(len(list(capture.round_dir.glob('frame_*.png'))),5)

    def test_tile_batching_preserves_offsets_and_ids(self):
        from test_rank_detector_unseen import detect_full_frame
        import torch
        from types import SimpleNamespace
        class Detector:
            def __init__(self):self.calls=0
            def predict(self,source,**kwargs):
                self.calls+=1
                return [SimpleNamespace(boxes=[SimpleNamespace(conf=torch.tensor([.9]),
                    xyxy=torch.tensor([[20.,20.,30.,40.]]))]) for _ in (source if isinstance(source,list) else [source])]
        image=np.zeros((900,1440,3),np.uint8)
        one,batched=Detector(),Detector()
        self.assertEqual(detect_full_frame(one,image),detect_full_frame(batched,image,tile_batch_size=4))
        self.assertLess(batched.calls,one.calls)
        with self.assertRaises(ValueError):detect_full_frame(one,image,tile_batch_size=0)

    def test_stop_timeout_keeps_session_owned_and_can_retry(self):
        class SlowStop(RoundCapture):
            attempts=0
            def stop(self):
                self.attempts+=1
                if self.attempts==1:raise RuntimeError('Capture has not stopped yet; please retry.')
                return super().stop()
        with tempfile.TemporaryDirectory() as root:
            capture=SlowStop(fps=20,rounds_dir=root,screen_factory=Screen)
            def pipeline(paths,output,**kwargs):
                return process_frames(paths,output,engine=Engine(),**kwargs)
            with patch('cardvision_webapp.app.process_frames',side_effect=pipeline):
                client=create_app(capture=capture,artifact_root=Path(root)/'out',delay=0).test_client()
                client.post('/api/regions',json=dict(regions=REGIONS,shared_view=False))
                client.post('/api/start',json={})
                self.wait(client,lambda s:s['processed']>=2)
                client.post('/api/stop',json={})
                self.wait(client,lambda s:s['session_status']=='STOPPING' and s['error'])
                self.assertEqual(client.post('/api/start',json={}).status_code,409)
                self.assertEqual(client.post('/api/reset',json={}).status_code,409)
                self.assertEqual(client.post('/api/stop',json={}).status_code,202)
                final=self.wait(client,lambda s:s['session_status']=='COMPLETE')
                self.assertEqual(final['frames_processed'],capture.frame_count)
                self.assertEqual(final['pending_frames'],0)
                self.assertIsNone(final['error'])
