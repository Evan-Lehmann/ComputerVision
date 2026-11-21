"""Only annotate the verified selector; no desktop input is implemented."""
import unittest
from unittest.mock import Mock,patch
from pathlib import Path
import tempfile
import cv2
import numpy as np
from cardvision_webapp.chip_preview import locate_chip,template
from cardvision_webapp.app import create_app

class ChipPreviewTests(unittest.TestCase):
    def screen(self):
        image=np.full((900,1440,3),40,np.uint8)
        row=template();h,w=row.shape[:2];image[532:532+h,602:602+w]=row
        return image
    def test_selector_is_found_at_multiple_sizes_and_not_a_lone_red_circle(self):
        for scale in [.8,1.,1.4]:
            image=cv2.resize(self.screen(),None,fx=scale,fy=scale)
            match=locate_chip(image);self.assertIsNotNone(match)
            self.assertLess(abs(match['box'][0]-602*scale),5)
        blank=np.full((900,1440,3),40,np.uint8)
        cv2.circle(blank,(620,550),16,(0,0,255),-1)
        self.assertIsNone(locate_chip(blank))
    def test_route_off_by_default_and_uses_saved_frames_during_recording(self):
        with tempfile.TemporaryDirectory() as folder:
            capture=Mock();capture.thread=None;capture.preview.return_value=self.screen()
            app=create_app(capture=capture,artifact_root=folder);client=app.test_client()
            with patch('cardvision_webapp.chip_preview.locate_chip',wraps=locate_chip) as detect:
                off=client.get('/api/dashboard-preview');self.assertEqual(off.status_code,200)
                self.assertIsNone(off.json['highlight']);detect.assert_not_called()
                on=client.get('/api/dashboard-preview?highlight=1');self.assertIsNotNone(on.json['highlight'])
            cv2.imwrite(str(Path(folder)/'frame_000001.png'),self.screen())
            state=app.extensions['cardvision']['state'];state.update(session_status='RECORDING',round_dir=folder)
            capture.preview.reset_mock()
            frame=client.get('/api/dashboard-preview?highlight=1')
            self.assertEqual(frame.status_code,200);self.assertIn('Saved capture',frame.json['source'])
            capture.preview.assert_not_called()
            self.assertEqual(frame.headers['Cache-Control'],'no-store')
