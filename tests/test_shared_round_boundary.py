"""Assign every supported view after one common round-boundary decision."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from round_pipeline import process_frames
from shared_cards import SharedMonitor

class SharedBoundaryTests(unittest.TestCase):
    def replay(self,primary_frames,shared_frames,total,streamed=False,shared=True):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);paths=[];seen=[]
            for f in range(total):
                image=np.full((100,200,3),(60,100,50),np.uint8);image[0,0]=f
                if f in primary_frames:
                    image[20:60,20:50]=255
                    cv2.putText(image,'6',(21,40),0,.5,(0,0,200),1)
                image[10:40,120:155]=255
                p=root/f'frame_{f:03d}.png';cv2.imwrite(str(p),image);paths.append(p)
            class Engine:
                def detect(self,image,path):
                    return [dict(x1=21,y1=22,x2=34,y2=41)] if int(image[0,0,0]) in primary_frames else []
                def classify(self,crops):
                    p=np.full(13,.001);p[5]=.988
                    return [p.copy() for _ in crops]
            def hand(image,rect):
                return dict(count=1,box=[120,10,155,40],starts=[120]) if int(image[0,0,0]) in shared_frames else None
            original=SharedMonitor.observe
            def observe(self,image,frame,source,round_number,**kw):
                seen.append((frame,round_number));return original(self,image,frame,source,round_number,**kw)
            view=dict(name='central',rect=[.6,0,.9,.5],linked_region='hand') if shared else False
            with patch('round_pipeline.mini_hand',side_effect=hand,create=True),patch('shared_cards.mini_hand',side_effect=hand),patch.object(SharedMonitor,'observe',observe):
                snapshots=[]
                result=process_frames([] if streamed else paths,root/'out',engine=Engine(),
                    regions=[dict(name='hand',rect=[0,0,.5,1])],shared_view=view,
                    mini_reader=False,suit_reader=False,ocr_reader=False,reference_reader=False,
                    frame_stream=iter(paths) if streamed else None,
                    live_progress=snapshots.append if streamed else None)
            return result,seen,snapshots

    def test_early_shared_card_starts_round_before_primary_detection(self):
        args=({0,1,7,8},{0,1,6,7,8},9)
        offline,seen,_=self.replay(*args)
        live,live_seen,snapshots=self.replay(*args,streamed=True)
        self.assertEqual(live,offline);self.assertEqual(seen,live_seen)
        self.assertEqual(offline['boundaries'][0]['first_frame'],6)
        self.assertIn((6,2),seen)
        self.assertEqual([e['round_number'] for e in offline['events'] if e['region']=='hand'],[1,2])
        self.assertEqual(next(r for r in snapshots if r['frames_processed']==7)['boundaries'][0]['first_frame'],6)
        slots=offline['coverage']['slots']
        self.assertEqual([(s['round_number'],s['first_frame'],s['last_frame']) for s in slots],[(1,0,1),(2,6,8)])

    def test_visible_shared_hand_prevents_false_empty_table_boundary(self):
        result,seen,_=self.replay({0,1,6,7},set(range(8)),8)
        self.assertEqual(result['boundaries'],[])
        self.assertTrue(all(number==1 for _,number in seen))

    def test_disabling_shared_view_keeps_primary_only_boundary(self):
        result,seen,_=self.replay({0,1,7,8},{0,1,6,7,8},9,shared=False)
        self.assertEqual(result['boundaries'][0]['first_frame'],7)
        self.assertEqual(seen,[])
