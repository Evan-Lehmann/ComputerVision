"""Recorded pixels: animation/reflow must not multiply physical card identities."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import tempfile
import cv2
import numpy as np
from card_faces import reconcile_faces, face_rank_crop
from card_tracking import CardIdentityMatcher
from round_pipeline import crop_box,process_frames

class RecordedContinuityTests(unittest.TestCase):
    def test_recorded_pixels_have_equal_live_and_offline_results(self):
        root=Path(__file__).parent/'fixtures'
        data=json.loads((root/'face_continuity_recorded.json').read_text())
        with np.load(root/'face_continuity_recorded.npz',allow_pickle=False) as images:
            for case in data['cases']:
                with self.subTest(case=case['name']),tempfile.TemporaryDirectory() as folder:
                    folder=Path(folder);paths=[];boxes={}
                    for row in case['frames']:
                        p=folder/(row['key']+'.png');cv2.imwrite(str(p),images[row['key']])
                        paths.append(p);boxes[p.name]=row['boxes']
                    class Engine:
                        def detect(self,image,path):
                            return [dict(zip(('x1','y1','x2','y2'),b)) for b in boxes[path.name]]
                        def classify(self,crops):
                            # Deliberately identical predictions for every card:
                            # identity must come from pixels, paper and order.
                            p=np.full(13,.001);p[5]=.988
                            return [p.copy() for _ in crops]
                    kw=dict(engine=Engine(),regions=[dict(name='hand',rect=[0,0,1,1])],
                            shared_view=False,suit_reader=False,ocr_reader=False,reference_reader=False)
                    offline=process_frames(paths,folder/'offline',**kw)
                    snapshots=[]
                    live=process_frames([],folder/'live',frame_stream=iter(sorted(paths)),
                                        live_progress=snapshots.append,**kw)
                    self.assertEqual(live,offline)
                    self.assertEqual(len(live['events']),case['physical_cards'])
                    self.assertEqual(live['unresolved'],0)
                    self.assertTrue(all(len(r['events'])<=case['physical_cards'] for r in snapshots))

    def test_complete_paper_does_not_override_a_different_rank_or_ink_colour(self):
        from test_card_faces import canvas,paper,observation,candidate
        for old,new,red in [('K','3',False),('10','6',False),('3','3',True)]:
            with self.subTest(old=old,new=new,red=red):
                before,after=canvas(),canvas()
                fa=paper(before,160,text=old);fb=paper(after,130,text=new,red=red)
                o=observation(before,[160,200,177,222],fa,0)
                o['face_rank_crop']=face_rank_crop(before,o)
                t=SimpleNamespace(track_id=1,last_box=o['box'],last_sample_index=0,observations=[o])
                d=candidate(after,130);d['face_box']=fb
                from card_faces import secondary_rank
                d['secondary_crop']=secondary_rank(after,fb)
                matches=CardIdentityMatcher(ordered_hand=True).match([d],[t],1,image=after)
                self.assertFalse(matches)

    def test_order_and_identity_survive_animation_tint_and_compression(self):
        root=Path(__file__).parent/'fixtures'
        data=json.loads((root/'face_continuity_recorded.json').read_text())
        with np.load(root/'face_continuity_recorded.npz',allow_pickle=False) as images:
            for case in data['cases']:
                with self.subTest(case=case['name']):
                    matcher=CardIdentityMatcher(ordered_hand=True,clear_after_empty=4)
                    tracks=[];ordinal_ids={}
                    for row in case['frames']:
                        image=images[row['key']]
                        ds=[dict(box=b,crop=crop_box(image,b)) for b in row['boxes']]
                        ds,_=reconcile_faces(image,ds,58)
                        ds.sort(key=lambda d:d['box'][0])
                        matches=matcher.match(ds,tracks,row['frame'],image=image)
                        for i,d in enumerate(ds):
                            track=matches.get(i)
                            if track is None:
                                track=SimpleNamespace(track_id=len(tracks)+1,observations=[])
                                tracks.append(track)
                            track.last_box=d['box'];track.last_sample_index=row['frame']
                            track.observations.append(dict(d,frame=row['frame'],face_rank_crop=face_rank_crop(image,d)))
                            if i in ordinal_ids:self.assertEqual(track.track_id,ordinal_ids[i],(case['name'],row['frame'],i))
                            else:ordinal_ids[i]=track.track_id
                        self.assertEqual(len({matches[i].track_id for i in matches}),len(matches))
                    self.assertEqual(len(tracks),case['physical_cards'])
