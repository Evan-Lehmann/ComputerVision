"""Synthetic paper/corner tests; no uploaded screenshots are published."""
from types import SimpleNamespace
import unittest
import cv2
import numpy as np
from card_faces import (reconcile_faces, secondary_rank, secondary_similarity,
                        secondary_animation_similarity, projected_rank_crop)
from card_tracking import CardIdentityMatcher, merge_track_fragments
from round_pipeline import crop_box


def canvas():
    return np.full((600,800,3),35,np.uint8)


def paper(image,x=100,y=200,text='7',red=False,opacity=1):
    face=np.full((60,40,3),255,np.uint8)
    ink=(20,25,190) if red else (15,15,15)
    cv2.putText(face,text,(2,20),cv2.FONT_HERSHEY_SIMPLEX,.6,ink,1,cv2.LINE_AA)
    face[36:60,18:40]=cv2.rotate(face[:24,:22].copy(),cv2.ROTATE_180)
    image[y:y+60,x:x+40]=cv2.addWeighted(face,opacity,image[y:y+60,x:x+40],1-opacity,0)
    return [x,y,x+40,y+60]


def candidate(image,x=100,y=200,width=17):
    box=[x,y,x+width,y+22]
    return dict(box=box,crop=crop_box(image,box))


def observation(image,box,face,frame):
    return dict(frame=frame,box=box,crop=crop_box(image,box),face_box=face,
                secondary_crop=secondary_rank(image,face))


class CardFaceTests(unittest.TestCase):
    def test_contained_clipped_box_is_suppressed_with_a_retained_counterpart_and_audit(self):
        image=canvas();paper(image)
        full=candidate(image);partial=candidate(image,width=6)
        kept,audit=reconcile_faces(image,[partial,full])
        self.assertEqual(len(kept),1)
        self.assertEqual(kept[0]['box'],full['box'])
        self.assertEqual(audit[0]['box'],partial['box'])
        self.assertEqual(audit[0]['retained_box'],full['box'])

    def test_only_candidate_is_never_deleted_even_if_narrow(self):
        image=canvas();paper(image)
        kept,audit=reconcile_faces(image,[candidate(image,width=6)])
        self.assertEqual(len(kept),1);self.assertFalse(audit)

    def test_equal_rank_cards_on_separate_faces_remain_separate(self):
        image=canvas();paper(image,100);paper(image,145)
        kept,audit=reconcile_faces(image,[candidate(image,100),candidate(image,145)])
        self.assertEqual(len(kept),2);self.assertFalse(audit)

    def test_overlapping_cards_with_visible_seams_remain_separate(self):
        image=canvas()
        for x in [100,119,138]:
            paper(image,x)
            image[200:260,x]=210
        kept,audit=reconcile_faces(image,[candidate(image,x) for x in [100,119,138]])
        self.assertEqual(len(kept),3);self.assertFalse(audit)

    def test_two_full_candidates_are_not_deleted_just_because_one_face_was_inferred(self):
        image=canvas();paper(image)
        kept,audit=reconcile_faces(image,[candidate(image),candidate(image,118)])
        self.assertEqual(len(kept),2);self.assertFalse(audit)

    def test_face_reconciliation_scales_with_the_recording_resolution(self):
        for scale in [.75,1.5]:
            image=canvas();paper(image)
            image=cv2.resize(image,None,fx=scale,fy=scale,interpolation=cv2.INTER_NEAREST)
            ds=[]
            for width in [17,6]:
                box=[100*scale,200*scale,(100+width)*scale,222*scale]
                ds.append(dict(box=box,crop=crop_box(image,box)))
            kept,audit=reconcile_faces(image,ds)
            self.assertEqual(len(kept),1)
            self.assertEqual(len(audit),1)

    def test_no_face_and_no_detections_fail_open_for_review(self):
        image=canvas();ds=[candidate(image)]
        kept,audit=reconcile_faces(image,ds)
        self.assertEqual(kept,ds);self.assertFalse(audit)
        self.assertEqual(reconcile_faces(image,[]),([],[]))

    def test_opposite_corner_requires_a_full_face_not_a_suit_strip(self):
        image=canvas();paper(image)
        self.assertIsNotNone(secondary_rank(image,[100,200,140,260]))
        self.assertIsNone(secondary_rank(image,[100,200,119,260]))

    def test_secondary_identity_evidence_rejects_colour_and_shape_changes(self):
        crops=[]
        for text,red in [('7',False),('7',True),('A',False)]:
            image=canvas();f=paper(image,text=text,red=red);crops.append(secondary_rank(image,f))
        self.assertGreater(secondary_similarity(crops[0],crops[0]),.99)
        self.assertLess(secondary_similarity(crops[0],crops[1]),.94)
        self.assertLess(secondary_similarity(crops[0],crops[2]),.94)
        self.assertEqual(secondary_similarity(None,crops[0]),0)

    def test_every_pair_of_different_synthetic_rank_corners_fails_secondary_gate(self):
        ranks=['A','2','3','4','5','6','7','8','9','10','J','Q','K']
        crops={}
        for rank in ranks:
            image=canvas();f=paper(image,text=rank);crops[rank]=secondary_rank(image,f)
        for a in ranks:
            for b in ranks:
                if a!=b:
                    self.assertLess(secondary_similarity(crops[a],crops[b]),.94,(a,b))

    def test_face_processing_switch_and_suppression_provenance_in_full_pipeline(self):
        import tempfile
        from pathlib import Path
        from round_pipeline import process_frames
        class Engine:
            def detect(self,image,path):
                return [dict(x1=100,y1=200,x2=117,y2=222),dict(x1=100,y1=200,x2=106,y2=222)]
            def classify(self,crops):
                p=np.full(13,.001);p[6]=.988  # 7; numerical zero alone hides duplicates.
                return [p for crop in crops]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);frames=[]
            image=canvas();paper(image)
            for i in range(3):
                path=root/f'frame_{i}.png';cv2.imwrite(str(path),image);frames.append(path)
            kwargs=dict(engine=Engine(),regions=[dict(name='hand',rect=[0,0,1,1])],ocr_reader=False,reference_reader=False)
            enabled=process_frames(frames,root/'enabled',**kwargs)
            disabled=process_frames(frames,root/'disabled',face_evidence=False,**kwargs)
            self.assertEqual(enabled['cards'],['7'])
            self.assertEqual(disabled['cards'],['7','7'])
            self.assertEqual(enabled['events'][0]['observation_frames'],[0,1,2])
            self.assertEqual([s['frame'] for s in enabled['suppressed_detections']],[0,1,2])
            self.assertFalse(enabled['count_verified'])
            self.assertFalse(disabled['face_evidence_enabled'])
            self.assertEqual(disabled['suppressed_detections'],[])

    def test_secondary_birth_repair_requires_dim_fade_and_respects_co_visibility(self):
        early,late=canvas(),canvas();fa=paper(early,160,opacity=.5);fb=paper(late,100)
        oa=observation(early,[160,200,177,222],fa,0)
        ob=observation(late,[100,200,117,222],fb,1)
        self.assertGreater(secondary_animation_similarity(oa,ob),.94)
        self.assertEqual(secondary_animation_similarity(ob,ob),0)
        def track(i,obs):
            return SimpleNamespace(track_id=i,region='player',scene_id=0,round_number=1,
                    observations=obs,last_box=obs[-1]['box'],last_sample_index=obs[-1]['frame'])
        # Simulate a corrupted top corner. The opposite printed rank is intact.
        oa['crop']=np.full_like(oa['crop'],80)
        self.assertEqual(len(merge_track_fragments([track(1,[oa]),track(2,[ob])])),1)
        ob['frame']=0
        self.assertEqual(len(merge_track_fragments([track(1,[oa]),track(2,[ob])])),2)

    def test_projected_crop_preserves_stable_identity_when_detector_clips_the_corner(self):
        image=canvas();f=paper(image,text='A')
        box=[100,200,117,222];obs=[observation(image,box,f,i) for i in [0,1]]
        track=SimpleNamespace(track_id=1,last_box=box,last_sample_index=1,observations=obs)
        d=candidate(image,107,width=10);d['face_box']=f
        matches=CardIdentityMatcher(ordered_hand=True).match([d],[track],2,image=image)
        self.assertIs(matches[0],track)
        self.assertEqual(d['crop_source'],'face_projected')
        self.assertEqual(d['rank_box'],box)
        self.assertTrue(np.array_equal(d['crop'],obs[-1]['crop']))

    def test_projected_crop_cannot_turn_different_printed_rank_into_same_card(self):
        before,after=canvas(),canvas();fa=paper(before,text='A');fb=paper(after,text='7')
        box=[100,200,117,222];obs=[observation(before,box,fa,i) for i in [0,1]]
        track=SimpleNamespace(track_id=1,last_box=box,last_sample_index=1,observations=obs)
        d=candidate(after,107,width=10);d['face_box']=fb
        matches=CardIdentityMatcher(ordered_hand=True).match([d],[track],2,image=after)
        self.assertFalse(matches)
        self.assertNotIn('crop_source',d)

    def test_small_crops_and_crop_edges_do_not_crash_or_suppress_without_a_face(self):
        image=np.full((30,30,3),255,np.uint8)
        kept,audit=reconcile_faces(image,[dict(box=[0,0,5,8],crop=image[:8,:5])])
        self.assertEqual(len(kept),1);self.assertFalse(audit)

    def test_suit_sliver_requires_both_rank_corners_to_agree(self):
        for matching in [True,False]:
            image=canvas();face=paper(image,text='7')
            # Add an independent narrow candidate on the top suit's side.
            small=candidate(image,123,width=5)
            full=candidate(image)
            if not matching:
                other=canvas();paper(other,text='A')
                image[233:260,118:140]=other[233:260,118:140]
            kept,audit=reconcile_faces(image,[full,small])
            self.assertEqual(len(kept),1 if matching else 2)

    def test_projection_does_not_steal_new_identical_rank_at_previous_position(self):
        before,after=canvas(),canvas();old_face=paper(before,145,text='7')
        paper(after,100,text='7');paper(after,145,text='7')
        old_box=[145,200,162,222]
        obs=[observation(before,old_box,old_face,i) for i in [0,1]]
        old=SimpleNamespace(track_id=1,last_box=old_box,last_sample_index=1,observations=obs)
        ds,_=reconcile_faces(after,[candidate(after,145),candidate(after,100)])
        matches=CardIdentityMatcher(ordered_hand=True).match(ds,[old],2,image=after)
        self.assertEqual(len(matches),1)
        matched_index=next(iter(matches))
        self.assertEqual(ds[matched_index]['face_box'][0],100)

    def test_projection_refuses_previous_suit_or_partial_corner(self):
        image=canvas();f=paper(image)
        d=candidate(image);d['face_box']=f
        for box in [[122,200,136,222],[100,200,106,222]]:
            self.assertIsNone(projected_rank_crop(image,d,observation(image,box,f,0)))


if __name__=='__main__':unittest.main()
