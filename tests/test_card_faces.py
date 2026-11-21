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


class PhysicalFaceGlyphTests(unittest.TestCase):
    def test_verified_suit_is_attached_to_existing_face_not_counted_again(self):
        from unittest.mock import patch,Mock
        from card_faces import validate_face_glyphs
        image=canvas();paper(image)
        ds=[candidate(image),candidate(image,120,width=18)]
        with patch('card_faces._suit_validator',return_value=Mock(read=Mock(return_value={'accepted':True}))):
            kept,audit=validate_face_glyphs(image,ds,60)
        self.assertEqual(len(kept),1)
        self.assertEqual(audit[0]['reason'],'suit_position_on_same_physical_face')
        self.assertEqual(audit[0]['retained_box'],ds[0]['box'])
    def test_suit_without_rank_counterpart_is_never_silently_deleted(self):
        from unittest.mock import patch,Mock
        from card_faces import validate_face_glyphs
        image=canvas();paper(image)
        with patch('card_faces._suit_validator',return_value=Mock(read=Mock(return_value={'accepted':True}))):
            kept,audit=validate_face_glyphs(image,[candidate(image,120,width=18)],60)
        self.assertEqual(len(kept),1);self.assertFalse(audit)

    def test_geometry_alone_cannot_remove_an_unverified_right_corner(self):
        from unittest.mock import patch,Mock
        from card_faces import validate_face_glyphs
        image=canvas();paper(image)
        ds=[candidate(image),candidate(image,120,width=18)]
        for result in [None,{'accepted':False}]:
            with patch('card_faces._suit_validator',return_value=Mock(read=Mock(return_value=result))):
                kept,audit=validate_face_glyphs(image,ds,60)
            self.assertEqual(len(kept),2);self.assertFalse(audit)



class FaceBoundedAssociationTests(unittest.TestCase):
    def test_neighbour_ink_cannot_fragment_a_face_bounded_rank(self):
        from card_faces import face_rank_crop
        before,after=canvas(),canvas()
        fa=paper(before,160,text='2',red=True);fb=paper(after,130,text='2',red=True)
        # A neighbouring black corner lies outside the current card's paper.
        after[202:221,122:130]=0
        old_box=[160,200,177,222];new_box=[126,200,147,222]
        history=[]
        for i in [0,1]:
            old=observation(before,old_box,fa,i)
            old['face_rank_crop']=face_rank_crop(before,old)
            history.append(old)
        track=SimpleNamespace(track_id=1,last_box=old_box,last_sample_index=1,observations=history)
        d=dict(box=new_box,crop=crop_box(after,new_box),face_box=fb)
        from card_tracking import glyph_similarity
        self.assertLess(glyph_similarity(d['crop'],history[-1]['crop']),.78)
        unbounded=[dict(o) for o in history]
        for o in unbounded:o.pop('face_rank_crop')
        unsupported=SimpleNamespace(track_id=1,last_box=old_box,last_sample_index=1,observations=unbounded)
        self.assertFalse(CardIdentityMatcher(ordered_hand=True).match([dict(d)],[unsupported],2,image=after))
        matches=CardIdentityMatcher(ordered_hand=True).match([d],[track],2,image=after)
        self.assertIs(matches.get(0),track)
        # Classifier input must retain its original pixels, including padding.
        self.assertTrue(np.array_equal(d['crop'],crop_box(after,new_box)))

    def test_pipeline_preserves_one_identity_through_contaminated_compression(self):
        import tempfile
        from pathlib import Path
        from round_pipeline import process_frames
        before,after=canvas(),canvas()
        paper(before,160,text='2',red=True);paper(after,130,text='2',red=True)
        after[202:221,122:130]=0
        class Engine:
            def detect(self,image,path):
                box=[126,200,147,222] if path.stem=='frame_2' else [160,200,177,222]
                return [dict(zip(['x1','y1','x2','y2'],box))]
            def classify(self,crops):
                p=np.full(13,.001);p[1]=.988
                return [p for _ in crops]
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);paths=[]
            for i,image in enumerate([before,before,after]):
                path=root/f'frame_{i}.png';cv2.imwrite(str(path),image);paths.append(path)
            kwargs=dict(engine=Engine(),regions=[dict(name='hand',rect=[0,0,1,1])],
                        shared_view=False,ocr_reader=False,reference_reader=False)
            snapshots=[]
            result=process_frames([],root/'enabled',frame_stream=iter(paths),live_progress=snapshots.append,**kwargs)
            unbounded=process_frames(paths,root/'disabled',face_evidence=False,**kwargs)
            self.assertEqual(result['cards'],['2'])
            self.assertEqual(result['events'][0]['observation_frames'],[0,1,2])
            self.assertEqual(result['known_count_change'],1)
            self.assertEqual([r['events'][0]['id'] for r in snapshots],['1']*3)
            self.assertEqual(unbounded['cards'],['2','UNKNOWN'])
            self.assertIsNone(unbounded['count_change'])

    def test_bounded_view_requires_supported_face_and_complete_rank_width(self):
        from card_faces import face_rank_crop
        image=canvas();face=paper(image)
        for box,fb in [([100,200,117,222],None),([100,200,106,222],face),
                       ([100,200,117,222],[100,200,110,260]),([120,200,137,222],face)]:
            self.assertIsNone(face_rank_crop(image,dict(box=box,face_box=fb)))

    def test_bounded_view_cannot_link_different_rank_or_colour(self):
        from card_faces import face_rank_crop
        before=canvas();fa=paper(before,160,text='2',red=True);box=[160,200,177,222]
        history=[]
        for i in [0,1]:
            old=observation(before,box,fa,i);old['face_rank_crop']=face_rank_crop(before,old);history.append(old)
        track=SimpleNamespace(track_id=1,last_box=box,last_sample_index=1,observations=history)
        for rank,red in [('3',True),('2',False)]:
            after=canvas();fb=paper(after,130,text=rank,red=red)
            d=dict(box=[130,200,147,222],crop=crop_box(after,[130,200,147,222]),face_box=fb)
            self.assertFalse(CardIdentityMatcher(ordered_hand=True).match([d],[track],2,image=after))

    def test_different_synthetic_ranks_do_not_pass_the_bounded_identity_gate(self):
        from card_faces import face_rank_crop
        from card_tracking import glyph_similarity
        ranks=['A','2','3','4','5','6','7','8','9','10','J','Q','K']
        crops={}
        for rank in ranks:
            image=canvas();face=paper(image,text=rank)
            crops[rank]=face_rank_crop(image,dict(box=[100,200,117,222],face_box=face))
        for a in ranks:
            for b in ranks:
                if a!=b:self.assertLess(glyph_similarity(crops[a],crops[b]),.94,(a,b))

    def test_near_ties_in_bounded_view_still_abstain(self):
        from card_faces import face_rank_crop
        image=canvas();f=paper(image,text='2',red=True);box=[100,200,117,222]
        history=[]
        for i in [0,1]:
            o=observation(image,box,f,i);o['face_rank_crop']=face_rank_crop(image,o);history.append(o)
        tracks=[SimpleNamespace(track_id=i,last_box=box,last_sample_index=1,observations=history) for i in [1,2]]
        d=dict(box=box,crop=crop_box(image,box),face_box=f)
        self.assertFalse(CardIdentityMatcher(ordered_hand=True).match([d],tracks,2,image=image))


class RankSuitFragmentTests(unittest.TestCase):
    def test_fragment_requires_verified_suit_and_independent_complete_paper(self):
        from unittest.mock import patch,Mock
        from card_faces import validate_face_glyphs
        for accepted in [True,False]:
            image=canvas();paper(image,text='K')
            full=candidate(image,width=17)
            partial=candidate(image,112,width=10)
            with patch('card_faces._suit_validator',return_value=Mock(read=Mock(return_value={'accepted':accepted,'suit':'clubs'}))):
                kept,audit=validate_face_glyphs(image,[full,partial],60)
            self.assertEqual(len(kept),1 if accepted else 2)
            if accepted:
                self.assertEqual(audit[0]['reason'],'rank_suit_fragment_on_same_physical_face')
                self.assertEqual(audit[0]['validated_suit'],'clubs')
                self.assertEqual(audit[0]['retained_box'],full['box'])

    def test_full_second_corner_and_lone_fragment_are_kept(self):
        from unittest.mock import patch,Mock
        from card_faces import validate_face_glyphs
        image=canvas();paper(image,text='K')
        for ds in [[candidate(image,112,width=17),candidate(image)], [candidate(image,112,width=10)]]:
            with patch('card_faces._suit_validator',return_value=Mock(read=Mock(return_value={'accepted':True,'suit':'clubs'}))):
                kept,audit=validate_face_glyphs(image,ds,60)
            self.assertEqual(kept,ds);self.assertFalse(audit)

    def test_partial_on_neighbouring_overlapping_face_is_kept(self):
        from unittest.mock import patch,Mock
        from card_faces import validate_face_glyphs
        image=canvas();paper(image,100,text='K');paper(image,112,text='7')
        ds=[candidate(image),candidate(image,112,width=10)]
        with patch('card_faces._suit_validator',return_value=Mock(read=Mock(return_value={'accepted':True,'suit':'clubs'}))):
            kept,audit=validate_face_glyphs(image,ds,60)
        self.assertEqual(kept,ds);self.assertFalse(audit)


if __name__=='__main__':unittest.main()
