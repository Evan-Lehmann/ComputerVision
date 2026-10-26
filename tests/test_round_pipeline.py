import copy
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import cv2
import numpy as np
from round_pipeline import process_frames, apply_review, validate_regions
from card_tracking import merge_track_fragments

REGIONS = [dict(name='cards',rect=[0,0,1,1])]

class Engine:
    def __init__(self, rank='6', boxes=None):
        self.rank = rank
        self.boxes = boxes or [[20,20,35,40]]
    def detect(self, image, path):
        return [dict(zip(['x1','y1','x2','y2'],b)) for b in self.boxes]
    def classify(self, crops):
        from round_pipeline import RANKS
        p = np.full(13, .001)
        p[RANKS.index(self.rank)] = .988
        return [p for _ in crops]

class RoundPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.frames = []
        for i in range(3):
            image = np.full((100,100,3),255,np.uint8)
            cv2.putText(image,'6',(20,37),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),1)
            cv2.putText(image,'6',(60,37),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),1)
            path = self.root/('frame_%03d.png'%i)
            cv2.imwrite(str(path),image)
            self.frames.append(path)
    def tearDown(self):
        self.temp.cleanup()
    def process(self, frames=None, engine=None, regions=None):
        return process_frames(self.frames if frames is None else frames,self.root/'out',engine or Engine(),regions or REGIONS)
    def test_detects_once_per_physical_track_and_exports_provisional_count(self):
        result = self.process()
        self.assertEqual(result['cards'],['6'])
        self.assertEqual(result['count_change'],1)
        self.assertFalse(result['count_verified'])
        self.assertTrue((self.root/'out/result.json').is_file())
        self.assertEqual(len(result['events'][0]['evidence']),3)
    def test_same_rank_cards_visible_together_are_not_reconciled_as_duplicates(self):
        result = self.process(engine=Engine(boxes=[[20,20,35,40],[60,20,75,40]]))
        self.assertEqual(result['cards'],['6','6'])
        self.assertEqual(result['count_change'],2)
    def test_excluded_ui_region_does_not_add_duplicate_counts(self):
        result = self.process(engine=Engine(boxes=[[20,20,35,40],[60,20,75,40]]),regions=[dict(name='main',rect=[0,0,.5,1])])
        self.assertEqual(result['cards'],['6'])
    def test_unknown_card_prevents_final_count_but_retains_subtotal(self):
        result = self.process(frames=self.frames[:1])
        self.assertEqual(result['cards'],['UNKNOWN'])
        self.assertIsNone(result['count_change'])
        self.assertEqual(result['known_count_change'],0)
        event=result['events'][0]
        reviewed=apply_review(result,[dict(id=event['id'],rank='A',excluded=False)],True)
        self.assertEqual(reviewed['count_change'],-1)
        self.assertTrue(reviewed['count_verified'])
    def test_review_preserves_audit_and_supports_missing_cards_and_order(self):
        result=self.process()
        original=copy.deepcopy(result)
        reviewed=apply_review(result,[dict(id='new-added',rank='10'),dict(id=result['events'][0]['id'],rank='6',excluded=True)],True)
        self.assertEqual(reviewed['cards'],['10'])
        self.assertEqual(result,original)
        self.assertEqual(len(reviewed['events']),2)
        with self.assertRaises(ValueError):apply_review(result,[],True)
        with self.assertRaises(ValueError):apply_review(result,[dict(id='bad',rank='6')],True)
    def test_invalid_regions_and_unreadable_or_empty_inputs_fail(self):
        with self.assertRaises(ValueError):validate_regions([dict(name='a',rect=[0,0,1,1]),dict(name='b',rect=[.5,.5,1,1])])
        with self.assertRaises(ValueError):self.process(frames=[])
        bad=self.root/'missing.png'
        with self.assertRaises(ValueError):self.process(frames=[bad])
    def test_optional_ocr_resolves_unknown_and_preserves_cnn_provenance(self):
        from unittest.mock import Mock
        class Uncertain(Engine):
            def classify(self,crops):return [np.full(13,1/13) for _ in crops]
        reader=Mock(available=True,error=None)
        reader.consensus.return_value=dict(rank='3',confidence=.95,votes=[dict(frame=0,rank='3',confidence=.96),dict(frame=2,rank='3',confidence=.95)])
        result=process_frames(self.frames,self.root/'out',Uncertain(),REGIONS,ocr_reader=reader)
        self.assertEqual(result['cards'],['3'])
        self.assertEqual(result['count_change'],1)
        self.assertFalse(result['count_verified'])
        event=result['events'][0]
        self.assertEqual(event['recognition_source'],'ocr_consensus')
        self.assertEqual(event['suggested_rank'],'3')
        self.assertLess(event['cnn_confidence'],.8)
        self.assertEqual(len(event['ocr_votes']),2)

    def test_optional_ocr_never_overrides_confident_cnn(self):
        from unittest.mock import Mock
        reader=Mock(available=True,error=None)
        result=process_frames(self.frames,self.root/'out',Engine(),REGIONS,ocr_reader=reader)
        self.assertEqual(result['cards'],['6'])
        reader.consensus.assert_not_called()

    def test_fragment_repair_never_merges_two_observations_in_same_frame(self):
        image=cv2.imread(str(self.frames[0]))
        crop=image[18:42,18:37]
        def track(i):return SimpleNamespace(track_id=i,region='cards',observations=[dict(frame=0,crop=crop,box=[20,20,35,40])])
        self.assertEqual(len(merge_track_fragments([track(1),track(2)])),2)
    def test_fragment_repair_rejoins_adjacent_identical_views(self):
        image=cv2.imread(str(self.frames[0]));crop=image[18:42,18:37]
        a=SimpleNamespace(track_id=1,region='cards',observations=[dict(frame=0,crop=crop,box=[20,20,35,40])])
        b=SimpleNamespace(track_id=2,region='cards',observations=[dict(frame=1,crop=crop,box=[20,20,35,40])])
        merged=merge_track_fragments([a,b])
        self.assertEqual(len(merged),1)
        self.assertEqual(merged[0].merged_from,[1,2])

if __name__=='__main__':unittest.main()
