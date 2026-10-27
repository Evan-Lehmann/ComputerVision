import copy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from review_export import reviewed_crop_bundle


class ReviewExportTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)/'analysis';self.root.mkdir()
        # Export copies evidence bytes; it must not relabel unreviewed inputs.
        (self.root/'a.png').write_bytes(b'image-a')
        (self.root/'same.png').write_bytes(b'image-a')
        (self.root/'b.png').write_bytes(b'image-b')
        def event(i,rank,evidence,**extra):
            return dict(id=str(i),rank=rank,region='player',reviewed=True,excluded=False,
                        evidence=[dict(image=name,frame=n,source='frame_%03d.png'%n) for n,name in enumerate(evidence)],**extra)
        self.result=dict(review_confirmed=True,count_verified=True,events=[
            event(1,'3',['a.png','same.png','b.png']),
            event(2,'3',['a.png']),
            dict(event(3,'UNKNOWN',['not-present.png']),excluded=True),
            event(4,'5',[])])
    def tearDown(self):self.temp.cleanup()

    def test_bundle_preserves_physical_groups_and_excludes_false_detections(self):
        original=copy.deepcopy(self.result)
        data=reviewed_crop_bundle(self.result,self.root,'stable-recording')
        with zipfile.ZipFile(data) as archive:
            rows=list(csv.DictReader(io.StringIO(archive.read('labels.csv').decode())))
            metadata=json.loads(archive.read('metadata.json'))
            self.assertEqual(len(rows),3)  # repeated view deduped within card 1 only
            self.assertEqual({row['card_group'] for row in rows},{'stable-recording:1','stable-recording:2'})
            self.assertEqual({row['rank'] for row in rows},{'3'})
            self.assertTrue(all(row['annotation_source']=='user_confirmed_review' for row in rows))
            self.assertEqual(archive.read('crops/a.png'),b'image-a')
            self.assertNotIn('crops/not-present.png',archive.namelist())
            self.assertEqual(metadata['confirmed_cards'],3)
            self.assertEqual(metadata['labelled_cards'],2)
            self.assertEqual(metadata['crop_count'],3)
            self.assertEqual(metadata['count_scope'],'captured_cards')
        self.assertEqual(self.result,original)

    def test_unconfirmed_unresolved_and_unreviewed_labels_are_rejected(self):
        cases=[dict(self.result,review_confirmed=False),dict(self.result,count_verified=False)]
        invalid=copy.deepcopy(self.result);invalid['events'][0]['rank']='UNKNOWN';cases.append(invalid)
        invalid=copy.deepcopy(self.result);invalid['events'][0]['reviewed']=False;cases.append(invalid)
        for case in cases:
            with self.subTest(case=case):
                with self.assertRaises(ValueError):reviewed_crop_bundle(case,self.root,'group')

    def test_missing_or_escaping_evidence_is_not_read(self):
        for name in ['../outside.png','/tmp/outside.png']:
            invalid=copy.deepcopy(self.result);invalid['events'][0]['evidence'][0]['image']=name
            with self.assertRaises(ValueError):reviewed_crop_bundle(invalid,self.root,'group')
        outside=Path(self.temp.name)/'outside.png';outside.write_bytes(b'outside')
        (self.root/'link.png').symlink_to(outside)
        invalid=copy.deepcopy(self.result);invalid['events'][0]['evidence'][0]['image']='link.png'
        with self.assertRaises(ValueError):reviewed_crop_bundle(invalid,self.root,'group')
        invalid=copy.deepcopy(self.result);invalid['events'][0]['evidence'][0]['image']='missing.png'
        with self.assertRaises(OSError):reviewed_crop_bundle(invalid,self.root,'group')

    def test_no_evidence_does_not_create_an_empty_training_bundle(self):
        self.result['events']=[dict(id='manual',rank='3',region='manual',reviewed=True,evidence=[])]
        with self.assertRaisesRegex(ValueError,'no reviewed crop'):
            reviewed_crop_bundle(self.result,self.root,'group')

if __name__=='__main__':unittest.main()
