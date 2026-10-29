import csv
import hashlib
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np
from train_rank_classifier_v5 import load_reviewed


class AuditedTrainingInputsTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        _,raw=cv2.imencode('.png',np.full((24,20,3),255,np.uint8))
        (self.root/'crop.png').write_bytes(raw.tobytes())
        self.row=dict(crop='crop.png',rank='5',recording_group='development',card_group='card1',
                      sha256=hashlib.sha256(raw.tobytes()).hexdigest(),annotation_source='visual_frame_audit')
    def tearDown(self):self.temp.cleanup()
    def write(self,rows):
        with (self.root/'labels.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(self.row));w.writeheader();w.writerows(rows)
    def test_groups_must_be_explicit_and_checksums_match(self):
        self.write([self.row]);self.assertEqual(len(load_reviewed(self.root,{'development'})),1)
        with self.assertRaises(ValueError):load_reviewed(self.root,{'heldout'})
        with self.assertRaises(ValueError):load_reviewed(self.root,{'development','heldout'})
        (self.root/'crop.png').write_bytes(b'changed')
        with self.assertRaises(ValueError):load_reviewed(self.root,{'development'})
    def test_conflicting_physical_labels_and_unreviewed_sources_are_rejected(self):
        self.write([self.row,dict(self.row,rank='3')])
        with self.assertRaises(ValueError):load_reviewed(self.root,{'development'})
        self.write([dict(self.row,annotation_source='model_prediction')])
        with self.assertRaises(ValueError):load_reviewed(self.root,{'development'})
    def test_path_escape_rejected_and_repeated_views_deduplicated_per_card(self):
        self.write([dict(self.row,crop='../crop.png')])
        with self.assertRaises(ValueError):load_reviewed(self.root,{'development'})
        self.write([self.row,self.row,dict(self.row,card_group='card2')])
        self.assertEqual(len(load_reviewed(self.root,{'development'})),2)


if __name__=='__main__':unittest.main()
