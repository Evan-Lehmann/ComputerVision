import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
import numpy as np
from rank_reference import ReferenceRankReader, RANKS

ROOT = Path(__file__).resolve().parent.parent


def vote(rank='Q',accepted=True,score=.99):
    return dict(rank=rank,accepted=accepted,score=score,margin=.1,
                reference_key='existing-public-crop',reference_group='existing-public-group')


class ReferenceReaderTests(unittest.TestCase):
    def test_published_bank_uses_only_preexisting_training_examples_and_covers_all_ranks(self):
        reader = ReferenceRankReader()
        self.assertTrue(reader.available, reader.error)
        rows = json.loads((ROOT/'validation/v4_training_split.json').read_text())['train']
        self.assertEqual(set(reader.keys),{row['key'] for row in rows})
        self.assertEqual(dict(zip(reader.keys,reader.ranks)),{row['key']:row['rank'] for row in rows})
        self.assertEqual(set(reader.ranks),set(RANKS))
        self.assertEqual(len(reader.features),169)

    def test_consensus_requires_two_distinct_frames_and_cnn_top_rank_agreement(self):
        reader = ReferenceRankReader()
        observations = [dict(crop=None,frame=i) for i in range(3)]
        reader.read = Mock(side_effect=[vote(),vote(),None])
        self.assertEqual(reader.consensus(observations,'Q')['rank'],'Q')
        reader.read = Mock(return_value=vote())
        self.assertIsNone(reader.consensus(observations,'9'))
        self.assertIsNone(reader.consensus([observations[0],observations[0]],'Q'))
        self.assertIsNone(reader.consensus(observations[:1],'Q'))

    def test_conflicting_or_insufficient_strong_views_leave_unknown(self):
        reader = ReferenceRankReader()
        observations = [dict(crop=None,frame=i) for i in range(3)]
        reader.read = Mock(side_effect=[vote(),vote('9'),vote()])
        self.assertIsNone(reader.consensus(observations,'Q'))
        reader.read = Mock(side_effect=[vote(accepted=False),None,vote()])
        self.assertIsNone(reader.consensus(observations,'Q'))

    def test_spread_views_find_clear_evidence_missed_by_first_middle_last(self):
        reader = ReferenceRankReader()
        observations = [dict(crop=i,frame=i) for i in range(53)]
        reader.read = Mock(side_effect=lambda crop: vote('3',accepted=crop>=35))
        result = reader.consensus(observations,'3')
        self.assertEqual(result['rank'],'3')
        self.assertEqual([v['frame'] for v in result['votes']],[35,44,52])
        self.assertEqual(reader.read.call_count,7)

    def test_additional_views_cannot_vote_away_a_conflicting_accepted_rank(self):
        reader = ReferenceRankReader()
        observations = [dict(crop=i,frame=i) for i in range(53)]
        reader.read = Mock(side_effect=lambda crop: vote('9' if crop==35 else '3'))
        self.assertIsNone(reader.consensus(observations,'3'))

    def test_missing_invalid_and_empty_data_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'bank.npz'
            self.assertFalse(ReferenceRankReader(path).available)
            path.write_bytes(b'invalid')
            self.assertFalse(ReferenceRankReader(path).available)
        reader=ReferenceRankReader()
        self.assertIsNone(reader.read(None))
        self.assertIsNone(reader.read(np.full((30,30,3),255,np.uint8)))

    def test_synthetic_high_similarity_without_class_margin_is_not_accepted(self):
        from glyph_features import glyph_signature
        import cv2
        image=np.full((40,40,3),255,np.uint8)
        cv2.putText(image,'5',(8,30),cv2.FONT_HERSHEY_SIMPLEX,.8,(15,15,15),1)
        signature=glyph_signature(image)
        reader=ReferenceRankReader()
        # Every rank has the same appearance: no class can win by a margin.
        reader.features=np.stack([signature[0]]*13)
        reader.padded=[np.pad(f,2) for f in reader.features]
        reader.aspects=np.array([signature[1]]*13)
        reader.ranks=np.array(RANKS);reader.groups=np.array(['group']*13);reader.keys=np.array(RANKS)
        result=reader.read(image)
        self.assertGreater(result['score'],.99)
        self.assertFalse(result['accepted'])
        self.assertIsNone(reader.read(image,excluded_group='group'))


if __name__=='__main__':
    unittest.main()
