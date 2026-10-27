import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace
import tempfile
from benchmark_external_detector import card_label, validate_classes, corner_overlap, observation_metrics, load_external_model, source_controls


def detection(rank='Q',confidence=.9,box=None):
    return dict(region='player',rank=rank,confidence=confidence,box=box or [8,8,24,40])


def reference():
    return dict(scope='synthetic corner annotation',cards=[dict(id='one',rank='Q',round_number=1)],
                observations=[dict(frame=0,region='player',box=[10,10,20,25],identity='one')])


class ExternalDetectorBenchmarkTests(unittest.TestCase):
    def test_rank_suit_mapping_and_full_deck_validation(self):
        self.assertEqual(card_label('10s'),('10','s'))
        self.assertEqual(card_label('Qh'),('Q','h'))
        self.assertEqual(card_label('AC'),('A','c'))
        names={i:r+s for i,(r,s) in enumerate((r,s) for r in ['A','2','3','4','5','6','7','8','9','10','J','Q','K'] for s in 'cdhs')}
        validate_classes(names)
        with self.assertRaises(ValueError):validate_classes({i:r+'h' for i,r in enumerate(['A','2','3','4','5','6','7','8','9','10','J','Q','K'])})
        for bad in ['1h','11c','joker','10','']: 
            with self.subTest(label=bad),self.assertRaises(ValueError):card_label(bad)

    def test_corner_geometry_accepts_suit_extension_not_huge_or_disjoint_boxes(self):
        self.assertEqual(corner_overlap([10,10,20,25],[8,8,24,40]),1)
        self.assertEqual(corner_overlap([10,10,20,25],[0,0,1000,1000]),0)
        self.assertEqual(corner_overlap([10,10,20,25],[30,30,40,50]),0)
        self.assertEqual(corner_overlap([10,10,20,25],[1,1,1,1]),0)

    def test_separates_misses_wrong_ranks_and_unmatched_predictions(self):
        frames=[dict(frame=0,predictions=[detection('K'),detection('2',box=[40,10,55,35])])]
        metrics=observation_metrics(reference(),frames,.25)
        self.assertEqual(metrics['matched_observations'],1)
        self.assertEqual(metrics['wrong_rank_observations'],1)
        self.assertEqual(metrics['unmatched_predictions'],1)
        self.assertEqual(metrics['physical_cards_ever_correctly_recognized'],0)
        frames=[dict(frame=0,predictions=[detection(confidence=.1)])]
        high=observation_metrics(reference(),frames,.25)
        self.assertEqual(high['missed_annotated_observations'],1)
        low=observation_metrics(reference(),frames,.05)
        self.assertEqual(low['correct_rank_observations'],1)
        self.assertEqual(low['physical_cards_ever_correctly_recognized'],1)

    def test_duplicate_boxes_do_not_double_match_or_choose_correct_label_by_truth(self):
        frames=[dict(frame=0,predictions=[detection('Q',.8),detection('K',.9)])]
        metrics=observation_metrics(reference(),frames,.25)
        self.assertEqual(metrics['wrong_rank_observations'],1)
        self.assertEqual(metrics['correct_rank_observations'],0)
        self.assertEqual(metrics['unmatched_predictions'],1)
        self.assertEqual(metrics['matched_observations'],1)

    def test_unknown_checkpoint_globals_are_rejected_before_loading(self):
        import torch
        if not hasattr(torch.serialization,'get_unsafe_globals_in_checkpoint'):
            self.skipTest('Optional external benchmark requires PyTorch 2.6+')
        with patch('torch.serialization.get_unsafe_globals_in_checkpoint',return_value=['os.system']):
            with self.assertRaisesRegex(ValueError,'Unsupported checkpoint globals'):
                load_external_model('unused.pt')

    def test_source_control_rejects_mismatched_class_order(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'test').mkdir()
            (root/'data.yaml').write_text('names: [Kh]\n')
            with self.assertRaisesRegex(ValueError,'class IDs do not match'):
                source_controls(SimpleNamespace(names={0:'Qh'}),root/'test')


if __name__=='__main__':unittest.main()
