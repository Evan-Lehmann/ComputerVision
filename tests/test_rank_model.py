from pathlib import Path
import unittest
import numpy as np
import torch
from rank_model import RankCNN, RANKS, preprocess_crop

ROOT = Path(__file__).resolve().parent.parent

class RankModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        checkpoint=torch.load(ROOT/'models/rank_classifier_v3.pt',map_location='cpu',weights_only=True)
        cls.checkpoint=checkpoint
        cls.model=RankCNN();cls.model.load_state_dict(checkpoint['model_state_dict']);cls.model.eval()
    def test_checkpoint_class_order_and_representative_reserved_crops(self):
        self.assertEqual(self.checkpoint['ranks'],RANKS)
        self.assertEqual(self.checkpoint['image_size'],64)
        with np.load(ROOT/'tests/fixtures/card_tracking_crops.npz',allow_pickle=False) as crops:
            images=torch.cat([preprocess_crop(crops[key]) for key in ['t14_f115','t12_f110']])
        with torch.no_grad():predictions=self.model(images).argmax(1).tolist()
        self.assertEqual([RANKS[i] for i in predictions],['6','J'])
    def test_runtime_preprocessing_preserves_existing_checkpoint_contract(self):
        # No independent preprocessing drift between the experiment and app.
        from test_temporal_rank_tracking import preprocess_crop as existing
        with np.load(ROOT/'tests/fixtures/card_tracking_crops.npz',allow_pickle=False) as crops:
            for key in crops.files:
                self.assertTrue(torch.equal(preprocess_crop(crops[key]),existing(crops[key])),key)

if __name__=='__main__':unittest.main()
