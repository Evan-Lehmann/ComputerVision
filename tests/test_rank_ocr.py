import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import cv2
import numpy as np
from rank_ocr import TesseractRankReader, red_glyph_image


def red_three():
    image=np.full((38,30,3),255,np.uint8)
    image[:3]=40;image[:,:3]=40
    cv2.putText(image,'3',(7,29),cv2.FONT_HERSHEY_SIMPLEX,.8,(15,15,210),1,cv2.LINE_AA)
    return image


def tsv(text='3',confidence=96):
    return ('level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n'
            '5\t1\t1\t1\t1\t1\t0\t0\t20\t20\t%s\t%s\n' % (confidence,text)).encode()


class RankOCRTests(unittest.TestCase):
    def test_black_ink_is_not_sent_to_red_rank_reader(self):
        image=np.full((30,25,3),255,np.uint8)
        cv2.putText(image,'3',(4,23),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,0),1)
        self.assertIsNone(red_glyph_image(image))
        runner=Mock()
        self.assertIsNone(TesseractRankReader('test',runner).read(image))
        runner.assert_not_called()

    def test_red_extraction_keeps_both_digits_of_ten(self):
        image=np.full((35,45,3),255,np.uint8)
        cv2.putText(image,'10',(5,26),cv2.FONT_HERSHEY_SIMPLEX,.8,(10,10,220),1)
        cleaned=red_glyph_image(image)
        self.assertIsNotNone(cleaned)
        count,_,_,_=cv2.connectedComponentsWithStats((cleaned<128).astype(np.uint8))
        self.assertEqual(count-1,2)
        self.assertTrue(np.all(cleaned[:10]==255))

    def test_parses_ocr_score_and_rejects_non_rank_text(self):
        for text in ['3','10','Q']:
            runner=Mock(return_value=SimpleNamespace(returncode=0,stdout=tsv(text)))
            self.assertEqual(TesseractRankReader('test',runner).read(red_three()),dict(rank=text,confidence=.96))
        for text in ['1','0','37','FOO']:
            runner=Mock(return_value=SimpleNamespace(returncode=0,stdout=tsv(text)))
            self.assertIsNone(TesseractRankReader('test',runner).read(red_three()))

    def test_requires_high_confidence_agreement_in_distinct_frames(self):
        reader=TesseractRankReader('test')
        observations=[dict(frame=i,crop=None) for i in range(3)]
        reader.read=Mock(side_effect=[dict(rank='3',confidence=.95),dict(rank='3',confidence=.94),None])
        result=reader.consensus(observations)
        self.assertEqual(result['rank'],'3')
        self.assertEqual(result['confidence'],.94)
        self.assertEqual([v['frame'] for v in result['votes']],[0,1])
        reader.read=Mock(side_effect=[dict(rank='3',confidence=.95),dict(rank='5',confidence=.97),dict(rank='3',confidence=.94)])
        self.assertIsNone(reader.consensus(observations))
        reader.read=Mock(side_effect=[dict(rank='3',confidence=.89),None,dict(rank='3',confidence=.95)])
        self.assertIsNone(reader.consensus(observations))
        reader.read=Mock(return_value=dict(rank='3',confidence=.99))
        self.assertIsNone(reader.consensus([dict(frame=0,crop=None),dict(frame=0,crop=None)]))
        self.assertIsNone(reader.consensus(observations[:1]))

    def test_missing_binary_and_timeout_leave_rank_unresolved(self):
        with patch('rank_ocr.shutil.which',return_value=None):
            reader=TesseractRankReader()
            self.assertFalse(reader.available)
            self.assertIsNone(reader.read(red_three()))
        runner=Mock(side_effect=subprocess.TimeoutExpired('tesseract',10))
        reader=TesseractRankReader('test',runner)
        self.assertIsNone(reader.read(red_three()))
        self.assertIn('TimeoutExpired',reader.error)

    def test_native_ocr_if_installed(self):
        reader=TesseractRankReader()
        if not reader.available:
            self.skipTest('Optional Tesseract binary is not installed')
        result=reader.read(red_three())
        self.assertIsNotNone(result)
        self.assertEqual(result['rank'],'3')

if __name__=='__main__':unittest.main()
