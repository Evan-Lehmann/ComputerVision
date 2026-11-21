"""Off-table windows must never supply physical-card evidence."""
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
from round_pipeline import DEFAULT_REGIONS, RANKS, process_frames
from round_segmentation import visible_table


class Engine:
    def __init__(self):
        self.seen = []
    def detect(self, image, path):
        self.seen.append(path.name)
        return [dict(x1=105, y1=158, x2=115, y2=174)]
    def classify(self, crops):
        p = np.full(13, .001)
        p[RANKS.index('6')] = .988
        return [p.copy() for _ in crops]


class TableVisibilityTests(unittest.TestCase):
    def test_felt_required_in_both_hands_and_cards_do_not_hide_table(self):
        image = np.full((200, 300, 3), (35, 65, 40), np.uint8)
        cv2.rectangle(image, (105, 152), (125, 185), (245, 245, 245), -1)
        self.assertTrue(visible_table(image, DEFAULT_REGIONS))
        for color in [(30, 30, 30), (255, 255, 255), (45, 40, 35)]:
            self.assertFalse(visible_table(np.full_like(image, color), DEFAULT_REGIONS))
        image[:, 174:] = 30
        self.assertFalse(visible_table(image, DEFAULT_REGIONS))

    def test_thumbnails_are_skipped_audited_and_streaming_matches_offline(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = []
            for i in range(10):
                image = np.full((200, 300, 3), (35, 65, 40) if i < 3 or i >= 7 else (30, 30, 30), np.uint8)
                # Same glyph/box in a review thumbnail and on the real table.
                cv2.rectangle(image, (103, 153), (120, 183), (245, 245, 245), -1)
                cv2.putText(image, '6', (105, 171), 0, .45, (0, 0, 0), 1)
                path = root / ('frame_%03d.png' % i)
                cv2.imwrite(str(path), image)
                paths.append(path)
            engine = Engine()
            result = process_frames(paths, root/'offline', engine=engine, ocr_reader=False,
                                    reference_reader=False, shared_view=False, face_evidence=False)
            self.assertEqual(len(engine.seen), 6)
            self.assertEqual(result['frames_processed'], 10)
            self.assertEqual(result['unresolved'], 0)
            self.assertEqual(result['boundaries'], [])
            self.assertEqual(len(result['uncertain_transitions']), 1)
            self.assertTrue(result['coverage_pending'])
            self.assertIsNone(result['count_change'])
            self.assertFalse(result['count_verified'])
            self.assertEqual([(g['first_frame'], g['last_frame']) for g in result['visibility_gaps']], [(3, 6)])
            self.assertTrue(all(f not in range(3, 7) for e in result['events'] for f in e['observation_frames']))
            snapshots = []
            streamed = process_frames([], root/'stream', engine=Engine(), ocr_reader=False,
                reference_reader=False, shared_view=False, face_evidence=False,
                frame_stream=iter(paths), live_progress=snapshots.append)
            self.assertEqual(result, streamed)
            self.assertEqual(len(snapshots), 10)
            self.assertEqual(snapshots[3]['visibility_gaps'][0]['last_frame'], 3)
            self.assertIn('Table view is hidden', snapshots[3]['warnings'][0])
            self.assertNotIn('Table view is hidden', snapshots[-1]['warnings'][0])

    def test_custom_layout_is_not_subject_to_green_table_assumption(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = []
            for i in range(2):
                path = root / ('frame_%03d.png' % i)
                cv2.imwrite(str(path), np.full((200, 300, 3), 255, np.uint8))
                paths.append(path)
            engine = Engine()
            result = process_frames(paths, root/'out', engine=engine,
                regions=[dict(name='custom', rect=[0, 0, 1, 1])],
                ocr_reader=False, reference_reader=False, shared_view=False, face_evidence=False)
            self.assertEqual(len(engine.seen), 2)
            self.assertEqual(result['visibility_gaps'], [])

    def test_entire_hidden_recording_cannot_return_verified_zero(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root/'frame_000.png'
            cv2.imwrite(str(path), np.full((200, 300, 3), 30, np.uint8))
            engine = Engine()
            result = process_frames([path], root/'out', engine=engine,
                ocr_reader=False, reference_reader=False, shared_view=False)
            self.assertEqual(engine.seen, [])
            self.assertEqual(result['cards'], [])
            self.assertTrue(result['coverage_pending'])
            self.assertIsNone(result['count_change'])
            self.assertFalse(result['count_verified'])


if __name__ == '__main__':
    unittest.main()
