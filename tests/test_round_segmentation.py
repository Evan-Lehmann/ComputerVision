"""Round transitions and running counts, independent of trained model accuracy."""
import copy
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from round_pipeline import apply_review, process_frames, RANKS, summarize
from round_segmentation import RoundSegmenter, visible_empty_table

REGIONS = [dict(name='table', rect=[0, 0, 1, 1])]
BOX = [106, 98, 124, 124]


def table(card=False, red_back=False):
    image = np.full((200, 300, 3), (35, 65, 40), dtype=np.uint8)
    if card or red_back:
        cv2.rectangle(image, (100, 80), (135, 140), (70, 75, 200) if red_back else (245, 245, 245), -1)
        if card:
            cv2.putText(image, '6', (105, 123), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 0), 2)
    return image


class ScheduledEngine:
    def __init__(self, schedule):
        self.schedule = schedule
    def detect(self, image, path):
        self.current = self.schedule[int(path.stem.split('_')[-1])]
        return [dict(zip(['x1', 'y1', 'x2', 'y2'], BOX))] if self.current else []
    def classify(self, crops):
        probabilities = np.full(13, .001)
        if self.current == 'UNKNOWN':
            probabilities[:] = 1/13
        else:
            probabilities[RANKS.index(self.current or '6')] = .988
        return [probabilities.copy() for _ in crops]


class MultiRoundTests(unittest.TestCase):
    def process(self, schedule, overrides=None):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            frames = []
            for i, rank in enumerate(schedule):
                path = root/('frame_%03d.png' % i)
                image = (overrides or {}).get(i, table(card=bool(rank)))
                cv2.imwrite(str(path), image)
                frames.append(path)
            return process_frames(frames, root/'analysis', ScheduledEngine(schedule), REGIONS, ocr_reader=False)

    def test_two_rounds_preserve_repeated_same_rank_at_identical_position(self):
        result = self.process(['6']*3 + [None]*4 + ['6']*3)
        self.assertEqual(result['cards'], ['6', '6'])
        self.assertEqual([e['round_number'] for e in result['events']], [1, 2])
        self.assertEqual([e['observations'] for e in result['events']], [3, 3])
        self.assertEqual([r['count_change'] for r in result['rounds']], [1, 1])
        self.assertEqual([r['running_count_change'] for r in result['rounds']], [1, 2])
        self.assertEqual(result['count_change'], 2)
        self.assertFalse(result['count_verified'])
        self.assertTrue(result['needs_round_review'])
        boundary = result['boundaries'][0]
        self.assertEqual((boundary['clear_start'], boundary['clear_end'], boundary['first_frame']), (3, 6, 7))

    def test_leading_trailing_clear_table_and_partial_last_round_do_not_add_phantom_rounds(self):
        result = self.process([None]*5 + ['6']*3 + [None]*5)
        self.assertEqual(result['round_count'], 1)
        result = self.process(['6']*3 + [None]*4 + ['6'])
        self.assertEqual(result['round_count'], 2)
        self.assertIsNone(result['rounds'][1]['count_change'])
        self.assertEqual(result['rounds'][0]['running_count_change'], 1)
        self.assertIsNone(result['rounds'][1]['running_count_change'])

    def test_unknown_in_first_round_keeps_later_running_total_incomplete(self):
        result = self.process(['UNKNOWN']*3 + [None]*4 + ['6']*3)
        self.assertIsNone(result['rounds'][0]['count_change'])
        self.assertEqual(result['rounds'][1]['count_change'], 1)
        self.assertIsNone(result['rounds'][1]['running_count_change'])
        self.assertEqual(result['rounds'][1]['running_count_range'], dict(min=0, max=2))
        self.assertEqual(result['known_count_change'], 1)
        self.assertIsNone(result['count_change'])

    def test_detector_dropout_with_visible_cards_or_offscreen_window_is_not_a_boundary(self):
        for image in [table(card=True), np.full((200, 300, 3), 35, np.uint8),
                      np.full((200, 300, 3), 255, np.uint8)]:
            with self.subTest(image=int(image.mean())):
                result = self.process(['6']*3 + [None]*4 + ['6']*3,
                                      {i: image for i in range(3, 7)})
                self.assertEqual(result['round_count'], 1)
                self.assertEqual(result['boundaries'], [])
                self.assertEqual(len(result['uncertain_transitions']), 1)
                self.assertTrue(result['needs_round_review'])

    def test_short_clear_interval_requires_review_instead_of_silent_split(self):
        result = self.process(['6']*3 + [None]*2 + ['6']*3)
        self.assertEqual(result['round_count'], 1)
        self.assertEqual(len(result['uncertain_transitions']), 1)

    def test_only_sustained_clear_immediately_before_cards_can_split(self):
        segmenter = RoundSegmenter()
        segmenter.observe(0, True)
        for i in range(1, 5):
            segmenter.observe(i, False, True)
        segmenter.observe(5, False, False)  # App/window changed after table clearing.
        self.assertFalse(segmenter.observe(6, True))
        self.assertEqual(segmenter.boundaries, [])
        self.assertEqual(len(segmenter.uncertain_transitions), 1)

    def test_one_hand_disappearing_while_other_has_cards_does_not_split(self):
        segmenter = RoundSegmenter()
        for i in range(12):
            self.assertFalse(segmenter.observe(i, True))
        self.assertEqual(segmenter.round_number, 1)

    def test_visible_empty_table_rejects_faces_backs_nonfelt_and_empty_regions(self):
        self.assertTrue(visible_empty_table(table(), REGIONS))
        for image in [table(card=True), table(red_back=True), np.full((200, 300, 3), 255, np.uint8)]:
            self.assertFalse(visible_empty_table(image, REGIONS))
        self.assertFalse(visible_empty_table(table(), [dict(name='empty', rect=[0, 0, 0, 0])]))

    def test_review_can_correct_round_assignments_and_recompute_all_totals(self):
        result = self.process(['6']*3 + [None]*4 + ['A']*3)
        original = copy.deepcopy(result)
        edits = [dict(id=e['id'], rank=e['rank'], round_number=e['round_number']) for e in result['events']]
        reviewed = apply_review(result, edits, True)
        self.assertFalse(reviewed['count_verified'])  # Boundaries not yet confirmed.
        reviewed = apply_review(result, edits, True, True)
        self.assertTrue(reviewed['count_verified'])
        self.assertEqual([r['running_count_change'] for r in reviewed['rounds']], [1, 0])
        edits[1]['round_number'] = 1
        reviewed = apply_review(result, edits, True, True)
        self.assertEqual(reviewed['round_count'], 1)
        self.assertEqual(reviewed['rounds'][0]['cards'], ['6', 'A'])
        self.assertEqual(reviewed['count_change'], 0)
        self.assertEqual(result, original)
        edits.append(dict(id='new-extra', rank='4', round_number=2))
        reviewed = apply_review(result, edits, True, True)
        self.assertEqual([r['running_count_change'] for r in reviewed['rounds']], [0, 1])
        edits[2]['excluded'] = True
        reviewed = apply_review(result, edits, True, True)
        self.assertEqual(reviewed['count_change'], 0)
        self.assertEqual(reviewed['rounds'][1]['cards'], [])
        self.assertFalse(apply_review(reviewed, edits, False, True)['rounds_confirmed'])

    def test_invalid_round_assignment_and_confirmation_types_are_rejected(self):
        result = summarize(dict(events=[dict(id='1', rank='6')], review_confirmed=False))
        for number in [0, -1, 1001, True, '2', 1.5, None]:
            with self.subTest(number=number), self.assertRaises(ValueError):
                apply_review(result, [dict(id='1', rank='6', round_number=number)], True, True)
        with self.assertRaises(ValueError):
            apply_review(result, [dict(id='1', rank='6')], True, 'yes')


if __name__ == '__main__':
    unittest.main()
