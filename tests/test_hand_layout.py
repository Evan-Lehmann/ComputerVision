"""Hand movement and identity regressions, not only final count checks."""
import unittest
from types import SimpleNamespace
import cv2
import numpy as np
from card_tracking import CardIdentityMatcher, glyph_similarity, merge_track_fragments
from glyph_features import glyph_signature


def glyph(text, red=True):
    crop = np.full((64, 60, 3), 255, np.uint8)
    crop[:3] = 35
    crop[:, :3] = 35
    cv2.putText(crop, text, (10,46), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (40,45,200) if red else (15,15,15), 2, cv2.LINE_AA)
    return crop


def track(i, text, x, frames=(0,1)):
    box = [x,0,x+15,20]
    return SimpleNamespace(track_id=i, last_box=box, last_sample_index=frames[-1],
        region='player', observations=[dict(crop=glyph(text), box=box, frame=f) for f in frames])


def detection(text,x):
    return dict(crop=glyph(text),box=[x,0,x+15,20])


class HandLayoutTests(unittest.TestCase):
    def test_old_four_follows_left_shift_instead_of_new_king_at_old_position(self):
        old = track(1,'4',40)
        ds = [detection('K',40),detection('4',5)]  # Detector output is unordered.
        matches = CardIdentityMatcher(ordered_hand=True).match(ds,[old],2)
        self.assertEqual(set(matches),{1})
        self.assertIs(matches[1],old)

    def test_identical_new_rank_on_right_does_not_steal_existing_card(self):
        old = track(1,'7',40)
        ds = [detection('7',40),detection('7',5)]
        matches = CardIdentityMatcher(ordered_hand=True).match(ds,[old],2)
        self.assertEqual(set(matches),{1})
        self.assertIs(matches[1],old)

    def test_shared_hand_translation_keeps_two_existing_identities(self):
        first,second = track(1,'4',40),track(2,'K',75)
        ds = [detection('7',75),detection('K',40),detection('4',5)]
        matches = CardIdentityMatcher(ordered_hand=True).match(ds,[second,first],2)
        self.assertEqual({i:t.track_id for i,t in matches.items()},{1:2,2:1})

    def test_layout_rejects_identity_crossing_instead_of_swapping_cards(self):
        first,second = track(1,'4',0),track(2,'K',35)
        matches = CardIdentityMatcher(ordered_hand=True).match(
            [detection('K',0),detection('4',35)],[first,second],2)
        ordered = sorted(matches.items())
        self.assertLess(len(matches),2)
        self.assertEqual([t.track_id for _,t in ordered], sorted(t.track_id for _,t in ordered))

    def test_dropout_and_both_directions_of_detector_order_preserve_ids(self):
        first,second = track(1,'4',0),track(2,'K',35)
        matcher = CardIdentityMatcher(ordered_hand=True)
        matcher.match([], [first,second], 2)
        matches = matcher.match([detection('K',35),detection('4',0)],[second,first],3)
        self.assertEqual({i:t.track_id for i,t in matches.items()},{0:2,1:1})

    def test_compact_edge_digit_is_preserved_while_spanning_border_is_removed(self):
        # A detector crop clips the 0 at its right edge. It is still a full
        # glyph component, unlike the spanning dark top/left card border.
        crop = np.full((30,32,3),255,np.uint8)
        crop[:3] = 35;crop[:,:3] = 35
        cv2.putText(crop,'10',(7,24),cv2.FONT_HERSHEY_SIMPLEX,.65,(10,10,10),1,cv2.LINE_AA)
        signature = glyph_signature(crop)
        self.assertIsNotNone(signature)
        self.assertGreater(signature[1],.8)
        self.assertLess(glyph_similarity(glyph('4'),glyph('K')),.9)

    def test_fragments_in_separate_rounds_cannot_merge_even_with_identical_glyphs(self):
        a,b = track(1,'7',0,frames=(0,)),track(2,'7',0,frames=(1,))
        a.round_number,b.round_number = 1,2
        self.assertEqual(len(merge_track_fragments([a,b])),2)


if __name__=='__main__':
    unittest.main()
