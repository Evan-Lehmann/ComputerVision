"""Synthetic animation regressions; uploaded recording images stay outside Git."""
from types import SimpleNamespace
import unittest

import cv2
import numpy as np

from card_tracking import animation_similarity, CardIdentityMatcher, merge_track_fragments


def glyph(text, red=True):
    image = np.full((64, 60, 3), 255, dtype=np.uint8)
    image[:, :3] = (30, 45, 35)
    cv2.putText(image, text, (10, 46), cv2.FONT_HERSHEY_SIMPLEX,
                1, (45, 50, 200) if red else (15, 15, 15), 2, cv2.LINE_AA)
    return image


def dim(image, opacity=.4):
    overlay = np.full_like(image, (50, 60, 40))
    return cv2.addWeighted(image, opacity, overlay, 1-opacity, 0)


def track(i, frames, crop, x=0, region='player', scene=0):
    observations = [dict(frame=f, crop=crop.copy(), box=[x, 0, x+20, 25]) for f in frames]
    return SimpleNamespace(track_id=i, region=region, scene_id=scene,
                           observations=observations, last_box=observations[-1]['box'],
                           last_sample_index=frames[-1])


class DealAnimationTests(unittest.TestCase):
    def test_dim_moving_birth_rejoins_for_red_black_and_two_digits(self):
        for text, red in [('2', True), ('10', True), ('A', False), ('3', False)]:
            for opacity in [.35, .5, .65]:
                with self.subTest(text=text, red=red, opacity=opacity):
                    bright = glyph(text, red)
                    a = track(1, [0], dim(bright, opacity), x=80)
                    b = track(2, [1, 2, 3], bright)
                    merged = merge_track_fragments([a, b])
                    self.assertEqual(len(merged), 1)
                    self.assertEqual(merged[0].merged_from, [1, 2])
                    self.assertEqual([o['frame'] for o in merged[0].observations], [0, 1, 2, 3])

    def test_co_visible_same_rank_cards_never_merge(self):
        bright = glyph('2')
        a = track(1, [0, 1], dim(bright), x=80)
        b = track(2, [1, 2, 3], bright)
        self.assertEqual(len(merge_track_fragments([a, b])), 2)

    def test_ambiguous_same_rank_settling_destinations_remain_for_review(self):
        bright = glyph('2')
        a = track(1, [0], dim(bright), x=80)
        b = track(2, [1, 2, 3], bright, x=0)
        c = track(3, [1, 2, 3], bright, x=20)
        self.assertEqual(len(merge_track_fragments([a, b, c])), 3)

    def test_colour_rank_shape_and_digit_count_changes_are_not_animation_matches(self):
        for before, after in [(glyph('2'), glyph('3')), (glyph('3'), glyph('5')),
                              (glyph('10'), glyph('1')), (glyph('2'), glyph('2', False))]:
            self.assertLess(animation_similarity(dim(before), after), .96)
            self.assertEqual(len(merge_track_fragments([
                track(1, [0], dim(before), x=80), track(2, [1, 2], after)])), 2)

    def test_all_different_synthetic_ranks_fail_the_animation_shape_gate(self):
        ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
        for before in ranks:
            for after in ranks:
                if before != after:
                    with self.subTest(before=before, after=after):
                        self.assertLess(animation_similarity(dim(glyph(before)), glyph(after)), .96)

    def test_animation_repair_respects_time_distance_region_and_scene(self):
        bright = glyph('2')
        for options in [dict(frames=[4, 5]), dict(x=-100), dict(region='dealer'), dict(scene=1)]:
            with self.subTest(options=options):
                b = dict(frames=[1, 2], crop=bright)
                b.update(options)
                self.assertEqual(len(merge_track_fragments([
                    track(1, [0], dim(bright), x=80), track(2, **b)])), 2)

    def test_stable_tracks_do_not_get_animation_movement_allowance(self):
        bright = glyph('2')
        self.assertEqual(len(merge_track_fragments([
            track(1, [0], bright, x=80), track(2, [1, 2], bright)])), 2)
        self.assertEqual(len(merge_track_fragments([
            track(1, [0, 1, 2, 3], dim(bright), x=80), track(2, [4, 5], bright)])), 2)

    def test_blank_table_boundary_survives_fragment_repair(self):
        bright = glyph('2')
        matcher = CardIdentityMatcher(clear_after_empty=2)
        a = track(1, [0], bright, scene=matcher.scene_id)
        matcher.match([], [a], 1)
        matcher.match([], [a], 2)
        self.assertEqual(matcher.match([dict(box=a.last_box, crop=bright)], [a], 3), {})
        b = track(2, [3], bright, scene=matcher.scene_id)
        self.assertEqual(len(merge_track_fragments([a, b])), 2)

    def test_missing_blank_or_border_only_crops_cannot_trigger_repair(self):
        for crop in [None, np.zeros((0, 0, 3), dtype=np.uint8),
                     np.full((32, 32, 3), 255, dtype=np.uint8)]:
            self.assertEqual(animation_similarity(crop, glyph('2')), 0)


if __name__ == '__main__':
    unittest.main()
