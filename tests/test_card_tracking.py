import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from card_tracking import CardIdentityMatcher


FIXTURES = Path(__file__).parent / "fixtures"


class CardTrackingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads((FIXTURES / "card_tracking.json").read_text())
        with np.load(FIXTURES / "card_tracking_crops.npz", allow_pickle=False) as crops:
            cls.crops = {key: crops[key] for key in crops.files}

    def replay(self):
        by_frame = defaultdict(list)
        for record in self.fixture["records"]:
            by_frame[record["sample_index"]].append(record)
        tracks, assignments = [], {}
        matcher = CardIdentityMatcher()
        for sample in range(self.fixture["sample_count"]):
            records = by_frame[sample]
            detections = [dict(box=r["box"], crop=self.crops[r["key"]]) for r in records]
            matches = matcher.match(detections, tracks, sample)
            for index, record in enumerate(records):
                track = matches.get(index)
                if track is None:
                    track = SimpleNamespace(track_id=len(tracks) + 1, observations=[])
                    tracks.append(track)
                track.last_box = record["box"]
                track.last_sample_index = sample
                track.observations.append(dict(crop=self.crops[record["key"]]))
                assignments[record["key"]] = track.track_id
        return tracks, assignments

    def test_reviewed_identity_switches_are_removed(self):
        baseline, revised = defaultdict(set), defaultdict(set)
        tracks, assignments = self.replay()
        for r in self.fixture["records"]:
            if r["identity"] is not None:
                baseline[r["baseline_track_id"]].add(r["identity"])
                revised[assignments[r["key"]]].add(r["identity"])
        self.assertGreaterEqual(sum(len(ids) > 1 for ids in baseline.values()), 6)
        self.assertEqual({k: v for k, v in revised.items() if len(v) > 1}, {})

    def test_existing_card_moves_to_new_box_instead_of_becoming_new_card(self):
        _, ids = self.replay()
        for before, after in [("t2_f20", "t4_f25"),
                              ("t3_f20", "t5_f25"),
                              ("t12_f110", "t15_f135"),
                              ("t21_f200", "t22_f230"),
                              ("t19_f205", "t23_f255"),
                              ("t7_f55", "t10_f65")]:
            with self.subTest(before=before, after=after):
                self.assertEqual(ids[before], ids[after])

    def test_new_cards_do_not_inherit_old_card_identity(self):
        _, ids = self.replay()
        for before, after in [("t3_f20", "t3_f25"),
                              ("t12_f110", "t12_f115"),
                              ("t18_f200", "t18_f205"),
                              ("t18_f205", "t18_f235"),
                              ("t21_f200", "t21_f205")]:
            with self.subTest(before=before, after=after):
                self.assertNotEqual(ids[before], ids[after])

    def track(self, crop, box, sample=0, track_id=1):
        return SimpleNamespace(track_id=track_id, last_box=box,
                               last_sample_index=sample,
                               observations=[dict(crop=crop)])

    def test_two_same_rank_cards_remain_distinct(self):
        crop = self.crops["t20_f230"]
        boxes = [[0, 0, 15, 20], [35, 0, 50, 20]]
        tracks = [self.track(crop, box, track_id=i + 1) for i, box in enumerate(boxes)]
        detections = [dict(crop=crop, box=box) for box in reversed(boxes)]
        matches = CardIdentityMatcher().match(detections, tracks, 1)
        self.assertEqual([matches[i].track_id for i in range(2)], [2, 1])

    def test_ambiguous_or_missing_appearance_does_not_force_match(self):
        crop, box = self.crops["t20_f230"], [0, 0, 15, 20]
        track = self.track(crop, box)
        matcher = CardIdentityMatcher()
        self.assertEqual(matcher.match([dict(crop=None, box=box)], [track], 1), {})
        duplicates = [dict(crop=crop, box=box), dict(crop=crop, box=box)]
        self.assertEqual(matcher.match(duplicates, [track], 1), {})

    def test_blank_table_retires_tracks_before_same_rank_returns(self):
        crop, box = self.crops["t20_f230"], [0, 0, 15, 20]
        track = self.track(crop, box)
        matcher = CardIdentityMatcher()
        matcher.match([], [track], 1)
        matcher.match([], [track], 2)
        self.assertEqual(matcher.match([dict(crop=crop, box=box)], [track], 3), {})

    def test_single_detection_dropout_can_recover(self):
        crop, box = self.crops["t20_f230"], [0, 0, 15, 20]
        track = self.track(crop, box)
        matcher = CardIdentityMatcher()
        matcher.match([], [track], 1)
        self.assertIs(matcher.match([dict(crop=crop, box=box)], [track], 2)[0], track)


if __name__ == "__main__":
    unittest.main()
