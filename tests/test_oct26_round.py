"""Fresh-recording regressions; this round became development data after failure."""
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
from card_tracking import CardIdentityMatcher, merge_track_fragments
from round_pipeline import COUNT_VALUES

FIXTURES = Path(__file__).parent/'fixtures'

class October26RoundTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = json.loads((FIXTURES/'oct26_round.json').read_text())
        with np.load(FIXTURES/'oct26_round_crops.npz', allow_pickle=False) as file:
            cls.crops = {key:file[key] for key in file.files}

    def replay(self):
        frames = defaultdict(list)
        for row in self.reference['records']:
            frames[row['sample_index']].append(row)
        tracks = {'dealer': [], 'player': []}
        matchers = {region:CardIdentityMatcher(clear_after_empty=4) for region in tracks}
        next_id = 1
        for frame in range(self.reference['sample_count']):
            for region in tracks:
                rows = [row for row in frames[frame] if row['region']==region]
                detections = [dict(box=row['box'],crop=self.crops[row['key']]) for row in rows]
                matches = matchers[region].match(detections,tracks[region],frame)
                for index,row in enumerate(rows):
                    track = matches.get(index)
                    if track is None:
                        track = SimpleNamespace(track_id=next_id,region=region,observations=[])
                        next_id+=1
                        tracks[region].append(track)
                    track.last_box=row['box'];track.last_sample_index=frame
                    track.observations.append(dict(frame=frame,box=row['box'],crop=self.crops[row['key']],identity=row['identity'],rank=row['rank']))
        return [track for regional in tracks.values() for track in merge_track_fragments(regional)]

    def test_nine_fragment_rows_become_five_physical_cards(self):
        tracks = self.replay()
        self.assertEqual(len(tracks),5)
        self.assertEqual(sum(len(track.observations) for track in tracks),246)
        for track in tracks:
            self.assertEqual(len({o['identity'] for o in track.observations}),1)
        tracks.sort(key=lambda t:t.observations[0]['frame'])
        ranks=[t.observations[0]['rank'] for t in tracks]
        self.assertEqual(ranks,['K','7','3','8','2'])
        self.assertEqual(sum(COUNT_VALUES[rank] for rank in ranks),1)

    def test_a_new_three_at_the_old_king_position_is_not_merged(self):
        player=[track for track in self.replay() if track.region=='player']
        self.assertEqual(len(player),2)
        self.assertEqual({t.observations[0]['rank'] for t in player},{'K','3'})
        # The two distinct cards are simultaneously visible after dealing.
        frames=[{o['frame'] for o in track.observations} for track in player]
        self.assertGreater(len(frames[0]&frames[1]),0)

if __name__=='__main__':unittest.main()
