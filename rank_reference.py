"""Local reference matching for uncertain CNN ranks, with temporal agreement.

References come exclusively from the previously published V4 training crops.
A score is image similarity, not a calibrated probability. The reader cannot
change a confident CNN result or resolve a different CNN top-rank suggestion.
"""
from pathlib import Path
import cv2
import numpy as np
from glyph_features import glyph_signature

RANKS = ['A','2','3','4','5','6','7','8','9','10','J','Q','K']
DEFAULT_BANK = Path(__file__).resolve().parent/'models/rank_reference_v1.npz'


class ReferenceRankReader:
    def __init__(self, path=None, min_similarity=.97, min_margin=.04):
        self.error = None
        self.available = False
        self.min_similarity, self.min_margin = min_similarity, min_margin
        try:
            with np.load(path or DEFAULT_BANK, allow_pickle=False) as data:
                self.features = data['features'].astype(np.float32)
                self.aspects = data['aspects'].astype(float)
                self.ranks = data['ranks'].astype(str)
                self.groups = data['groups'].astype(str)
                self.keys = data['keys'].astype(str)
                self.version = str(data['version'].item())
            n = len(self.ranks)
            if (self.features.shape != (n,32,32) or self.aspects.shape != (n,)
                    or self.groups.shape != (n,) or self.keys.shape != (n,)
                    or set(self.ranks) != set(RANKS) or self.version != '1'
                    or not np.isfinite(self.features).all() or not np.isfinite(self.aspects).all()
                    or np.any(self.aspects <= 0) or np.any(np.linalg.norm(self.features,axis=(1,2)) <= 0)):
                raise ValueError('Invalid reference bank.')
            self.padded = [np.pad(feature, 2) for feature in self.features]
            self.available = True
        except (OSError, ValueError, KeyError):
            self.error = 'Rank references are unavailable or invalid; unresolved cards still need review.'

    def read(self, crop, excluded_group=None):
        if not self.available:
            return None
        glyph = glyph_signature(crop)
        if glyph is None:
            return None
        scores = {}
        for index, (feature, aspect, rank, group) in enumerate(zip(
                self.padded, self.aspects, self.ranks, self.groups)):
            if group == excluded_group or max(glyph[1]/aspect, aspect/glyph[1]) > 1.4:
                continue
            score = float(np.clip(cv2.matchTemplate(feature, glyph[0], cv2.TM_CCORR_NORMED).max(),0,1))
            if rank not in scores or score > scores[rank][0]:
                scores[rank] = (score, index)
        order = sorted(scores, key=lambda rank: scores[rank][0], reverse=True)
        if len(order) < 2:
            return None
        rank, runner_up = order[:2]
        score, index = scores[rank]
        margin = score-scores[runner_up][0]
        return dict(rank=rank, score=score, margin=margin,
            accepted=score >= self.min_similarity and margin >= self.min_margin,
            reference_key=str(self.keys[index]), reference_group=str(self.groups[index]),
            competing_rank=runner_up, competing_score=scores[runner_up][0])

    def consensus(self, observations, cnn_rank):
        if not self.available or len(observations) < 2:
            return None
        frames, votes = set(), []
        # First/middle/last alone can all land on unfavourable detector crops.
        # Bound work to seven evenly spread views, retaining those three views.
        # Acceptance thresholds and rejection of conflicting votes are unchanged.
        indices = {min(len(observations)-1, i*len(observations)//6) for i in range(7)}
        for index in sorted(indices):
            observation = observations[index]
            if observation['frame'] in frames:
                continue
            frames.add(observation['frame'])
            vote = self.read(observation['crop'])
            if vote and vote['accepted']:
                votes.append(dict(vote,frame=observation['frame']))
        if len(votes) < 2 or len({vote['rank'] for vote in votes}) != 1 or votes[0]['rank'] != cnn_rank:
            return None
        return dict(rank=cnn_rank, confidence=min(vote['score'] for vote in votes),
                    votes=votes, bank_version=self.version)
