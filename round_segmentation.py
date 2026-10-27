"""Conservative round boundaries for the configured green play-money table.

Detection absence alone is insufficient: require consecutive visible felt frames
without upright light/red card-sized components. This is a layout heuristic,
not general scene recognition, round completion or shoe-change detection.
"""
import cv2
import numpy as np


def visible_empty_table(image, regions):
    h, w = image.shape[:2]
    for region in regions:
        x1, y1, x2, y2 = region['rect']
        crop = image[int(y1*h):int(y2*h), int(x1*w):int(x2*w)].astype(np.float32)
        if not crop.size:
            return False
        b, g, r = cv2.split(crop)
        felt = (g > r+3) & (g > b+1) & (g >= 20)
        if float(felt.mean()) < .25:
            return False
        # Reject card faces/backs even when the rank detector drops out. Upright
        # compact components distinguish cards from diffuse table reflections.
        light = (crop.min(axis=2) > 90) & (crop.max(axis=2)-crop.min(axis=2) < 45)
        red = (r > g*1.1) & (r > b*1.1) & (r > 90)
        mask = (light | red).astype(np.uint8)
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        for _, _, cw, ch, area in stats[1:]:
            if (cw >= max(3, w*.01) and ch >= max(5, h*.025)
                    and .35 <= cw/ch <= 1.0 and area/(cw*ch) >= .6):
                return False
    return True


class RoundSegmenter:
    def __init__(self, min_clear_frames=4):
        self.min_clear_frames = min_clear_frames
        self.round_number = 1
        self.last_seen = None
        self.gap_start = None
        self.clear_streak = 0
        self.clear_start = None
        self.boundaries = []
        self.uncertain_transitions = []

    def observe(self, frame, has_cards, empty_table=False):
        """Return whether this observation starts a new round."""
        if not has_cards:
            if self.last_seen is not None:
                if self.gap_start is None:
                    self.gap_start = frame
                self.clear_streak = self.clear_streak+1 if empty_table else 0
                if not self.clear_streak:
                    self.clear_start = None
                elif self.clear_streak == 1:
                    self.clear_start = frame
            return False
        new_round = False
        if self.last_seen is not None and self.gap_start is not None:
            if self.clear_streak >= self.min_clear_frames:
                self.round_number += 1
                self.boundaries.append(dict(round_number=self.round_number,
                    previous_last_frame=self.last_seen, first_frame=frame,
                    clear_start=self.clear_start, clear_end=frame-1,
                    source='visible_empty_table', confirmed=False))
                new_round = True
            elif frame-self.gap_start >= self.min_clear_frames or self.clear_streak:
                self.uncertain_transitions.append(dict(gap_start=self.gap_start,
                    gap_end=frame-1, first_frame=frame,
                    reason='No sustained visible empty table; review round assignments.'))
        self.last_seen = frame
        self.gap_start = None
        self.clear_streak = 0
        self.clear_start = None
        return new_round
