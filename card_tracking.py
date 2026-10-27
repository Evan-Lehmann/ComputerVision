"""Conservative association of rank detections, independent of rank predictions.

Different UI representations remain separate tracks. This module does not infer
logical-card duplicates or recognize ranks. Ambiguous associations start a new
track rather than mixing observations from different cards.
"""

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from glyph_features import glyph_signature as _animation_glyph


def crop_appearance(crop):
    """Keep glyph intensity and colour without connected-component filtering."""
    if crop is None or crop.size == 0:
        return None
    h, w = crop.shape[:2]
    # Inputs already include 20% detector-box padding. Remove most of that
    # context for association only; classifier inputs remain untouched.
    tx, ty = int(w * 0.12), int(h * 0.12)
    inner = crop[ty:h - ty, tx:w - tx]
    if min(inner.shape[:2]) < 2:
        return None
    gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
    white = max(float(np.percentile(gray, 90)), 1.0)
    ink = np.clip((white - gray.astype(np.float32)) / white, 0, 1)
    canvas = cv2.resize(ink, (32, 32), interpolation=cv2.INTER_LINEAR)
    # Box aspect ratios fluctuate at this resolution. Compare in box-relative
    # coordinates and downweight peripheral card borders, without deleting
    # components or changing the recognition crop.
    axis = np.linspace(-1, 1, 32, dtype=np.float32)
    window = np.exp(-(axis[:, None] ** 2 + axis[None, :] ** 2) / 0.9)
    canvas *= window
    # A second view preserves aspect ratio. Requiring agreement with it avoids
    # mistaking a narrow/partial 10 for a newly dealt 6 after box stretching.
    scale = min(28.0 / inner.shape[1], 28.0 / inner.shape[0])
    nw, nh = max(1, round(inner.shape[1] * scale)), max(1, round(inner.shape[0] * scale))
    context = np.zeros((32, 32), dtype=np.float32)
    context[(32 - nh) // 2:(32 - nh) // 2 + nh,
            (32 - nw) // 2:(32 - nw) // 2 + nw] = cv2.resize(
                (255.0 - gray.astype(np.float32)) / 255.0,
                (nw, nh), interpolation=cv2.INTER_AREA)
    b, g, r = cv2.split(inner.astype(np.float32))
    red = (r > g + 35) & (r > b + 35)
    dark = gray < 180
    red_fraction = float(red.sum()) / max(int((red | dark).sum()), 1)
    return canvas, red_fraction, context


def appearance_similarity(a, b):
    if a is None or b is None:
        return 0.0
    aa, red_a, context_a = a
    bb, red_b, context_b = b
    if abs(red_a - red_b) > 0.45:
        return 0.0
    if min(float(np.linalg.norm(aa)), float(np.linalg.norm(bb))) < 0.1:
        return 0.0
    # Small detector-box jitter should not create a new identity.
    padded = np.pad(bb, 2)
    scores = cv2.matchTemplate(padded, aa, cv2.TM_CCORR_NORMED)
    context_scores = cv2.matchTemplate(np.pad(context_b, 2), context_a,
                                       cv2.TM_CCORR_NORMED)
    shape = float(scores.max())
    context = float(context_scores.max())
    # Rendering/border changes can alter the padded view while the glyph is
    # effectively unchanged. When central-shape agreement is very strong,
    # context is a continuous penalty rather than a hard border-dependent veto.
    # A partial 10 resembling 6 or K changing to 3 still fails the shape guard.
    score = 0.75 * shape + 0.25 * context if shape >= 0.90 else min(shape, context)
    return float(np.clip(score, 0.0, 1.0))


def _geometry(a, b):
    aw, ah = a[2] - a[0], a[3] - a[1]
    bw, bh = b[2] - b[0], b[3] - b[1]
    if min(aw, ah, bw, bh) <= 0:
        return None
    if max(aw / bw, bw / aw, ah / bh, bh / ah) > 1.8:
        return None
    dx = ((a[0] + a[2]) - (b[0] + b[2])) / 2
    dy = ((a[1] + a[3]) - (b[1] + b[3])) / 2
    # Card ranks can shift horizontally as a hand grows, but should not jump
    # between the small central hand and the larger player/dealer hands.
    if abs(dy) > max(ah, bh) * 0.7:
        return None
    return float(np.hypot(dx, dy) / max((aw + bw) / 2, (ah + bh) / 2))


def animation_similarity(early, late):
    """Evidence for a dim birth settling, never an arbitrary same-rank merge."""
    a, b = _animation_glyph(early), _animation_glyph(late)
    if a is None or b is None:
        return 0.0
    aa, aspect_a, red_a, white_a = a
    bb, aspect_b, red_b, white_b = b
    # Only a fading-in card, not an unrestricted alternative appearance metric.
    if (white_a >= 200 or white_b-white_a < 35 or red_a != red_b
            or max(aspect_a/aspect_b, aspect_b/aspect_a) > 1.25):
        return 0.0
    return float(cv2.matchTemplate(np.pad(bb, 2), aa, cv2.TM_CCORR_NORMED).max())


def glyph_similarity(a, b):
    """Compare normalized interior glyphs without predicting their ranks.

    Boundary pixels and absolute red thresholds are unreliable during fades.
    Preserve aspect and colour, and retain the older descriptor only when an
    interior glyph cannot be extracted. Weak matches cannot bypass a strong
    contradiction in the original view.
    """
    original = appearance_similarity(crop_appearance(a), crop_appearance(b))
    # Tiny miniature-hand crops lose discriminating strokes. Retain the
    # conservative original view instead of amplifying them to a false match.
    if a is None or b is None or min(a.shape[0], b.shape[0]) < 20:
        return original
    aa, bb = _animation_glyph(a), _animation_glyph(b)
    if aa is None or bb is None:
        return original
    if aa[2] != bb[2] or max(aa[1]/bb[1], bb[1]/aa[1]) > 1.4:
        return 0.0
    score = float(cv2.matchTemplate(np.pad(bb[0], 2), aa[0], cv2.TM_CCORR_NORMED).max())
    return score if original >= .65 or score >= .94 else min(score, original)


def _center(box):
    return np.array([(box[0]+box[2])/2, (box[1]+box[3])/2])


def _ordered_assignment(costs, active):
    """Sequence alignment: match, birth, or missing card; never swap order."""
    n, m = costs.shape
    score = np.full((n+1, m+1), np.inf)
    score[0, :] = 0
    steps = {}
    for i in range(1, n+1):
        score[i, 0] = i*.65
        steps[i, 0] = (i-1, 0, False)
        for j in range(1, m+1):
            options = [(score[i-1, j]+.65, i-1, j, False),
                       (score[i, j-1], i, j-1, False),
                       (score[i-1, j-1]+costs[i-1, j-1], i-1, j-1, True)]
            best = min(options, key=lambda value: value[0])
            score[i, j] = best[0]
            steps[i, j] = best[1:]
    matches, i, j = {}, n, m
    while i:
        pi, pj, matched = steps[i, j]
        if matched:
            matches[i-1] = active[j-1]
        i, j = pi, pj
    return matches


class CardIdentityMatcher:
    def __init__(self, max_gap_samples=3, max_distance=3.0,
                 min_similarity=0.78, ambiguity_margin=0.04,
                 clear_after_empty=2, ordered_hand=False):
        self.max_gap_samples = max_gap_samples
        self.max_distance = max_distance
        self.min_similarity = min_similarity
        self.ambiguity_margin = ambiguity_margin
        self.clear_after_empty = clear_after_empty
        self.empty_samples = 0
        self.retired_ids = set()
        self.scene_id = 0
        self.ordered_hand = ordered_hand

    def match(self, detections, tracks, sample_index):
        if not detections:
            self.empty_samples += 1
            if self.empty_samples >= self.clear_after_empty:
                self.retired_ids.update(t.track_id for t in tracks)
                if self.empty_samples == self.clear_after_empty:
                    self.scene_id += 1
            return {}
        self.empty_samples = 0
        active = [t for t in tracks
                  if t.track_id not in self.retired_ids
                  and 0 < sample_index - t.last_sample_index <= self.max_gap_samples
                  and t.observations]
        if not active:
            return {}
        boxes = [d['box'] for d in detections]+[t.last_box for t in active]
        heights = [box[3]-box[1] for box in boxes]
        ys = [_center(box)[1] for box in boxes]
        ordered_row = (self.ordered_hand and min(heights) >= 12
                       and max(ys)-min(ys) <= max(heights)*.7)
        detection_order = list(range(len(detections)))
        if ordered_row:
            detection_order.sort(key=lambda i: _center(detections[i]['box'])[0])
            detections = [detections[i] for i in detection_order]
            active.sort(key=lambda t: _center(t.last_box)[0])
        similarities = np.zeros((len(detections), len(active)))
        for i, detection in enumerate(detections):
            for j, track in enumerate(active):
                similarities[i, j] = min(
                    glyph_similarity(detection.get('crop'), observation.get('crop'))
                    if ordered_row else appearance_similarity(
                        crop_appearance(detection.get('crop')), crop_appearance(observation.get('crop')))
                    for observation in track.observations[-2:])
        motion = np.zeros(2)
        # Use distinctive, reciprocal glyph matches as hand-motion anchors.
        # This separates a shifting old card from a new card at its old box.
        if ordered_row and len(detections) >= len(active):
            anchors = []
            for j, track in enumerate(active):
                if track.last_sample_index != sample_index-1 or len(track.observations) < 2:
                    continue
                order = np.argsort(similarities[:, j])[::-1]
                i = int(order[0])
                if (similarities[i, j] >= .94
                        and (len(order) == 1 or similarities[i, j]-similarities[order[1], j] >= .05)
                        and int(np.argmax(similarities[i])) == j):
                    distance = _geometry(detections[i]['box'], track.last_box)
                    if distance is not None and distance <= self.max_distance:
                        anchors.append(_center(detections[i]['box'])-_center(track.last_box))
            # When identical cards offer no distinctive anchor, the configured
            # UI appends one new card on the right. An unbroken ordered prefix
            # with coherent movement can still establish hand translation.
            if not anchors and len(detections) == len(active)+1:
                prefix = [(_center(detections[j]['box'])-_center(t.last_box))
                          for j, t in enumerate(active)
                          if t.last_sample_index == sample_index-1 and len(t.observations) >= 2
                          and similarities[j, j] >= .94
                          and _geometry(detections[j]['box'], t.last_box) is not None
                          and _geometry(detections[j]['box'], t.last_box) <= self.max_distance]
                if len(prefix) == len(active):
                    anchors = prefix
            if anchors:
                median = np.median(anchors, axis=0)
                height = np.median([t.last_box[3]-t.last_box[1] for t in active])
                if max(np.linalg.norm(value-median) for value in anchors) <= height*.5:
                    motion = median
        # A private dummy column for every detection permits unmatched births.
        costs = np.full((len(detections), len(active) + len(detections)), 1e6)
        costs[:, len(active):] = 0.65
        for i, detection in enumerate(detections):
            for j, track in enumerate(active):
                distance = _geometry(detection["box"], track.last_box)
                if distance is None or distance > self.max_distance:
                    continue
                # Check the two most recent appearances. Keeping a permanently
                # blurred birth crop as an anchor fragments cards as they settle.
                similarity = similarities[i, j]
                if similarity < self.min_similarity:
                    continue
                if ordered_row:
                    shifted = [track.last_box[k]+motion[k % 2] for k in range(4)]
                    residual = _geometry(detection['box'], shifted)
                    distance = distance if residual is None else residual
                costs[i, j] = 1.0 - similarity + 0.12 * distance
        # Refuse near ties before global assignment. Predicted rank is never an
        # identity key: two cards with the same rank must remain distinct.
        real = costs[:, :len(active)].copy()
        for i in range(len(detections)):
            choices = np.sort(real[i][real[i] < 0.65])
            if len(choices) > 1 and choices[1] - choices[0] < self.ambiguity_margin:
                costs[i, :len(active)] = 1e6
        for j in range(len(active)):
            choices = np.sort(real[:, j][real[:, j] < 0.65])
            if len(choices) > 1 and choices[1] - choices[0] < self.ambiguity_margin:
                costs[:, j] = 1e6
        if ordered_row:
            matches = _ordered_assignment(costs[:, :len(active)], active)
        else:
            rows, columns = linear_sum_assignment(costs)
            matches = {int(i): active[j] for i, j in zip(rows, columns)
                       if j < len(active) and costs[i, j] < .65}
        return {detection_order[i]: track for i, track in matches.items()}


def merge_track_fragments(tracks, max_gap=3, min_similarity=0.90):
    """Rejoin disjoint observations of one card without merging co-visible cards.

    This is only same-region fragmentation repair. It deliberately does not
    reconcile separate UI views or use predicted ranks as physical identity.
    """
    tracks = list(tracks)
    changed = True
    while changed:
        changed = False
        candidates = []
        for i, left in enumerate(tracks):
            for j in range(i + 1, len(tracks)):
                right = tracks[j]
                if left.region != right.region:
                    continue
                if getattr(left, 'scene_id', None) != getattr(right, 'scene_id', None):
                    continue
                if getattr(left, 'round_number', 1) != getattr(right, 'round_number', 1):
                    continue
                a, b = left.observations, right.observations
                af, bf = {o['frame'] for o in a}, {o['frame'] for o in b}
                if af & bf:
                    continue
                gap = max(min(af) - max(bf), min(bf) - max(af), 0)
                if gap > max_gap:
                    continue
                overlap = max(min(af), min(bf)) <= min(max(af), max(bf))
                # Interleaved fragments should occupy almost the same position;
                # adjacent fragments may move as an overlapping hand grows.
                limit = 0.6 if overlap else 3.0
                best = 0.0
                for oa in a[:2] + a[-2:]:
                    for ob in b[:2] + b[-2:]:
                        distance = _geometry(oa['box'], ob['box'])
                        if distance is None or distance > limit:
                            continue
                        similarity = glyph_similarity(oa['crop'], ob['crop'])
                        best = max(best, similarity)
                # Short, consecutive animation births can move farther than a
                # stable card and change colour under the table's dim overlay.
                # Require a tight full-glyph match, not the classifier's rank.
                early, late = (a, b) if max(af) < min(bf) else (b, a)
                if not overlap and gap <= 2 and len(early) <= 3:
                    distance = _geometry(early[-1]['box'], late[0]['box'])
                    if distance is not None and distance <= 4.0:
                        score = animation_similarity(early[-1]['crop'], late[0]['crop'])
                        if score >= .96:
                            best = max(best, score)
                if best >= min_similarity:
                    candidates.append((best, i, j))
        if not candidates:
            break
        # Only take an unambiguous best partner for either fragment.
        candidates.sort(reverse=True)
        for score, i, j in candidates:
            alternatives = [s for s, x, y in candidates if (x, y) != (i, j)
                            and (i in (x, y) or j in (x, y))]
            if alternatives and score - max(alternatives) < 0.03:
                continue
            left, right = tracks[i], tracks[j]
            left.observations = sorted(left.observations + right.observations,
                                       key=lambda o: o['frame'])
            left.merged_from = getattr(left, 'merged_from', [left.track_id]) + getattr(
                right, 'merged_from', [right.track_id])
            left.last_box = left.observations[-1]['box']
            left.last_sample_index = left.observations[-1]['frame']
            tracks.pop(j)
            changed = True
            break
    return tracks
