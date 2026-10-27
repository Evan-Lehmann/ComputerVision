"""Recorded-table pipeline. Cards and round divisions remain provisional."""
from pathlib import Path
from types import SimpleNamespace
import json
import hashlib
import cv2
import numpy as np
from card_tracking import CardIdentityMatcher, merge_track_fragments
from round_segmentation import RoundSegmenter, visible_empty_table

ROOT = Path(__file__).resolve().parent
RANKS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
COUNT_VALUES = {rank: (-1 if rank in ['A', '10', 'J', 'Q', 'K'] else 1 if rank in ['2','3','4','5','6'] else 0) for rank in RANKS}
COUNT_VALUES['UNKNOWN'] = 0
# Canonical physical-card views in the supplied 1440x900 recordings. The small
# central hand is excluded by choosing disjoint source regions; its relationship
# to the main hand must be checked before treating it as a duplicate.
DEFAULT_REGIONS = [dict(name='dealer', rect=[.28, .76, .58, .93]),
                   dict(name='player', rect=[.58, .76, .84, .93])]


def validate_regions(regions):
    if not isinstance(regions, list) or not 1 <= len(regions) <= 10:
        raise ValueError('Choose between one and ten card regions.')
    cleaned = []
    for region in regions:
        name = str(region.get('name', '')).strip()
        rect = region.get('rect', [])
        if not name or len(rect) != 4 or not all(isinstance(v, (int, float)) and np.isfinite(v) for v in rect):
            raise ValueError('Each region needs a name and four finite relative coordinates.')
        x1, y1, x2, y2 = rect
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError('Region coordinates must lie between 0 and 1.')
        for previous in cleaned:
            a,b,c,d = previous['rect']
            if min(x2,c) > max(x1,a) and min(y2,d) > max(y1,b):
                raise ValueError('Card regions must not overlap.')
            if name == previous['name']:
                raise ValueError('Card region names must be unique.')
        cleaned.append(dict(name=name, rect=list(rect)))
    return cleaned


def crop_box(image, box):
    x1,y1,x2,y2 = box
    h,w = image.shape[:2]
    px,py = max(2, int((x2-x1)*.2)), max(2, int((y2-y1)*.2))
    crop = image[max(0,int(y1)-py):min(h,int(y2)+py), max(0,int(x1)-px):min(w,int(x2)+px)]
    return crop.copy() if crop.size else None


class VisionEngine:
    def __init__(self, detector_path=None, classifier_path=None):
        import torch
        from ultralytics import YOLO
        from rank_model import RankCNN, preprocess_crop
        from test_rank_detector_unseen import detect_full_frame
        detector_path = Path(detector_path or ROOT/'models/rank_detector_v2.pt')
        classifier_path = Path(classifier_path or ROOT/'models/rank_classifier_v4.pt')
        for path in (detector_path, classifier_path):
            if not path.is_file():
                raise FileNotFoundError('Missing model: ' + str(path))
        self.torch, self.preprocess, self.detect_full_frame = torch, preprocess_crop, detect_full_frame
        self.detector = YOLO(str(detector_path))
        checkpoint = torch.load(classifier_path, map_location='cpu', weights_only=True)
        if checkpoint.get('ranks') != RANKS or checkpoint.get('image_size') != 64:
            raise ValueError('Classifier class order or image size is incompatible.')
        self.model = RankCNN()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        torch.set_num_threads(min(torch.get_num_threads(), 4))

    def detect(self, image, path):
        return self.detect_full_frame(self.detector, image)[3]

    def classify(self, crops):
        if not crops:
            return []
        tensors = self.torch.cat([self.preprocess(crop) for crop in crops])
        with self.torch.no_grad():
            return self.model(tensors).softmax(1).cpu().numpy()


def summarize(result):
    included = [event for event in result['events'] if not event.get('excluded')]
    cards = [event['rank'] for event in included]
    subtotal = sum(COUNT_VALUES.get(rank, 0) for rank in cards)
    unresolved = sum(rank == 'UNKNOWN' for rank in cards)
    numbers = sorted({event.get('round_number', 1) for event in result['events']})
    needs_round_review = (len(numbers) > 1 or bool(result.get('boundaries'))
                          or bool(result.get('uncertain_transitions')))
    verified = (bool(result.get('review_confirmed')) and not unresolved
                and (not needs_round_review or bool(result.get('rounds_confirmed'))))
    rounds, running, running_unknown = [], 0, 0
    for number in numbers:
        events = [event for event in included if event.get('round_number', 1) == number]
        ranks = [event['rank'] for event in events]
        change = sum(COUNT_VALUES.get(rank, 0) for rank in ranks)
        unknown = ranks.count('UNKNOWN')
        running += change
        running_unknown += unknown
        rounds.append(dict(round_number=number, cards=ranks, unresolved=unknown,
            known_count_change=change, count_change=None if unknown else change,
            count_range=dict(min=change-unknown, max=change+unknown),
            running_known_count_change=running,
            running_count_change=None if running_unknown else running,
            running_count_range=dict(min=running-running_unknown, max=running+running_unknown)))
    result.update(cards=cards, known_count_change=subtotal, unresolved=unresolved,
                  count_change=None if unresolved else subtotal,
                  count_verified=verified, rounds=rounds, round_count=len(rounds),
                  needs_round_review=needs_round_review,
                  count_scope='captured_cards',
                  count_range=dict(min=subtotal-unresolved, max=subtotal+unresolved))
    return result


def process_frames(frame_paths, output_dir, engine=None, regions=None, progress=None, ocr_reader=None):
    """Process every frame; callers may inject an engine for replay/testing."""
    frame_paths = sorted(Path(p) for p in frame_paths)
    if not frame_paths:
        raise ValueError('No frames were found. Record a round or select a frame folder.')
    regions = validate_regions(regions if regions is not None else DEFAULT_REGIONS)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    engine = engine or VisionEngine()
    if ocr_reader is None:
        from rank_ocr import TesseractRankReader
        ocr_reader = TesseractRankReader()
    matchers = {r['name']: CardIdentityMatcher(clear_after_empty=4) for r in regions}
    tracks = {r['name']: [] for r in regions}
    next_id, read_frames = 1, 0
    expected_shape = None
    recording_digest = hashlib.sha256(b'cardvision-recording-v1\0')
    segmenter = RoundSegmenter()
    for index, path in enumerate(frame_paths):
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError('Could not read frame: ' + path.name)
        h,w = image.shape[:2]
        if expected_shape is None:
            expected_shape = (h,w)
        elif expected_shape != (h,w):
            raise ValueError('Frame sizes changed during capture. Record with one monitor size.')
        # Content-based grouping survives reanalysis, folder moves and renaming.
        recording_digest.update(np.asarray(image.shape, dtype='<u4').tobytes())
        recording_digest.update(image.tobytes())
        detections = engine.detect(image, path)
        grouped = {r['name']: [] for r in regions}
        for det in detections:
            box = [float(det[key]) for key in ['x1','y1','x2','y2']]
            cx,cy = (box[0]+box[2])/(2*w), (box[1]+box[3])/(2*h)
            for region in regions:
                x1,y1,x2,y2 = region['rect']
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    crop = crop_box(image, box)
                    if crop is not None:
                        grouped[region['name']].append(dict(box=box, crop=crop))
                    break
        has_cards = any(grouped.values())
        if segmenter.observe(index, has_cards,
                empty_table=not has_cards and visible_empty_table(image, regions)):
            for name, matcher in matchers.items():
                matcher.retired_ids.update(t.track_id for t in tracks[name])
                matcher.scene_id += 1
        for region in regions:
            name = region['name']
            current = grouped[name]
            probabilities = engine.classify([d['crop'] for d in current])
            matches = matchers[name].match(current, tracks[name], index)
            for i,detection in enumerate(current):
                track = matches.get(i)
                if track is None:
                    track = SimpleNamespace(track_id=next_id, observations=[], region=name,
                                            scene_id=matchers[name].scene_id,
                                            round_number=segmenter.round_number)
                    next_id += 1
                    tracks[name].append(track)
                probabilities_i = np.asarray(probabilities[i], dtype=float)
                if probabilities_i.shape != (13,) or not np.isfinite(probabilities_i).all():
                    raise ValueError('Classifier returned invalid probabilities.')
                track.last_box = detection['box']
                track.last_sample_index = index
                track.observations.append(dict(crop=detection['crop'], box=detection['box'],
                    probabilities=probabilities_i, frame=index, source=path.name))
        read_frames += 1
        if progress:
            progress(read_frames, len(frame_paths))
    events = []
    for regional_tracks in tracks.values():
        for track in merge_track_fragments(regional_tracks):
            observations = track.observations
            average = np.mean([o['probabilities'] for o in observations], axis=0)
            order = np.argsort(average)[::-1]
            confidence = float(average[order[0]])
            margin = float(average[order[0]] - average[order[1]])
            rank = RANKS[order[0]] if len(observations) >= 2 and confidence >= .8 and margin >= .2 else 'UNKNOWN'
            cnn_rank,cnn_confidence = RANKS[order[0]],confidence
            recognition_source,ocr_result = 'cnn',None
            if rank == 'UNKNOWN' and ocr_reader:
                ocr_result = ocr_reader.consensus(observations)
                if ocr_result:
                    rank = ocr_result['rank']
                    confidence = ocr_result['confidence']
                    recognition_source = 'ocr_consensus'

            evidence = []
            for label, observation in zip(['first','middle','last'], [observations[0], observations[len(observations)//2], observations[-1]]):
                filename = 'card_%04d_%s.png' % (track.track_id, label)
                if not cv2.imwrite(str(output_dir/filename), observation['crop']):
                    raise OSError('Could not write card evidence.')
                evidence.append(dict(image=filename, frame=observation['frame'], source=observation['source']))
            events.append(dict(id=str(track.track_id), region=track.region, rank=rank,
                round_number=track.round_number,
                merged_from=getattr(track, 'merged_from', [track.track_id]),
                suggested_rank=rank if ocr_result else cnn_rank, confidence=confidence, observations=len(observations),
                recognition_source=recognition_source, cnn_suggested_rank=cnn_rank,
                cnn_confidence=cnn_confidence, ocr_votes=ocr_result['votes'] if ocr_result else [],
                first_frame=observations[0]['frame'], last_frame=observations[-1]['frame'],
                evidence=evidence, excluded=False, reviewed=False,
                alternatives=[dict(rank=RANKS[i], probability=float(average[i])) for i in order[:3]]))
    # Surface possible fragmentation without silently deleting real short-lived
    # cards. A reviewer can exclude a duplicate while preserving its evidence.
    for event in events:
        candidates = [other['id'] for other in events
                      if other['id'] != event['id'] and other['region'] == event['region']
                      and other['round_number'] == event['round_number']
                      and other['suggested_rank'] == event['suggested_rank']
                      and max(event['first_frame'] - other['last_frame'],
                              other['first_frame'] - event['last_frame'], 0) <= 3]
        event['possible_duplicates'] = candidates
    # Within-frame ordering is deterministic, not a claim about sub-frame deal order.
    events.sort(key=lambda event: (event['first_frame'], event['region'], int(event['id'])))
    result = summarize(dict(events=events, frames_processed=read_frames, regions=regions,
        recording_digest=recording_digest.hexdigest(),
        boundaries=segmenter.boundaries, uncertain_transitions=segmenter.uncertain_transitions,
        rounds_confirmed=False,
        review_confirmed=False, warnings=[
            'Model predictions are provisional. Review every card and check for missing or duplicate cards.',
            'Cards first seen in the same frame have uncertain relative deal order.',
            'Capture may start or end mid-round; completion is not inferred. Check round assignments and boundaries.',
            'Only selected source regions are counted. The small central hand is excluded by default; add distinct cards from other views during review.'
        ]))
    if not events:
        result['warnings'].append('No ranks detected in the selected regions. Check the region preview before trusting an empty list.')
    result['ocr_available'] = bool(ocr_reader and ocr_reader.available)
    if ocr_reader and not ocr_reader.available:
        result['warnings'].append('Optional OCR cross-check is unavailable. Install Tesseract to enable it; unresolved ranks still need review.')
    if ocr_reader and ocr_reader.error:
        result['warnings'].append(ocr_reader.error)
    save_result(result, output_dir)
    return result


def save_result(result, output_dir):
    path = Path(output_dir)/'result.json'
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, indent=2), encoding='utf-8')
    temporary.replace(path)


def apply_review(result, edits, confirmed, rounds_confirmed=False):
    """Review an explicitly ordered event list; excluded items stay in the audit."""
    import copy
    if (not isinstance(edits, list) or len(edits) > 1000 or not isinstance(confirmed, bool)
            or not isinstance(rounds_confirmed, bool)):
        raise ValueError('Invalid review request.')
    known = {event['id']: event for event in result['events']}
    seen, events = set(), []
    for edit in edits:
        event_id = str(edit.get('id', ''))
        if event_id in seen:
            raise ValueError('Duplicate event ID.')
        seen.add(event_id)
        if event_id in known:
            event = copy.deepcopy(known[event_id])
        elif event_id.startswith('new-'):
            event = dict(id=event_id, region='manual', evidence=[], first_frame=None,
                last_frame=None, observations=0, suggested_rank='UNKNOWN', confidence=0, alternatives=[])
        else:
            raise ValueError('Unknown card ID.')
        rank = edit.get('rank')
        if rank not in RANKS + ['UNKNOWN'] or not isinstance(edit.get('excluded', False), bool):
            raise ValueError('Choose a valid rank.')
        number = edit.get('round_number', event.get('round_number', 1))
        if type(number) is not int or not 1 <= number <= 1000:
            raise ValueError('Round numbers must be whole numbers between 1 and 1000.')
        event.update(rank=rank, excluded=edit.get('excluded', False), reviewed=True,
                     round_number=number)
        events.append(event)
    if not set(known).issubset(seen):
        raise ValueError('Keep all original events; mark unwanted events excluded.')
    reviewed = copy.deepcopy(result)
    reviewed.update(events=events, review_confirmed=confirmed,
                    rounds_confirmed=rounds_confirmed and confirmed)
    return summarize(reviewed)
