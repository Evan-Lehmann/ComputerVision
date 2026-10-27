# Card identity tracking

The temporal experiment now associates detections using crop appearance,
geometry and a global one-to-one assignment. It checks the two most recent crops to resist sudden identity changes while
allowing the initial deal animation to settle. Rank classifier predictions are not used
as identity keys. Matching rejects ambiguous candidates instead of forcing a
card-to-card transition, and two consecutive empty sampled frames retire old
tracks. A single empty sampled frame can recover.

The current web pipeline uses this matcher, with four consecutive empty
frames as its retirement heuristic, followed by conservative fragment repair.
For a short dim birth followed by a bright settled card, repair additionally
compares background-normalized interior glyphs and allows a limited larger
move. It retains all observations and merged track IDs. No co-visible tracks
are joined, competing partners are rejected, and scene IDs prevent repair from
undoing a table-clear retirement. See VALIDATION.md for the six-recording
comparison and the remaining unresolved fragment.

The YOLO detector, tile dedupe, recognition preprocessing, model weights and
rank acceptance thresholds are unchanged. Different UI representations remain
separate tracks; logical-card reconciliation is still required before counting.

## Install and run

Use the existing project virtual environment:

```sh
python -m pip install -r requirements-tracking.txt
python -m unittest discover -s tests -v
```

The full temporal experiment additionally requires the existing PyTorch,
Ultralytics and trained models. Default paths still match the original project.
Paths can also be supplied explicitly:

```sh
python test_temporal_rank_tracking.py \
  --input captures \
  --detector models/rank_detector_v2.pt \
  --classifier models/rank_classifier_v4.pt \
  --output temporal_rank_test \
  --every 5
```

Optionally set `--threads 2` to limit CPU parallelism. The script writes:

- `tracks.csv`: aggregate results for tracks with at least three observations.
- `all_tracks.csv`: all tracks, with explicit short/unknown/accepted status.
- `observations.csv`: per-observation classifier predictions.
- `track_summary.jpg`: first/middle/last crops for every track, including short
  tracks. These crops preserve aspect ratio. A single confident crop must not
  be treated as ground truth for a whole track.

Existing files with these names in the output directory are overwritten.

## Validation and limits

Seven regression tests cover recorded card transitions, moving cards, two
same-rank cards, ambiguous input, missing appearance, table clearing and a
single detection dropout. The fixtures contain 179 tiny rank crops from the
uploaded 300-frame sequence, not full screenshots or model checkpoints.
Physical identity annotations cover 74 reviewed snapshots; null identities
mean unannotated observations. Mini-hand and main-hand identities are separate.

On these reviewed snapshots, the original matcher mixed different physical
identities in eight tracks; the new matcher mixed none. A full replay of the
300 frames (60 sampled frames) confirmed this result. There were 34 total tracks
instead of 26, with 21 tracks reaching three observations instead of 19.
This increase includes fragmentation: the change is deliberately conservative
and does not establish perfect tracking. More accepted ranks do not establish
higher recognition accuracy.

These examples informed the matcher and are regression evidence, not an
independent test set. Recognition errors such as 6 predicted as 5 remain. Short
tracks can still represent genuine cards, so count calculations must not drop
them silently. Two empty detection samples are a tracking reset heuristic,
not shoe-change detection. The historical numbers above describe the initial
experiment; the Flask app now uses this tracking pipeline for review-assisted
round analysis. Mac execution has not been tested in this cloud environment.

The Oct-26 follow-up adds a strong central-glyph agreement gate to tolerate
background/border changes. The added fresh-recording regression reduces nine
fragment rows to five physical cards while preserving the original different-card
regressions. See VALIDATION.md for the development-data scope and remaining limits.

When the central glyph matches strongly, padded-context similarity now acts as
a continuous penalty rather than a hard veto. This reduces track splits caused
by card-border changes while retaining the existing strong-shape guard. The
rank classifier and its acceptance thresholds are unchanged by this tracking
revision; unresolved recognition remains explicitly unknown.
