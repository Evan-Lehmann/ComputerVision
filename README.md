# CardVision

A local school-project dashboard for reviewing cards from one recorded play-money blackjack round and calculating its Hi-Lo count change.

**Current status: working record → analyze → review → count prototype. Automatic recognition and tracking still make mistakes. An unreviewed count is provisional; unknown cards make it incomplete.** Continuous shoe tracking and betting signals are not implemented.

## Run on your Mac

From the project root, use your existing virtual environment:

```sh
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m cardvision_webapp.app
```

Open `http://127.0.0.1:5000` locally on your Mac. The models are included in `models/`. Keep the server bound to your own machine.

1. Expand **Check which cards are counted**, load a preview and check the hand boxes. Defaults match the supplied 1440×900 recordings. Draw new boxes if your browser/table layout differs. Include space for additional cards; exclude small duplicate views.
2. Start recording before cards appear. The app gives a three-second countdown; switch to the play-money table. It records at two frames per second by default.
3. Stop after the round's face-up cards have appeared. Analysis runs in the background and reports progress.
4. Inspect each card's first/middle/last crop. Click a crop to open its original screenshot. Correct incorrect/unknown ranks, exclude duplicate or false detections, add missed cards, and adjust order. All original detections remain in the saved audit even when excluded.
5. Check the confirmation box only after checking the whole round, then save the review. Download the JSON result if wanted.

Hi-Lo: A/10/J/Q/K = −1; 2–6 = +1; 7–9 = 0. UNKNOWN contributes zero to the **known-card subtotal**, but the full count is unavailable until all included ranks are resolved. “Reviewed by you” records user confirmation, not a claim that the models independently achieved perfect accuracy.

Screenshots and analysis are saved under ignored `round_captures/`. They are not uploaded by the app. Existing recordings are preserved. Different recordings use distinct folders.

### macOS screen recording

Allow Screen Recording for the application launching Python (for example VS Code or Terminal) in System Settings → Privacy & Security → Screen Recording. Restart that application if required. The app now reports capture failures instead of pretending an empty recording succeeded.

Choose another monitor or port if needed:

```sh
python -m cardvision_webapp.app --monitor 2 --fps 2 --port 5002
```

The implementation was validated on Linux with recorded frames and a simulated capture source. Actual capture and dependency installation on the original Apple M2 / Python 3.9.6 machine still require a live check.

## Analyze existing screenshots

Register a folder at startup; choose it in **Analyze saved screenshots**:

```sh
python -m cardvision_webapp.app --replay captures
```

`--replay` can be repeated for multiple folders. Prefer one complete round per folder. The app never deletes input screenshots. Multi-round segmentation and automatic round-completeness checks are not implemented.

For command-line processing:

```sh
python process_recording.py captures --output round_captures/replay_analysis
```

This writes a provisional `result.json` and card-evidence PNGs. A custom region JSON can be passed using `--regions`:

```json
[
  {"name": "dealer", "rect": [0.28, 0.76, 0.58, 0.93]},
  {"name": "player", "rect": [0.58, 0.76, 0.84, 0.93]}
]
```

Coordinates are fractions of screenshot width/height. Regions must not overlap. The central miniature hand is excluded by these defaults. Its relationship to the larger hand is not yet established for every recorded state; if it contains distinct physical cards, those must be added during review or included using an additional disjoint region. Do not automatically deduplicate cards just because their ranks match.

Outputs under a reused CLI output directory are overwritten; input frames are not. The web app uses unique output directories.

## Share a recording for debugging

If a recording ZIP exceeds the workspace transfer limit, split it from the project root:

```sh
python split_recording_upload.py /path/to/round.zip
```

Upload the `part_01.zip`, `part_02.zip`, etc. printed by the command. Each is an independent ZIP under 24 MiB, with original filenames and analysis files retained. The original ZIP is untouched. No dependencies beyond Python are required.

## Validation

```sh
python -m unittest discover -s tests -v
```

See [VALIDATION.md](VALIDATION.md) for measured results, failures, dataset separation and limits; [TRACKING.md](TRACKING.md) for the tracking experiment. The current app uses `models/rank_classifier_v4.pt`. The reproducible training experiments are `train_rank_classifier_v3.py` and `train_rank_classifier_v4.py`.

The YOLO localization/dedupe implementation remains unchanged. Historical training scripts and their data stay available. The web app no longer calls the inaccurate legacy round recognizer.
