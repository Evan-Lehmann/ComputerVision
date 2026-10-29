# CardVision

A local school-project dashboard for reviewing cards across recorded play-money blackjack rounds and calculating their Hi-Lo changes and running total.

**Current status: working record → analyze → review → count prototype, with conservative multi-round segmentation. Automatic recognition, tracking and boundaries still make mistakes. An unreviewed count is provisional; unknown cards make it incomplete.** Counts update after recording stops. Continuous live shoe tracking and betting signals are not implemented.

## Run on your Mac

From the project root, use your existing virtual environment:

```sh
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m cardvision_webapp.app
```

Open `http://127.0.0.1:5000` locally on your Mac. The models are included in `models/`. Keep the server bound to your own machine.

1. Expand **Check which cards are counted**, load a preview and check the hand boxes. Defaults match the supplied 1440×900 recordings and select the large player/dealer hands. Draw new boxes if the layout differs. Inspect excluded views separately: the miniature hand can contain additional cards, and its physical-card relationship is not yet resolved.
2. Start recording before cards appear. The app gives a three-second countdown; switch to the play-money table. It records at two frames per second by default.
3. For multiple rounds, keep recording through the cleared table between deals. Stop after the last round's face-up cards have appeared. Analysis runs in the background and reports progress.
4. Inspect each card's first/middle/last crop. Click a crop to open its original screenshot. Correct incorrect/unknown ranks, exclude duplicate or false detections, add missed cards, and adjust order. All original detections remain in the saved audit even when excluded.
5. Check each round's cards and suggested clearing evidence in **Rounds & running total**. Correct a missed or false boundary using the **Round** number beside each card; add missed cards to the appropriate round. Confirm the card list and, when requested, the round divisions, then save the review. Download the JSON result if wanted.

Hi-Lo: A/10/J/Q/K = −1; 2–6 = +1; 7–9 = 0. UNKNOWN contributes zero to the **known-card subtotal**, but the full count is unavailable until all included ranks are resolved. “Reviewed by you” records user confirmation, not a claim that the models independently achieved perfect accuracy.

The displayed count covers cards observed in this recording, including cards already visible when recording starts. It starts at zero and carries across rounds within the recording. Separate recordings are not added together, and a partial recording is not a complete round. With unknown ranks, the app labels the number as a subtotal and shows the possible range for the detected events. An unknown in an earlier round keeps later running totals incomplete. Missing or duplicate cards can change that range.

A suggested new round requires at least four consecutive frames with no selected-region detections and visual evidence of an empty green table in every selected region. At the default two frames per second this is approximately two seconds. Bright upright card faces/backs veto the empty-table check even if their ranks are missed. Leading/trailing empty frames do not add empty rounds. Short clear intervals, long detector gaps and window changes can require manual review; they do not silently reset the total. This heuristic targets the supplied green-table layout and may miss boundaries in other layouts or under occlusion. Review is required. A round boundary never resets the recording total; shoe changes are not detected, and a shoe-spanning recording cannot automatically exclude/reset its earlier cards. Begin a new recording for a new shoe.

After confirming and saving a fully resolved review, **Download reviewed crops** exports a compact ZIP containing card crops, `labels.csv` and grouping metadata. Excluded cards and manually added cards without images are omitted; identical images within one card are deduplicated. Full screenshots are not included. Nothing is trained or uploaded automatically. Keep all views of a card and recording together in future training/validation splits. The recording identifier uses decoded frame content, so reanalyzing the same frames under another folder name preserves its group; overlapping or shortened recordings still need separate leakage checks.

Screenshots and analysis are saved under ignored `round_captures/`. They are not uploaded by the app. Existing recordings are preserved. Different recordings use distinct folders.

### Hand-aware association and rank references

The current pipeline preserves card order within a configured hand row and estimates shared hand movement. This helps keep an existing card when a new one arrives near its old position. The supplied UI appends new cards on the right; other dealing layouts, splits and overlapping views are not established as supported. Tiny or multiple-row inputs fall back to conservative association. Normalized glyph signatures tolerate tinted backgrounds and detector jitter while preserving compact digits at crop edges. The YOLO V2 detector remains unchanged. The default CNN is now V5; earlier V4 weights are retained for comparison.

Visible rectangular card faces provide additional evidence in the configured upright layout. Partial boxes can be reconciled only with a retained counterpart on the same face; narrow suit-edge suppression also requires agreement of the printed rank corners. Opposite-corner evidence can rejoin short dim animation fragments, while face-relative projection can recover a stable corner when a detector box clips it. Unestablished geometry leaves candidates available for review. The JSON preserves suppressed boxes/counterparts, inferred face boxes, original observation boxes and projected-crop provenance. No card is discarded merely for having one observation or a low rank score.

The three new development rounds return 18 distinct cards with no unknowns and count changes 0, +2 and -2. These recordings also supplied V5 training crops, so this is a development regression result. The frozen candidate also passes the first previously uninspected round, r4 (five exact cards, count 0, no unknowns); r5 remains reserved and uninspected. One held-out success does not establish general accuracy. See [VALIDATION.md](VALIDATION.md) for remaining failures and the frozen comparison protocol.

When the CNN leaves a rank unknown, the app can compare it against `models/rank_reference_v1.npz`, built only from the previously published V4 training crops. It requires a strong match, separation from other ranks, agreement in at least two distinct frames sampled from up to seven spread-out observations, and the same top rank as the CNN. Conflicting accepted reference votes leave the rank unresolved. Confident CNN results are preserved. Rows say **Reference agreement** and show an image-match score, which is not a probability of correctness. This reader handles red and black glyphs and needs no Tesseract installation. Missing references leave the remaining CNN/OCR path available with a warning.

The new two-round development recording now returns the 12 main-view cards, with changes −1 and +2, totaling +1. This recording informed development; it is not proof of accuracy on new captures. The miniature-view ambiguity remains, and older recordings still have an unresolved rank and a duplicate fragment. See [VALIDATION.md](VALIDATION.md).

Rebuild/audit the reference signatures from the existing training split:

```sh
python build_rank_reference.py
```

For comparison, `process_recording.py` accepts `--no-references` independently of `--no-ocr`. Use `--classifier models/rank_classifier_v4.pt --no-face-evidence` to reproduce the previous pipeline behavior while retaining the same source regions. `--no-face-evidence` disables face suppression, secondary-corner repair and projected crops together.

### Optional OCR cross-check for uncertain red ranks

A local pretrained Tesseract reader can resolve an unknown **red** rank when at least two distinct first/middle/last frames agree at 90% or greater OCR confidence. It does not override a confident CNN result. The row says **OCR consensus**, and the JSON keeps both the CNN suggestion and OCR votes. Results remain provisional until reviewed.

On a Mac with Homebrew:

```sh
brew install tesseract
tesseract --version
```

Then restart Flask. Tesseract runs entirely on your machine; no screenshot is sent to an OCR service. The English language data must be installed (included by the Homebrew package). If Homebrew is unavailable, follow the [official Tesseract installation instructions](https://tesseract-ocr.github.io/tessdoc/Installation.html).

If Tesseract is missing, times out, disagrees across frames or has insufficient confidence, the app keeps the rank unknown and displays a warning when appropriate. Installing the Python requirements alone does not install the native Tesseract binary. This fallback does not establish general OCR accuracy for all ranks/fonts.

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

`--replay` can be repeated for multiple folders. A folder may contain one or several consecutive rounds; each folder is analyzed independently. The app never deletes input screenshots. Boundary suggestions do not establish round completeness. Registering separate folders does not combine their totals.

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

With native Tesseract installed, reproduce the diagnostic on existing labelled repository crops:

```sh
python audit_rank_ocr.py --output validation/ocr_crop_audit.json
```

The recorded audit evaluated 128 crops: 57 were eligible red glyphs and only 9 met the OCR confidence gate, all correctly labelled. Coverage is low; these correlated examples do not establish general OCR or complete-round accuracy. Runtime also requires agreement across distinct frames.

See [VALIDATION.md](VALIDATION.md) for measured results, failures, dataset separation and limits; [TRACKING.md](TRACKING.md) for the tracking experiment. The current app uses `models/rank_classifier_v5.pt`. Earlier checkpoints remain available. The V5 training entry point is `train_rank_classifier_v5.py`; it requires explicitly audited local crops and named development groups. Private new screenshots/crops are not distributed in Git.

The external 52-class pretrained detector was also tested unchanged on the digital-table recording and source-domain controls. It performed poorly on the screenshots and was not adopted. See [EXTERNAL_MODEL_EVALUATION.md](EXTERNAL_MODEL_EVALUATION.md) for the protocol, results and reproducible benchmark.

`evaluate_card_result.py result.json physical_reference.json --require-exact` compares explicit physical-card annotations with exported per-frame boxes. It measures mixed identities, extra fragments, unknown/wrong ranks, round assignments and count separately. Its JSON reference schema is documented in the script. Matching totals alone do not pass. Annotation scope must be stated; detector-observation annotations do not measure missed cards outside those detections.

The YOLO localization/dedupe implementation remains unchanged. Historical training scripts and their data stay available. The web app no longer calls the inaccurate legacy round recognizer.

The tracker also repairs short deal-animation fragments: it compares glyphs after normalizing the dimmed card background and permits limited settling movement. It requires strong shape agreement, rejects competing matches, and keeps co-visible cards, separate regions and table-clear intervals separate. Predicted rank alone never triggers a merge. This reduces duplicate rows across several development recordings; remaining fragments still require review.
