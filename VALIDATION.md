# First-deliverable validation

## Outcome

The app now implements recording, saved-frame replay, the frozen V2 detector and tile dedupe, rank classification, temporal association, selected-region counting, review with source screenshots, explicit duplicate exclusion, ordered corrections, Hi-Lo calculation and JSON export.

**This is a review-assisted prototype. Fully automatic rank recognition and physical-card reconciliation have not reached reliable first-deliverable accuracy.** A saved review is a human-confirmed result, not an automatic-accuracy claim. Actual desktop capture on the user's Mac still needs a live test.

## Classifier experiment

The supplied V2 classifier reproduced its existing 15/22 (68.2%) validation result. The old split is not an independent measure: 13 source frames cross its train/validation boundary and four ranks have no validation examples.

V3 uses the same CNN architecture and exact runtime preprocessing, fresh initialization, seed 42, 60 epochs, weighted loss and the existing mild augmentation. No detector/dedupe settings were changed. Two audited crops labeled 10 but containing only 1 were excluded. Existing labels were reused; additional rank labels came from the already-reviewed crop fixtures, not a request to relabel the whole dataset.

The controlled split contains 154 training crops, 17 checkpoint-selection crops and 29 reserved round-2 crops. Both original and reviewed crops with source-frame indices 90–160 are excluded from optimization and checkpoint selection. Known uploaded rounds 1 and 3 stay grouped. Remaining historical examples use global source-frame groups; their unknown round/physical-card relationships remain a limitation of checkpoint-selection validation.

| Model | Correct reserved round-2 crop observations |
|---|---:|
| Supplied V2 | 17/29 (58.6%) |
| V3 candidate | 24/29 (82.8%) |

The reserved observations contain repeated views of a few physical cards; they are not 29 independent cards. The original V2 checkpoint already trained on some examples from this round. V3 starts fresh and excludes the round. Classifier suggestions are not calibrated probabilities of correctness. The five remaining V3 crop errors are stored in the report and include 9→8 and miniature J→4. All 13 ranks are not represented in this reserved round, so these numbers cannot establish general-purpose accuracy. Later exploratory reference-matching checks also inspected this development round; it must not be presented as a fresh final acceptance set for future changes.

Reproduce the training with `python train_rank_classifier_v3.py`. The script writes ignored `rank_classifier_v3/` artifacts, not the published models. `validation/v3_training_split.json` and per-crop prediction JSON files record the exact experiment. Runtime library versions are in `validation/runtime.json`.

## Tracking

The earlier 60-sampled-frame replay had eight tracks mixing reviewed physical identities. The appearance matcher separated those transitions in the 74 reviewed snapshots, while increasing total tracks from 26 to 34. These are development regressions, not independent tracking accuracy.

The app processes every frame, uses only selected disjoint source regions, keeps short tracks for review and attempts conservative same-region fragment repair. It never joins two tracks observed in the same frame, and does not deduplicate solely by predicted rank. There are still duplicates and uncertain short fragments in all three dense round replays. They are retained in the audit rather than silently discarded. Small duplicate UI views are excluded by choosing source regions; general cross-view reconciliation is not solved.

The central miniature hand sometimes shows 10–6–J while the larger right hand shows 10–6. The relationship between those views requires clarification; default canonical-region counts do not include an unmatched card from the miniature view. Therefore no full-table automatic count is claimed for that recording.

## Functional checks

- 24 unit/regression tests pass: recorded identity transitions, same-rank separate cards, frame dropout, ambiguous matches, capture-thread startup and monitor errors, async start/stop, busy-state races, stale review rejection, input validation, region exclusion, review correction and count calculation, model class order, and preprocessing compatibility on all 179 fixture crops.
- A real local HTTP server analyzed all 80 frames of uploaded round 1 using the unmodified detector/dedupe and the published V3 classifier. The unreviewed result had 9 candidate events, including unknowns/duplicates; it was correctly marked incomplete and unverified.
- A real Chromium browser opened the page, inspected original frame evidence, edited ranks, excluded duplicate detections, confirmed and saved the review, and downloaded the result. The reviewed canonical card list was **8, 10, J, 4, A, 8**, with Hi-Lo **−2**.
- The browser reported no JavaScript errors, and the 390px-wide mobile layout had no horizontal overflow.
- Dense recorded-frame processing was also executed on 75 frames of round 2 and 90 frames of round 3. These did not produce reliable complete automatic card lists; they remain review-assisted. Round 2's visible canonical physical cards are J, 6, 9, 8, 6 (Hi-Lo +1). Round 3's canonical cards are 10, 7, 6, 7, 2, 5 (Hi-Lo +2), before resolving the miniature-hand ambiguity.
- Dependency consistency and JavaScript syntax checks passed. No original captures were deleted or committed. Model hashes are in `models/README.md`.

## Remaining acceptance work

1. Clarify the small-hand/main-hand relationship and resolve unmatched cards across views. Do not assume equal ranks mean one physical card.
2. Improve the remaining rank errors and fragment duplicates; then measure complete automatic card lists and counts on a genuinely new recording excluded from all training and tuning.
3. On the Mac, install dependencies, check the hand-region preview and Screen Recording permissions, then exercise the three-second countdown → record → stop → analyze flow on a live round.
4. The two-fps recorder can miss a card that appears and disappears between screenshots. Exact sub-frame dealing order, completeness of an interrupted recording, automatic round boundaries, shoe changes, continuous counts and betting signals are not implemented.
