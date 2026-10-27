# First-deliverable validation

## Outcome

The app now implements recording, saved-frame replay, the frozen V2 detector and tile dedupe, rank classification, temporal association, selected-region counting, review with source screenshots, explicit duplicate exclusion, ordered corrections, Hi-Lo calculation and JSON export.

**This is a review-assisted prototype. Fully automatic rank recognition and physical-card reconciliation have not reached reliable first-deliverable accuracy.** A saved review is a human-confirmed result, not an automatic-accuracy claim. Actual desktop capture on the user's Mac still needs a live test.

## Multi-round recording analysis

The pipeline now proposes round divisions after four consecutive detection-free frames with a visible empty green table in all selected regions. Upright compact light/red components veto clearing when cards remain visible despite rank-detector failure. This is a heuristic for the supplied layout, not general table/scene recognition. Short clear intervals and long unverified gaps followed by cards are flagged for review. One hand disappearing does not create a new round while another selected hand still has detections. Leading/trailing empty frames do not create empty rounds. Round assignments prevent fragment repair from merging cards across rounds.

Each event has an editable `round_number`; the JSON includes round card lists, individual changes, cumulative changes and cumulative unknown-card ranges. An unknown in any preceding round keeps the subsequent running count unavailable, even if a later round is fully recognized. The total starts at zero for each recording and does not reset at a round boundary. Separate recordings and shoes are not automatically reconciled. Counts still update only after stopping; no live running count or shoe detection is implemented.

The first 180 frames of the original upload form an uninterrupted two-round replay. The proposal identifies the clearing interval at frame indices 70–108 and the next first card at 109, following the previous last observed cards at 69. It returns six physical-card candidates in round 1 and five in round 2, preserving both distinct same-rank 6 cards in round 2. Round 1 is **8, 10, J, 4, A, 8 → −2**. Round 2 remains **J, 6, unknown, unknown, 6**; its change and the cumulative count remain incomplete. Correcting the known ranks to **J, 6, 9, 8, 6 → +1** yields the human-reviewed running change **−1**. This does not claim improved recognition of the unresolved ranks.

The 84-frame full round, 54-frame partial round and latest 48-frame round each remain one round with unchanged card lists/counts (+1, +2 and 0). These four replays processed 366 frames, reusing cached outputs of the frozen detector while rerunning classification, tracking, visual boundary checks and aggregation. A separate scan of all 300 original frames found three suggested transitions, including the beginning of a fourth, partial round at frame 296. These are development recordings, not fresh independent acceptance data. No raw frames or derived crops are included in this update.

All **63 tests pass**. Added checks cover repeated same-rank cards at the same position across rounds, detector dropout with visible cards, white/gray offscreen windows, card backs, short/invalid clear intervals, partial endings, cumulative unknowns, round reassignment, invalid review input, persistence and the explicit boundary-confirmation gate. A multi-round or uncertain result cannot be marked reviewed or export confirmed training crops until the user confirms its divisions as well as its card list. Original single-round regression tests still pass.

Chromium replayed the uninterrupted two-round sequence through the Flask app using cached detector outputs. Browser checks covered cumulative unknowns, before/clear/after boundary screenshots, correcting ranks, withholding reviewed status without division confirmation, changing a card's round while preserving the overall total, persisted corrected round totals, per-round crop labels and adding an excluded missed-card entry. The 390px mobile page had no horizontal overflow or JavaScript errors.

Automatic segmentation can still fail on other felt colours/layouts, false detections, hidden cards, rapid clearing, or transitions without a visible cleared table. Users can correct each card's assignment; the recorded `boundaries` are original automatic suggestions, while reviewed event assignments are authoritative. Review does not make an interrupted recording complete. A fresh uninterrupted recording is still needed to evaluate generalization.

## Deal-animation tracking repair

The latest recording reproduced six candidate tracks for four physical cards. A dim initial 2 and a dim initial 10 each moved more than three rank-box widths before settling. Both the absolute-colour appearance gate and the movement gate rejected their continuations. A correct zero subtotal hid the incorrect card list because the two extra tracks were unknown; assigning them ranks would also cancel their count contributions.

The repair normalizes each crop against its per-channel background, removes boundary-connected borders, and compares the complete interior glyph while preserving aspect ratio and multiple digits. It applies only to short, non-overlapping birth fragments followed within two frames by a substantially brighter view. It permits movement up to four rank-box units only with at least 0.96 glyph similarity. Co-visible cards, different regions, competing near-equal partners and table-clear scene boundaries remain separate. No classifier rank is used as an identity key; original observations and merged track IDs remain available in the result. The detector, CNN, OCR and classifier inputs are unchanged.

All **52 tests pass**, including synthetic red/black fade-in and movement at three opacity levels, two-digit preservation, every pair of different synthetic ranks, same-rank co-visibility, ambiguous destinations, time/distance/region boundaries, and retirement after table clearing. Original recorded identity-transition regressions still pass. Synthetic glyphs do not establish accuracy across arbitrary fonts or animations.

Six development-recording replays processed 431 frames. Previously computed detector boxes were reused for older recordings; the latest recording was also checked with the actual detector through Flask and Chromium.

| Recording | Candidate rows before → after | Result after repair |
|---|---:|---|
| Latest 48-frame recording | 6 → 4 | 2, A, 3, 10; provisional 0 |
| Earlier 84-frame recording | 5 → 5 | K, 7, 3, 8, 2; provisional +1 |
| Earlier 54-frame partial recording | 3 → 3 | 3, 2, 7; provisional +2 |
| Original round 1, 80 frames | 7 → 6 | 8, 10, J, 4, A, 8; provisional −2 |
| Original round 2, 75 frames | 6 → 5 | Two unresolved ranks; count incomplete |
| Original round 3, 90 frames | 8 → 7 | One extra fragment remains; count incomplete |

The latest browser replay checked the four-card list, all 111 preserved observations, merged provenance, provisional status, original-frame evidence, saved review and crop download. No JavaScript errors or horizontal overflow at 390px were observed.

These recordings have informed development and are not an independent acceptance set. In the remaining original-round-3 fragment, extracted glyph aspect ratios disagree; the repair declines to force a match. Brief occlusions, cropped glyphs, longer animations and same-rank replacements without observable separation remain limitations. Raw frames and derived crops from the latest upload are not committed; the new tests use synthetic images.

## Optional local OCR fallback

### Latest scope and export checks

The suite now passes **43 tests**, including unknown-count intervals, content-stable recording grouping, confirmed-review export gating, exclusion and crop deduplication, missing images, and path/symlink containment. Export labels come only from an explicitly saved and confirmed review; editing or saving an unconfirmed review removes export availability. Manually added cards without images are not labelled crops. Model weights and tracking thresholds are unchanged by this update.

The UI explicitly scopes counts to captured cards. Unknowns produce a known-card subtotal and a range for the detected events only. Neither a narrow range nor a confirmed review proves that a recording contains a complete round or that all physical cards were detected.

A real Chromium replay of the supplied 54-frame partial recording returned three events, 3–2–7, with provisional +2. The browser confirmed and saved the review, downloaded the labelled crop ZIP, and checked its labels and grouping metadata. Changing 3 to unknown hid the export link and displayed subtotal +1 with a detected-event range of 0 to +2. No JavaScript errors or horizontal overflow at 390px were observed. This recording is development data, not a fresh acceptance set.

`audit_rank_ocr.py` evaluated all 128 valid labelled crops in the two existing public crop datasets with native Tesseract 5.5.0. Of 57 red-eligible crops, 9 met the 90% single-view gate: 9 correct, 0 incorrect. The per-crop results are in `validation/ocr_crop_audit.json`. This small, correlated and class-imbalanced sample has low coverage and is a development diagnostic, not independent acceptance accuracy. Runtime adds a separate multi-frame agreement gate.

The CNN stays V4. Additional crop-jitter and glyph-CNN candidates were investigated but rejected because they regressed on earlier checks or failed to confidently resolve the remaining case. None replaces the published CNN.

A pretrained Tesseract reader now offers a narrow cross-check for uncertain red ranks. It isolates red ink, preserves both digits of 10, and requires at least two distinct first/middle/last frames with the same valid rank and at least 90% OCR confidence each. Conflicting high-confidence readings, unsupported text (including a clipped standalone 1), missing binaries and timeouts leave the rank unknown. Confident CNN ranks are not overridden. OCR scores and provenance are stored separately from the original CNN suggestion. The fallback is local and uses no recording-specific training.

There are now **37 passing tests** in the environment with Tesseract installed. These include native OCR of a synthetic red glyph, TSV parsing, confidence/conflict/distinct-frame gates, component preservation for 10, missing/timeout behavior, CNN preservation and OCR-to-count integration. The native OCR test is explicitly skipped if Tesseract is not installed; it is not a required dependency for CNN-only processing. Original model, count and identity tests still pass. Frontend syntax and mobile display checks are also covered by the browser validation workflow.

Tesseract is optional and must be installed separately on the Mac (see README). The new recording used for debugging this fallback was not added to the CNN training set. Its raw frames, crops and detailed derived fixtures are not included in this change. Validation on a development recording does not establish accuracy on future recordings. Black ranks and high-confidence CNN mistakes are outside this fallback's scope, and review remains required.

## Oct-26 recording follow-up (current V4 default)

The supplied round_20261026_103552_7d8b395b archive contained 84 frames. The original app reproduced the reported nine rows exactly: duplicated K/7/2 tracks plus an unresolved 3 suggested as Q. Visual review of the selected large-hand regions established five physical cards in first-observed order: **K, 7, 3, 8, 2**, with Hi-Lo **+1**. This scope excludes the ambiguous central miniature view.

Tracking was too sensitive to card-border/background changes in its padded descriptor. The revision relaxes that view only when central-shape similarity is at least 0.90 and contextual similarity remains at least 0.68. It does not merge by rank. The original different-card transitions still pass, including K giving way to a newly dealt 3 at the old position. The new 246-observation crop regression produces five identity-pure tracks instead of nine fragments.

The original rank-3 training set had only two noisy examples with neighboring fragments. V4 uses exactly V3's architecture, preprocessing, seed, 60-epoch budget and split, plus 15 visually checked crops: three views each of the new round's K, 7, 3, 8, 2. Checkpoint selection uses the same 17 historical validation observations, not this recording's evaluation score. Current V4 was selected at epoch 23. No supplied model is overwritten.

| Model | Earlier reserved crop observations | Oct-26 development observations |
|---|---:|---:|
| V3 | 24/29 | 175/246 |
| V4 | 29/29 | 245/246 |

The new recording now belongs to development/training data. Its 246 observations are repeated views of just five physical cards, with 15 used for training. The sole V4 crop error is a transient 3→5 at frame index 10; aggregation over the card's track resolves 3 at approximately 94% mean model score. The recorded full CLI run using the unchanged detector/dedupe processes all 84 original frames and returns **K, 7, 3, 8, 2**, count **+1**, `count_verified: false`, without edits.

There are now **27 passing tests**, including the new identity and real-checkpoint count regressions. The earlier full-round replays still produce unknowns/extra fragments: 29/29 crop accuracy does not establish perfect full-round performance. In particular, border/padding changes can alter rank predictions, and the earlier round-2 main 9 is still unresolved in the runtime crop stream. Fresh recordings excluded from all training and tuning remain necessary for independent acceptance.

Reproduce with `python train_rank_classifier_v4.py`. Full per-crop predictions and the training split are stored in `validation/v4_evaluation.json` and `validation/v4_training_split.json`. The raw captures are retained outside Git; small cropped regression fixtures are under `tests/fixtures/oct26_round*`.

## Earlier V3 classifier experiment

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
4. The two-fps recorder can miss a card that appears and disappears between screenshots. Exact sub-frame dealing order and completeness of interrupted recordings are not inferred. Validate the conservative round-boundary heuristic on fresh recordings; general scene handling, shoe changes, continuous live counts and betting signals remain unimplemented.
