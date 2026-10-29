# First-deliverable validation

## Shared-hand continuation after the main display disappears

The supplied `split_round` recording exposes a source-continuity failure in baseline `2dae973`. The large player J spades/Q spades hand disappears after frame 42, while the central representation persists and adds a separate J spades and 6 spades. Dealer cards are 4 clubs, Q diamonds, 7 diamonds. The seven unique visible cards have Hi-Lo change −2. The action producing the additional shared cards is not verified as a split: these frames do not show separately displayed split hands.

The baseline already links the original two cards and recognizes both additional ranks, but its simultaneous-primary visibility rule forces those new identities to Unknown. It returns seven candidates, two unknowns, and a detection-only range −4..0. A separate geometry failure rejects an intact four-card hand because an antialiased outer row fails the two-row paper-border test.

The update preserves an established shared prefix through local source observations. Continuation needs at least two joint source frames showing the original prefix and its linked primary cards. Every overlapping ordinal rank strip must retain compatible shape and colour, with bounded position/height changes and no more than two missing sampled frames between accepted observations. Blank or missing strip evidence cannot establish continuity. At least two observations of a new fully exposed slot must descend from the established prefix before this alternative can accept its identity. A repeated rank/suit already in the linked prefix can still denote a distinct added card. Unlinked primary competitors, long gaps, prefix replacement, and shrinking hands still block automatic acceptance. This is a layout-specific continuity check, not a universal identity guarantee or recognition of split actions. The existing simultaneous-view path remains available.

The geometry check now looks for a qualifying adjacent pair of paper rows within each three-row edge band, retaining the existing brightness, shape, context and seam checks. It tolerates a single antialiased fringe row rather than lowering the paper threshold. The 6 is first retained at source frame 96 instead of 112.

The updated replay returns exactly seven rank/suit/view identities, zero unknowns, and change −2 with no coverage issues. Both added cards use `identity_basis: continuous_shared_hand`; the original J and Q link once and are not counted again. The result remains provisional until review. No detector, classifier, suit-reader or reference weights were changed or retrained. All twelve earlier replay summaries—candidate count, unknown count, and count change—remain unchanged. Primary identity/rank/round/observation-box provenance is unchanged in all thirteen jobs, covering 1,197 frames. Five older recordings still require review for their previously documented failures.

All 158 tests pass, including new cases for a hidden main display, repeated identical cards, prefix replacement/colour changes, large movements, long gaps, missing/blank patches, single-frame noise, brief new cards, and dealer conflicts. Real Flask/Chromium analysis reproduces the seven-card result and checks evidence screenshots, review and crop export, continuation captions, and a 390px mobile layout without JavaScript errors or horizontal overflow. Aggregate results are in `validation/shared_continuation_evaluation.json`; uploaded frames, crops and detailed annotations remain private. This recording informed implementation and is now development data. Fresh untouched recordings are needed for acceptance; arbitrary split-hand rearrangements and general multi-hand tracking remain unsupported.


## Outcome

The app now implements recording, saved-frame replay, the frozen V2 detector and tile dedupe, rank classification, temporal association, selected-region counting, review with source screenshots, explicit duplicate exclusion, ordered corrections, Hi-Lo calculation and JSON export.

**This is a review-assisted prototype. Fully automatic rank recognition and physical-card reconciliation have not reached reliable first-deliverable accuracy.** A saved review is a human-confirmed result, not an automatic-accuracy claim. Actual desktop capture on the user's Mac still needs a live test.

## Configured shared-view reconciliation

This update implements a separate central-hand monitor, suit recognition and compressed-rank recognition without changing primary V5, the V2 detector or its existing primary tracking. It addresses r5's additional 9 hearts rather than declaring a correct main-view subtotal to be complete card coverage. **R4/r5 and all older inspected captures are now development data for these auxiliary readers.** Their historical held-out results below still apply to the frozen prior main-view candidate only. No independent acceptance data remains for this update.

The central geometry reader looks for the flat compressed white hand, a suitable green/olive surround and visible internal seams. It follows ordinal append states within the primary round assignments and retains briefly seen or already-overlapped slots for review. Clipped/unsupported hands and a shared view lost for three primary-visible frames produce source-linked coverage issues, rather than preserving an earlier complete status. A separate 13-rank CNN reads fully exposed last-card corners. A small 4-suit CNN reads both printed suit corners; compressed suits are rectified vertically. Bright-frame, ink-colour, shape/aspect and temporal agreement gates reject weak or contradictory evidence. A moderate diamond score additionally needs vertical reflection symmetry. The shared-rank reader rejects blank crops and strong contradictory predictions.

Automatic linkage requires at least two matching ordered rank/suit anchors, at least two co-visible source frames for each, and a coherent first-observation lag bounded to eight samples with at most two samples of variation around its median. These are layout-specific gates, not a general identity theorem. Each primary event can satisfy only one shared slot. An additional shared slot must be supported while the prefix remains co-visible, with no plausible unlinked primary counterpart in **any** main hand. Matching rank/suit alone never collapses physical identities: a repeated tuple in another visible slot can be a distinct card, while a possible match in an unlinked dealer hand remains ambiguous.

On the developed r5 replay, primary 9 clubs/4 spades link once to the shared prefix; the additional 9 hearts becomes a fifth card with its own source evidence. All five rank/suit/view identities match the visual reference and the count change is 0. R4 remains five cards, with its three central slots linked and no extra rows. R2/r3 retain exact main lists with their shared prefixes linked. An unavailable shared view, as in r1, now leaves the resolved primary subtotal visible but withholds `count_change` pending coverage review rather than declaring the recording complete.

The UI adds suit correction, a central calibration box, a shared-slot/source-image coverage panel and a coverage-confirmation gate. Ambiguous shared identities remain Unknown candidates with source evidence and possible counterpart rows. Users can keep them as distinct after correcting ranks or exclude duplicate/false candidates, then explicitly confirm coverage. Changes to a linked rank/suit/round/exclusion invalidate the previous automatic coverage approval. Unknown ranks still block verification even after coverage is confirmed. An unresolved earlier round's coverage blocks later running totals. `captured_count_change` is explicitly separate from a complete `count_change`; numeric ranges cover included detections only, not unseen cards. Counts remain provisional until review. Context images of already-overlapped hands cannot be exported as isolated rank-training crops; reviewed exports include suit labels.

`validation/crossview_evaluation.json` records 12 replay jobs and 1,067 frames using cached unchanged detector outputs while rerunning recognition, association and shared-view analysis. Every primary event's identity, rank, round and frame/box provenance remains unchanged. Seven jobs automatically reconcile the configured views; five still require review. The latter include an unavailable shared view, pre-existing primary fragmentation, weak tiny-card evidence and a shared extra that could match an unlinked dealer card. These unresolved cases are deliberately not forced into a complete automatic count. Other seats, hidden cards, split/changed hands, different layouts and shoe changes remain outside automatic coverage.

Auxiliary development training uses 43 unique miniature rank crops from 25 observed slot groups, covering 12 rank labels (no real miniature A examples), plus historical/main crops and simulated downsampling. The mini reader is initialized from V5 but saved separately; the primary checkpoint is unchanged. Suit training uses 229 unique crops and 27 selection crops from four separate physical-card groups, with compressed training icons sampled at weight five. The suit selection diagnostic is 26/27 and the mini training diagnostic is 43/43. These correlated development diagnostics are not general accuracy estimates. Full counts, selection details and model hashes are in `validation/crossview_training_audit.json`. Original frames, rank/suit crops, detailed annotations and private training CSVs remain outside Git.

All 149 tests pass, including cross-view prefix linking, repeated identical rank/suit cards, dealer conflicts, view delays, missing/brief/overlapped observations, shared resets, per-round separation, incomplete coverage, invalidated review links, suit colour/geometry conflicts, invalid/missing models, blank mini ranks, training-input checks and calibration persistence. Real Flask/Chromium replays check r5's added 9 hearts, r4's linked-only slots, r1's zero-unknown **Check coverage** state, suit edits, source screenshots, review/coverage gates, suit-labelled crop exports and mobile layout. No JavaScript errors or horizontal overflow at 390px were observed. Fresh untouched recordings, preferably including a non-neutral additional shared card, are needed for independent acceptance.

## Second frozen held-out result: r5, and cross-view coverage gap

The same frozen candidate revision `04d8a6f` was evaluated on r5 without tuning, retraining or production-code changes after either held-out result. All 74 source frames were inspected and the visual reference saved and hashed before predictions. The selected main-view cards are player 9 clubs, 4 spades and dealer 7 spades, Q spades, count change 0.

| Measure | Previous pipeline (`82ce435`) | Frozen candidate (`04d8a6f`) |
|---|---:|---:|
| Candidate rows / expected main-view cards | 5 / 4 | 4 / 4 |
| Unknown rows | 1 | 0 |
| Exact main-view card list | Fail | Pass |
| Complete main-view count change | Unavailable | 0 |
| Detection range | -1 to +1 | 0 to 0 |

The previous pipeline retains a narrow partial-4 candidate alongside the complete 4. Face reconciliation suppresses that extra output while retaining all 185 positive annotated observations. The candidate has no mixed identities, missing annotated main-view observations/cards, extra fragments/events, wrong accepted ranks or round errors. Every distinct retained crop was inspected. Aggregate evidence is in `validation/r5_heldout_evaluation.json`; detailed labels, crops and screenshots remain outside Git.

**Main-view acceptance is not complete exposed-card coverage.** Beginning at frame 51, the central miniature hand shows an additional 9 hearts, while the large player hand remains 9 clubs and 4 spades. The differing suit establishes that the red 9 is a distinct additional card; default regions exclude it. There are therefore five distinct visible digital cards but only four in the main-view output. The additional 9 has Hi-Lo value 0, so the correct displayed count cannot reveal this missing-card scope issue. Both pipelines exclude that view; the main-view pass does not claim generic cross-view reconciliation or whole-shoe completeness.

Across r4/r5, the frozen candidate passes both main-view card-list checks (nine main-view cards, 339 positive observations, 141 screenshots) with zero unknowns; the previous pipeline fails both, with four unknown rows total. This is a positive independent check of this layout, not a near-perfect general-accuracy estimate. The two rounds were never used for training or tuning. Future changes informed by their errors or the coverage gap must classify them as development data and use new independent test footage.

## First frozen held-out result: r4

Candidate revision `04d8a6f` and its V5 checkpoint were frozen before r4 was opened. All 67 source screenshots were visually audited and the expected main-view card list saved and hashed before either pipeline ran. The expected cards are player Q,3,5 and dealer 8,10, count change 0. Both pipelines use identical source regions and cached outputs of the unchanged V2 detector. No code, weights, thresholds or training data were changed using this recording.

| Measure | Previous pipeline (`82ce435`) | Frozen candidate (`04d8a6f`) |
|---|---:|---:|
| Candidate rows / expected physical cards | 7 / 5 | 5 / 5 |
| Unknown rows | 3 | 0 |
| Exact main-view card list | Fail | Pass |
| Complete count change | Unavailable | 0 |
| Detection range | -4 to +2 | 0 to 0 |

The previous pipeline leaves the genuine 5 unknown, retains an extra narrow partial candidate and splits the animated dealer 10. V5 resolves 5; face reconciliation suppresses the partial with a retained counterpart and rejoins the dim 10. The candidate retains all 154 positive annotated observations with no mixed identities, missing annotated observations/cards, extra identity fragments/events, wrong accepted ranks or round errors. Every distinct retained crop was inspected to verify its track's identity; the extra partial output is excluded from positive observation truth and preserved in the suppression audit. This is a main-view physical-identity comparison, not generic full-table detector-recall measurement.

`validation/r4_heldout_evaluation.json` records the comparison and the hash of the pre-prediction visual reference. Screenshots, crops and detailed observation labels stay outside Git. **This is one successful previously unseen round, not evidence of near-perfect general accuracy.** R5 was still uninspected when this first report was saved; its result is recorded above. The app's result is still provisional/unreviewed despite passing the independent comparison. The frozen protocol below is unchanged; if this round later informs tuning, it must be reclassified as development data.

## Card-face evidence and V5 development candidate

The next candidate was developed using three new recordings (r1-r3), after reserving r4/r5 before inspecting any of their frames. The baseline is revision `82ce435`. All 206 new source frames were inspected, including the J visible for only two frames in r1. Ground truth in the selected main views is r1 player A,5,9,J / dealer 7,9,5 (0); r2 player 8,Q / dealer 6,3,5,A,6 (+2); r3 player 9,K / dealer 8,10 (-2). Same-rank cards in different positions/hands remain distinct.

| New development recording | Baseline rows / unknowns | Updated rows / unknowns | Correct cards / change |
|---|---:|---:|---|
| r1 | 13 / 9 | 7 / 0 | 7 / 0 |
| r2 | 11 / 5 | 7 / 0 | 7 / +2 |
| r3 | 5 / 1 | 4 / 0 | 4 / -2 |

All three exact main-view card lists now pass their development identity audit: no missing physical cards, mixed identities, extra fragments/events, wrong accepted ranks or wrong round assignments in the annotated scope. The ranges for these detections narrow from -10..+8, -4..+6 and -3..-1 to 0..0, +2..+2 and -2..-2. These are arithmetic detection ranges, not confidence intervals or guarantees against unseen missed cards. Results stay provisional until reviewed.

The changes address separate stages:

- Visible rectangular paper and internal seams establish optional face geometry for this upright digital layout. Suppression requires a retained counterpart and strong containment/partial-box evidence. Disjoint narrow suit-edge slivers additionally require agreement of both printed rank corners. Two full candidates are not deleted just because a single face was inferred. Unsupported/ambiguous geometry retains candidates. Original detector inference/weights and its tile/NMS thresholds are unchanged.
- A fully exposed opposite printed rank can support rejoining a short dim birth, with colour/aspect/shape, fade, height, distance, round/scene, non-co-visibility and unambiguous-partner gates. It is not a classifier rank identity key. A partly overlapped face exposes a suit instead of that opposite rank, so it cannot supply this cue.
- A stable track can project its previous complete rank box relative to a verified face into the current image. At least two preceding observations and >=0.94 glyph agreement are required; ordinary assignment/motion/ambiguity gates still apply. Only a selected continuation changes its classifier crop. Original detection boxes, projected-crop flags and inferred faces stay in the JSON. Suppressed detections have frame/region, original box, retained counterpart and reason in `suppressed_detections`.
- V5 uses the same CNN architecture/preprocessing, seed 42 and 60-epoch training budget, starting fresh on the 169 existing V4 training crops plus 61 visually audited crops from the 18 new development cards. The fixed historical 17-crop selection set selects epoch 57 by accuracy then log loss. New recording groups are training-only. No r4/r5 frames or labels were loaded. Historical selection omits six ranks; new training fit is not independent accuracy. The V1 reference bank and rank-acceptance/unknown thresholds remain unchanged.

With face evidence and the old V4 weights alone, all 18 new physical cards are separated, but four genuine ranks remain unknown. V5 resolves those four. Thus the result separates identity improvement from classifier training rather than crediting the classifier for removing duplicates. The candidate's 18 new ranks are accepted directly by CNN consensus, without requiring optional Tesseract.

The physical-identity comparison covers 530 retained detector observations. Thirteen explicitly identified extra partial rank/suit outputs are excluded from positive observation annotations and logged by the suppression audit. The baseline comparison counts those retained false candidates as unmatched outputs; visual physical-card truth independently establishes 18 cards. This is not a test of generic detection recall beyond the selected views. Source frames, crops, per-observation annotations and caches remain outside Git. Aggregate results are in `validation/face_v5_evaluation.json`; training counts, selection history and hashes are in `validation/v5_training_audit.json`.

Ten replay jobs cover 926 frames, reusing frozen detector boxes and rerunning subsequent stages. The earlier 335-observation two-round audit remains exact at 12 cards/+1. The original two-round recording now resolves its main 9 and returns 11 cards/-1. Earlier full, partial and single rounds retain their correct lists/counts (+1, +2 and 0). **Two older jobs still fail exact acceptance:** the original third round retains one brief duplicate-10 unknown, and the October 29 Q recording retains its dim Q fragment as an unknown. Their opposite-corner evidence does not meet the unchanged similarity guard, so neither is silently discarded. No new accepted-rank errors were observed in these audited comparisons. All inspected recordings are development regressions.

All 114 tests pass. New synthetic checks cover contained partials, absent counterparts, identical co-visible cards, visible overlap seams, multiple full candidates, small/no-face inputs, all distinct rank pairs in opposite-corner matching, fade/co-visibility restrictions, projected-corner recovery/contradiction, suppression provenance and the baseline switch. Training inputs reject unapproved groups, invalid checksums, path escapes, unreviewed sources and inconsistent card labels. Tests verify mechanics and regressions, not target accuracy on fresh rounds. Chromium additionally replays r1-r3 through the real Flask app with cached detector outputs and the published default V5 checkpoint, checking exact card rows/counts, original screenshots, provisional/reviewed status, reviewed-crop export gating and ZIP validity. A neutral rank changed to unknown still invalidates completeness. The 390px layout has no horizontal overflow or JavaScript errors.

`validation/r4_r5_protocol.json` freezes the unseen comparison. Use `--classifier models/rank_classifier_v4.pt --no-face-evidence` for the previous pipeline and V5/default face evidence for the candidate, with identical source regions. The baseline switch disables suppression, secondary-corner repair and projected-crop sampling together. Freeze code/weights, annotate every held-out frame before running predictions, then report both results without tuning on them. Two held-out rounds will be a limited independent check, not proof of near-perfect general accuracy.

## Sliding-card association and broader temporal reference sampling

The October 29 recording has 87 screenshots and six distinct main-view cards: player 3, 7, 4, J; dealer Q, K; Hi-Lo change -1. Revision `5e805e7` produces eight candidate events with four unknowns: the animated Q is split from its settled view, and the sliding J is also split. The genuine 3 remains unknown. The known-card subtotal is coincidentally -1; the detection range is -5 to +3, and the complete count is unavailable.

The matcher previously rejected a candidate beyond its raw movement limit before checking hand-motion compensation. It now permits one additional normalized box unit only when at least two coherent established hand anchors support motion, frames are consecutive, glyph similarity is at least 0.94 and compensated distance satisfies the original limit. Ordered alignment and ambiguity/appearance checks remain in force; no predicted rank establishes identity. This joins the J's two observations without a general relaxation of fragment-repair thresholds.

Reference consensus now checks up to seven spread-out observations instead of first/middle/last alone, retaining those three positions. The 3 has many clear matches that the previous sampling misses. The same bank, similarity/margin thresholds, CNN top-rank agreement, distinct-frame requirement and rejection of conflicting accepted votes remain. More samples can still increase false acceptance on unseen data; temporal agreement between repeated views is not independent validation. No weights, reference examples or classifier preprocessing were changed.

On the new recording, candidate events fall from eight to seven, extra identity fragments from two to one, and unknown events from four to one. All 196 detector observations are retained without mixed identities or wrong accepted ranks in the annotated comparison. The genuine 3 and the single J are resolved. The animated Q remains an extra unknown: the count is still incomplete, with subtotal -1 and detection range -2 to 0. **The exact automatic card list still fails acceptance.** `validation/motion_reference_evaluation.json` records these separate measures. The detailed observation map, original screenshots and crops remain outside Git. This audit covers frozen detector observations, not full detector recall.

Seven replay jobs cover 720 frames, using cached frozen-detector boxes and rerunning association, classification, references and segmentation. Across the six earlier jobs, every event's identity, rank, round and frame/box provenance remains unchanged; previous unresolved cases remain unresolved. The earlier two-round recording again passes its 335-observation physical-identity comparison with 12 cards and +1. These inspected recordings are development regressions, not independent acceptance data. All 93 tests pass, including supported/unsupported motion, different glyphs, excessive/opposing travel, bounded reference sampling and conflicting additional votes. Chromium checks the actual revised result renders seven candidates, one J, reference-resolved 3, one unknown, incomplete status and the -2 to 0 range; reviewed-crop export remains unavailable and the 390px layout has no horizontal overflow or JavaScript errors.

## External pretrained detector evaluation

The unchanged TeogopK synthetic 52-class YOLO checkpoint was tested on all 177 frames of the two-round development recording using full screenshots at 640/1280 and hand crops at 640. At confidence 0.25 the full-screen modes detect no selected-region cards; cropped hands correctly match 38/335 annotated rank observations, all from one physical card, with 79 unmatched outputs. At 0.05, cropped matches include 49 correct and 82 wrong ranks. Three source-domain test controls correctly match all 11 labelled corners, with two extra predictions, supporting that inference is operational. The model was not adopted and production behavior is unchanged. All 88 tests pass. [EXTERNAL_MODEL_EVALUATION.md](EXTERNAL_MODEL_EVALUATION.md) and `validation/external_yolo_benchmark.json` describe source hashes, geometry, annotation scope, manual unmatched analysis and comparison limitations.

## Hand-level association and reference recognition

The new 177-frame two-round recording exposed a mixed 4/K track and an extra A fragment. The frozen baseline is revision `2c3e002`: 13 candidate events, five unknowns and a known-card subtotal of +2; the correct main-view result is 12 physical cards and +1. The uploaded split archives contain two byte-identical directory copies of the 177 frames; only one chronological copy was evaluated.

Production association now models an ordered hand row. Distinctive glyph matches estimate shared motion; sequence alignment permits births/misses while preserving left-to-right identities. When the configured UI appends an identical card on the right, a coherent existing prefix can establish motion. This prevents the existing 4 following a new K near its old position. Normalized glyph evidence also rejoins the jittered A, while preserving compact components at crop edges so a 10 does not lose one digit. The unchanged original association path handles tiny or multiple-row inputs; general split/rotated-hand layouts are not established as supported.

The uncertain-rank reference bank contains signatures of the same 169 previously published V4 training crops. It introduces no new recording labels. Acceptance requires >=0.97 similarity, >=0.04 margin from a different rank, at least two distinct agreeing first/middle/last frames, and agreement with the CNN's top rank. Confident CNN outputs are preserved. OCR remains a separate optional fallback. Rows show **Reference agreement** with an image-match score; these scores are not probabilities. The detector, CNN weights, CNN preprocessing and CNN thresholds are unchanged.

The bank's leave-source-group-out development diagnostic accepts 93/169 crop observations with zero incorrect accepted ranks. Only nine ranks have any accepted observations in that diagnostic; coverage is zero for 3, 5, 6 and 10. Some groups are frame-level rather than complete recording/physical-card groups, and neighbouring groups can be correlated. Consequently this is neither independent validation nor evidence of general 13-rank accuracy. The runtime CNN/reference agreement also shares training data. Full diagnostic predictions are in `validation/reference_v1_audit.json`.

Against human-inspected physical-card identities across all 335 main-view detector observations, the new two-round result has:

| Measure | Frozen baseline | Updated pipeline |
|---|---:|---:|
| Candidate events / expected physical cards | 13 / 12 | 12 / 12 |
| Mixed-identity tracks | 1 | 0 |
| Extra identity fragments | 2 | 0 |
| Unknown events | 5 | 0 |
| Retained observations | 335 | 335 |
| Complete count change | Unavailable | +1 |

The baseline's one mixed track covers both the 4 and K; two physical cards (4 and A) have multiple track fragments. These are different failure measures, so the extra-fragment count is not just candidate events minus expected cards. The revised main-view card lists are:

| Round | Player | Dealer | Change / running change |
|---|---|---|---|
| 1 | 2, Q | K, 4, A, 9 | −1 / −1 |
| 2 | 4, K | 5, 7, 4, 7 | +2 / +1 |

Q, 9 and 5 are resolved by reference agreement. All results remain provisional until a saved human review. The central hand additionally shows 9 in round 1; its physical-card relationship remains unresolved, but a neutral 9 cannot alter this Hi-Lo result. This is a main-view evaluation, not a claim that excluded views are redundant.

`validation/hand_model_evaluation.json` records the baseline comparison and separate physical-identity evaluation. `evaluate_card_result.py` matches explicitly labelled observations to per-frame box provenance and reports mixed identities, extra fragments, unknown/wrong ranks, round assignments and count independently. Its tests reject duplicate contributions that cancel to a correct final total. Annotation coverage is explicit: the 335-observation audit does not measure detector recall outside those observations. Source frames, derived crops, caches and the detailed new observation annotations remain outside Git.

Six replay jobs processed 633 frames, reusing the frozen detector's cached outputs and rerunning association, classification, references and boundaries. The earlier 84-frame round remains five cards/+1, the 54-frame partial capture remains three/+2, and the prior 48-frame round remains four/0. The original first round remains six/−2. Original round 2 now resolves its uncertain 8 but leaves its main 9 unknown because the CNN's top suggestion disagrees with the reference reader. Original round 3 still retains one short duplicate-10 fragment and remains incomplete. No successful replay is treated as a fresh acceptance set.

All **82 tests pass**, covering hand expansion, identical-rank births, shared movement, identity crossing, dropout, crop-edge digits, original reviewed identity transitions, reference provenance/gates and stage-specific evaluation. Chromium replayed the new recording through Flask using cached detector outputs and checked 12 rows, 335 observations, the −1/+2 breakdown, provisional +1, reference scores/provenance, original evidence, review/export gates, neutral-unknown invalidation and mobile layout. No JavaScript errors or horizontal overflow at 390px were observed. Co-visible same-rank cards no longer receive a misleading duplicate warning.

This new recording has now informed algorithm development; it is not an untouched holdout. Fresh recordings, balanced rank coverage, full view semantics and reliable live Mac capture remain necessary before claiming first-deliverable reliability. This update is a measured improvement within the configured layout, not completion of automatic whole-table/shoe tracking.

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
