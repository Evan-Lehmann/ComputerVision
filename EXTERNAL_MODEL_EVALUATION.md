# External playing-card detector experiment

The unchanged 52-class synthetic checkpoint from
[TeogopK/Playing-Cards-Object-Detection](https://github.com/TeogopK/Playing-Cards-Object-Detection)
is not a useful replacement for the current CardVision pipeline on the evaluated
digital-table recording. No production model, threshold or association was changed.
This conclusion concerns the supplied pretrained checkpoint, not its potential
after fine-tuning on appropriate digital card graphics.

## Source and protocol

The source revision inspected was `6da007263c51cf5a44a3c06ea840030ff72ce678`.
The checkpoint is `final_models/yolov8m_synthetic.pt`, 52,085,835 bytes, SHA-256
`0b408029b1241b2489217f89b63f153d81cb0fc6a035361a87bc6e7dac5c62c9`.
It contains all 52 rank/suit classes. The separate tuned model advertises 13
heart-labelled classes and was not treated as an interchangeable full-deck model.

The model was evaluated unchanged on the 177 unique frames of the supplied
two-round development recording. Reference annotations cover 335 detected
main-player/dealer observations representing 12 physical cards. They do not
measure all visible cards or detector recall outside those observations. No
training or fine-tuning used this recording. It has informed earlier CardVision
development, so this comparison is not an independent acceptance test.

Three input modes were tested: whole screenshot at 640 and 1280, and each
configured hand region separately at 640. CPU inference used NMS IoU 0.7. The
primary confidence threshold was 0.25; predictions down to 0.05 were retained
for diagnostic analysis. Predicted corners can include a suit while reference
boxes contain only a rank, so matching requires >=50% reference-box coverage
and >=10% prediction-box coverage. Matching is one-to-one, with geometric ties
broken by confidence, never by the true rank.

## Screenshot results

At confidence 0.25:

| Input | Correct annotated rank observations | Missed annotations | Physical cards represented through annotated corners | Unmatched predictions |
|---|---:|---:|---:|---:|
| Whole screenshot, 640 | 0 / 335 | 335 | 0 / 12 | 0 |
| Whole screenshot, 1280 | 0 / 335 | 335 | 0 / 12 | 0 |
| Separate hand regions, 640 | 38 / 335 | 297 | 1 / 12 | 79 |

The 38 correct cropped observations all belong to the round-1 player 2. Manual
inspection of repeated-location clusters accounts for the remaining 79 cropped
predictions: 60 label the inverted lower corner of the dealer's 5 as 3, and 19
label part of the yellow player-total tag `14` as A. Generally unmatched
predictions can be legitimate additional corners; they are not automatically
false positives without inspection.

At confidence 0.05, cropped hands yield 49 correct and 82 wrong ranks among 131
matched annotations, with 162 additional unmatched outputs. Only two physical
cards ever have a correct annotated match. Increasing input size or lowering
confidence does not make this a complete card recognizer.

For context, the current CardVision pipeline returns the 12 main-view cards,
retains all 335 observations without mixed identities, and produces round
changes -1 and +2, totaling +1 on this development recording. It aggregates ranks
over time; the external results above are frame-level. These are not equivalent
general-purpose accuracy measurements. Detection failure alone is sufficient
to reject this checkpoint as a direct replacement here.

## Operational controls

To check that model loading and decoding work, the first three files listed in
the source repository's synthetic test-image directory were evaluated without
outcome-based selection. All 11 labelled corners were matched at IoU >=0.5 and
assigned the correct rank/suit class at confidence 0.25. There were 13 predictions,
including two extras. This is a small source-domain sanity check, not independent
accuracy evidence. Control class IDs were checked against `data.yaml`.

Additional smoke checks are included in the aggregate report: an author-provided
unlabelled photograph produced five detections, while a cropped, already-overlaid
webcam demo screenshot produced none. No accuracy is assigned to those unlabelled
images. The successful annotated controls support interpreting the digital-table
failure as a transfer problem rather than a completely broken loader; they do
not prove universal physical-card performance.

The source examples use traditional rank/suit corner layouts. CardVision's UI
has tiny raster glyphs and places the rank and suit side by side across the top.
Font, layout and scale differences are plausible contributors, not separately
isolated causal measurements. In-domain fine-tuning or training data designed
for this UI remains necessary to assess this architecture fairly.

## Reproduction and artifacts

`benchmark_external_detector.py` runs the three input modes on a local checkpoint
and annotated recording. It validates all 52 classes, inspects checkpoint globals
and uses PyTorch's restricted loader with a fixed allowlist of installed layers.
The optional benchmark requires PyTorch 2.6+ for inspection; the app's existing
runtime requirement is unchanged. No code from the external repository is run.

```sh
python benchmark_external_detector.py \
  --weights /path/to/yolov8m_synthetic.pt \
  --frames /path/to/recording \
  --reference /path/to/physical_reference.json \
  --baseline-result /path/to/current_result.json \
  --control-dataset /path/to/synthetic_dataset/test \
  --output round_captures/external_benchmark \
  --summary validation/external_yolo_benchmark.json
```

The local control folder contains `images/` and `labels/`, with class metadata
in `../data.yaml`. The reference schema is the same as `evaluate_card_result.py`.
Detailed predictions are written only to the chosen output directory. The
published [aggregate report](validation/external_yolo_benchmark.json) records
metrics, source hashes, runtime versions and limitations. Its manual unmatched
audit is an additional human analysis of the saved predictions, not an automatic
feature of the benchmark script.

The third-party weights, control images, uploaded frames and detailed annotations
are not redistributed. The source README describes its datasets as CC0; this
experiment does not assert the same license for all repository code or weights.
All 88 repository tests pass, including full-deck label validation, corner-box
matching, duplicate-assignment handling, class-map checks and rejection of
unsupported checkpoint globals. Production behavior remains unchanged.
