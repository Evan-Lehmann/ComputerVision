# Model checkpoints

`rank_reference_v1.npz` is the current uncertain-rank reference bank, used
alongside the CNN rather than replacing its weights. It contains normalized glyph
signatures and source keys/groups for exactly the 169 previously published V4
training crops, covering all 13 ranks. No examples from the new two-round
upload were added. Build it with `python build_rank_reference.py`; the audit is
`validation/reference_v1_audit.json`.

Reference matching requires similarity >=0.97, a different-rank margin >=0.04,
agreement in at least two distinct frames sampled from up to seven spread-out
observations, no conflicting accepted votes, and agreement with
the CNN's top rank. It never overrides confident CNN output. The bank shares
training data with the CNN, so agreement is not independent verification.
Similarity scores are not calibrated probabilities. References remain font/
layout dependent, and the CNN's remaining high-confidence errors are not
resolved by this reader.

- `rank_detector_v2.pt`: supplied YOLO V2 localization checkpoint, unchanged.
- `rank_classifier_v2.pt`: supplied original CNN checkpoint, unchanged (recorded validation 68.2%).
- `rank_classifier_v3.pt`: new controlled CNN run with the same architecture and preprocessing, fresh initialization, seed 42. This checkpoint is retained for comparison. It still makes errors and its output requires review.

V3 uses existing labels plus reviewed training snapshots from uploaded rounds 1 and 3. Uploaded round 2 was excluded from both training and checkpoint selection. On its 29 labeled crop observations, V2 got 17 correct and V3 got 24 correct. These are observations of a few repeated physical cards in one round, not 29 independent cards. See `VALIDATION.md` and the per-crop reports in `validation/`.

- `rank_classifier_v4.pt`: retained previous app default. A fresh controlled run with V3's fixed split plus 15 checked Oct-26 crops from five physical cards. It resolves the reported 3→Q error on that recording. This recording is now development/training data, so its improved score is not independent accuracy.

V4 checks: 29/29 earlier reserved crop observations and 245/246 Oct-26 development observations; the full 84-frame Oct-26 pipeline yields K, 7, 3, 8, 2 and provisional Hi-Lo +1. Other full recordings still contain unknowns. See `validation/v4_evaluation.json` and `validation/v4_training_split.json`.

Reproduce V4 with `python train_rank_classifier_v4.py`; candidates go to ignored `rank_classifier_v4/`, never over these models.

SHA-256:

```text
27c926decf47a3b21af99f5eabddd430be3cb6939920a4bd9bdd34f15d995f92  rank_detector_v2.pt
a8cecebee63b60f333104b5d1290f5d351625f1ee717a65763cb9e51d1626a7e  rank_classifier_v2.pt
c15100b0bb0c3e1ff41e551ec4671bfaa768b6657e957f1d4f4d99ffd9b7d6bb  rank_classifier_v3.pt
fab0c394e428ba9e9ad3faff3f36b4ff6bfe224f9594db217f1c39c80b8d24ae  rank_classifier_v4.pt
118ca48dccbe2c91c86cf74e6ca67c4392a4a20d4de1208648e137f9d013c774  rank_reference_v1.npz
```

Reproduction in the project virtual environment:

```sh
python train_rank_classifier_v3.py
```

This writes candidates under ignored `rank_classifier_v3/`; it does not overwrite these models. GPU/platform/library differences can affect exact training results. Do not load checkpoints from untrusted sources.

## Current V5 classifier

`rank_classifier_v5.pt` is the app default. It uses the same architecture and
preprocessing, trained from seed 42 for 60 epochs on the 169 existing V4 training
crops plus 61 visually audited crops from 18 physical cards in development
recordings r1-r3. The historical 17-crop selection set chooses the checkpoint
(accuracy, then log loss); none of the new rounds enter checkpoint selection.
The selection set lacks six ranks. R4/r5 were not loaded, inspected or trained on.
Epoch 57 is selected. Training-set fit and the older correlated crop diagnostics
are not independent accuracy validation. See `validation/v5_training_audit.json`.

Reproduce from an audited local crop export (new private crops are not in Git):

```sh
python train_rank_classifier_v5.py --reviewed /path/to/audited_crops \
  --development-groups new_r1 new_r2 new_r3 --output /path/to/new_candidate
```

The output directory must not exist. The loader requires labels.csv with crop,
rank, recording_group, card_group, sha256 and annotation_source columns. It
rejects checksum/path errors, conflicting card labels and undeclared groups.
Outputs never replace published weights automatically.

V5 SHA256: `97938c1e00f4c1d4e29a10c86c61f19ee050e8bf9d3f559b9a9084fab0dd8026`.

The V2 detector, V4 checkpoint and V1 reference bank are unchanged.
