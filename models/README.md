# Model checkpoints

- `rank_detector_v2.pt`: supplied YOLO V2 localization checkpoint, unchanged.
- `rank_classifier_v2.pt`: supplied original CNN checkpoint, unchanged (recorded validation 68.2%).
- `rank_classifier_v3.pt`: new controlled CNN run with the same architecture and preprocessing, fresh initialization, seed 42. The app defaults to this checkpoint. It still makes errors and its output requires review.

V3 uses existing labels plus reviewed training snapshots from uploaded rounds 1 and 3. Uploaded round 2 was excluded from both training and checkpoint selection. On its 29 labeled crop observations, V2 got 17 correct and V3 got 24 correct. These are observations of a few repeated physical cards in one round, not 29 independent cards. See `VALIDATION.md` and the per-crop reports in `validation/`.

SHA-256:

```text
27c926decf47a3b21af99f5eabddd430be3cb6939920a4bd9bdd34f15d995f92  rank_detector_v2.pt
a8cecebee63b60f333104b5d1290f5d351625f1ee717a65763cb9e51d1626a7e  rank_classifier_v2.pt
c15100b0bb0c3e1ff41e551ec4671bfaa768b6657e957f1d4f4d99ffd9b7d6bb  rank_classifier_v3.pt
```

Reproduction in the project virtual environment:

```sh
python train_rank_classifier_v3.py
```

This writes candidates under ignored `rank_classifier_v3/`; it does not overwrite these models. GPU/platform/library differences can affect exact training results. Do not load checkpoints from untrusted sources.
