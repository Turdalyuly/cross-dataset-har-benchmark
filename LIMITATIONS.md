# Claims this release does and does not support

## Supported by author-server report evidence

1. Original author-side raw-data preprocessing reproduces all 5 saved harmonized `X.npy` and `metadata.csv` artifacts (10/10 SHA256 checks).
2. All **250 original clean fold scores** agree with saved predictions. This is an artifact/metric audit, not 250 new trainings.
3. MantisV2 and MOMENT-1 saved embeddings plus fitted scalers/classifiers reproduce **50/50** downstream classifier folds (Stage 12).
4. MantisV2 original frozen encoder reproduces saved embeddings for all five datasets: **96,776 windows, 148,647,936 values, exact equality** (Stage 13).
5. MOMENT-1 original frozen encoder reproduces saved embeddings for all five datasets: **96,776 windows, 74,323,968 values, exact equality** (Stage 13).
6. CrossHAR original SSL encoder checkpoints reproduce saved `Z_test` representations for **25/25** subject-disjoint folds: **96,776 windows, 836,144,640 values, exact equality** (Stage 13). Stage 12 separately reproduced classifier outputs on saved `Z_test`.
7. HART original retained checkpoints reproduce labels/probabilities on original `X.npy` test folds for **25/25**, using original CUDA/BF16 conditions and batch size 1024 (Stage 12).

## Not established

- A third-party download, fresh harmonization or independent clean-environment replay.
- New training of any original model, or exact/statistically equivalent retraining of all 250 model runs.
- Fresh end-to-end SSL training of CrossHAR or finetuning of HART; Stage 13 uses retained trained encoder checkpoints.
- Independent repeats of robustness, efficiency, LODO, directional source-target and SSL-budget ablations (summary data/scripts may be bundled).
- Independent certification of all third-party package/driver versions or licenses on a clean machine.
- Permission to redistribute upstream CrossHAR code, raw datasets, model checkpoints or pretrained weights.

**Appropriate publication wording:** “Author-server preprocessing, prediction audits, and saved-checkpoint inference/feature extraction replay were verified.” Do not say that all training experiments have been independently reproduced.

**Evidence scope:** The included JSON summaries and SHA256 pointers document author-server runs. Without the original private report files and retained datasets/weights, readers can verify internal consistency of the published evidence but cannot independently reproduce those GPU inferences from this ZIP alone.
