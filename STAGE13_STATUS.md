# Stage 13 — author-server frozen feature extraction replay (2026-11-08)

**All three encoder replays passed on the authors' retained environment** with exact numeric equality to saved reference arrays:

| Model | Scope | Windows | Compared values | Max absolute difference |
|---|---|---:|---:|---:|
| MOMENT-1 | `X.npy` → frozen encoder → embeddings, five datasets | 96,776 | 74,323,968 | 0.0 |
| MantisV2 | `X.npy` → frozen encoder → embeddings, five datasets | 96,776 | 148,647,936 | 0.0 |
| CrossHAR-adapted | `X.npy` → saved SSL encoder → `Z_test`, five datasets × five subject-disjoint folds | 96,776 | 836,144,640 | 0.0 |

For CrossHAR, every one of the 25 folds reproduced `[N_test,120,72]` with `exact_equal=true`; `train_idx`/`val_idx`/`test_idx`, fivefold window coverage and train-only normalization were checked in the preceding Stage 13K preflight. The reproduced test windows cover each dataset exactly once across folds.

**Total compared numeric values across these three replays: 1,059,116,544.**

Public provenance and hashes of private original reports: [`evidence/HAR_STAGE13_AUTHOR_SERVER_FEATURE_EXTRACTION.json`](evidence/HAR_STAGE13_AUTHOR_SERVER_FEATURE_EXTRACTION.json). The full original reports, weights, raw data and private historical Jupyter code are not distributed.

**Claim boundary:** Exact author-server inference/feature-extraction replay using original saved checkpoints and data is not independent clean-machine replication or retraining. Stage 12 covered stored predictions and downstream classifier replays; Stage 13 expands evidence to true encoder extraction, not training.
