# Author-server reproducibility audit — 2026-11-08

**These findings come from the authors' successful server command logs.** The checkpoint and raw dataset files themselves are not included in this ZIP. The accompanying `tools/validate_repro_evidence.py` can validate the nine retained source reports and create a path-free summary. The author's PASS output is bundled as [`evidence/HAR_STAGE12_PUBLIC_EVIDENCE.json`](evidence/HAR_STAGE12_PUBLIC_EVIDENCE.json), with nine source JSON SHA256 fingerprints. The source reports and original artifacts are not distributed, so these findings are author-server verified rather than independently repeated by the release reader.

| Component | Real-data verification | Precise scope |
|---|---|---|
| Stage 05: preprocessing | 5/5 datasets; 10/10 SHA256 | Exact outputs `X.npy` and `metadata.csv` relative to original Stage 5E.2 manifest |
| Stage 06: CNN1D/ResNet1D/TCN | 75/75 fold metrics and test prediction recalculations; 3/3 sample checkpoint compatibility | Saved results and representative weight-to-architecture compatibility; not 75 retrainings |
| Stage 07: TimesNet/PatchTST/TSLANet | 75/75 scores and prediction recalculations; 3/3 sample checkpoint compatibility | Original Stage 5L.2 + 5L.3 outputs; not 75 retrainings |
| Stage 08: MantisV2/MOMENT-1/HART/CrossHAR | 100/100 scores and prediction recalculations, 8/8 sample artifacts present | Saved prediction integrity; file existence is not weight compatibility |
| Stage 09: four model artifact sets | 8/8 structure checks; trusted fitted sklearn objects verified | Uses only trusted original joblib in opt-in mode; does not run encoders |
| Stage 10: MantisV2/MOMENT-1 | 50/50 downstream test replays; zero label mismatches; max probability difference ~1.11e-16 | Saved encoder embeddings → scaler → logistic regression |
| Stage 11: CrossHAR | 25/25 replays, strict 3-checkpoint loading each; max probability difference ~2.98e-8 | Saved `Z_test` representations → classifier; not complete encoder inference |
| Stage 11: HART | 25/25 replays from `X.npy`, strict checkpoint load; max probability difference ~2.98e-8 | Full original saved test windows; CUDA/BF16, **batch size 1024** |

**HART batch-size sensitivity.** A diagnostic 256-window run with batch 128 initially passed using tolerance 0.005, but the full WISDM fold 0 with batch 128 had one wrong label and maximum probability difference 0.0156199 (failure). Using the original evaluation batch 1024 removed the mismatch (maximum probability difference approximately 2.98e-8), and all subsequent 25 folds passed. Do not silently increase numeric tolerances to bypass discrepancies.

**Stage 13 addendum:** Retained `X.npy` and saved encoder checkpoints now reproduce the original MantisV2 and MOMENT-1 frozen embeddings for all five datasets, and the CrossHAR encoder reproduces original `Z_test` for all 25 subject-disjoint folds. All three have exact numeric equality. See `STAGE13_STATUS.md` and `evidence/HAR_STAGE13_AUTHOR_SERVER_FEATURE_EXTRACTION.json`. The Stage 11 CrossHAR classifier replay above remains a distinct verification step.

**Outstanding work:** Independent fresh raw-data download/preparation; upstream redistribution license checks (Git commits inventoried); verified standalone original training entrypoints for all model classes; clean-machine end-to-end training and inference; independent robustness and efficiency evaluation. The absence of these items must be disclosed in any journal/code availability statement.
