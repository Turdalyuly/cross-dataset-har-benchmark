# Cross-dataset HAR benchmark — verified author-side reproducibility release candidate

**Manuscript:** *Cross-Dataset Human Activity Recognition with Time-Series Foundation Models: A Subject-Disjoint Benchmark of Generalization, Robustness, and Efficiency*.

> **STATUS (2026-11-08):** Author-side preprocessing, 250 saved-score/prediction audits, and downstream inference checks passed. **Stage 13 additionally reproduced exact original saved features** from the original retained inputs and checkpoints: MOMENT-1 (5 datasets), MantisV2 (5 datasets), and CrossHAR SSL encoder (25 subject-disjoint folds). No independent clean-machine reproduction or model retraining is claimed. See [Stage 13 status](STAGE13_STATUS.md), [AUDIT.md](AUDIT.md) and [LIMITATIONS.md](LIMITATIONS.md).

## Benchmark

Five public HAR datasets: WISDM, UCI-HAR, MHEALTH, PAMAP2 and MotionSense. The harmonized sample has three acceleration axes, **30 Hz**, windows of **4 s / 120 samples**, and **75% overlap** (30-sample hop). Labels: walking, sitting, standing and stairs. Each dataset has five saved participant-disjoint train/validation/test folds.

**Ten models:** CNN1D, ResNet1D, TCN, TimesNet, PatchTST, TSLANet, MantisV2, MOMENT-1, HART, CrossHAR-adapted.

## Contents

| Folder | Purpose | What has been checked |
|---|---|---|
| `preprocessing/` | Reconstructed full raw-to-120×3 harmonization | All five author-server outputs match retained SHA256 reference (10 files) |
| `splits/` | 25 participant-disjoint splits and cross-domain protocols | Subject identity/partition checks |
| `results/` | 250 clean fold scores and R1/R2/R3/R4 derived data | Independent aggregate checks; original predictions audited on authors' server |
| `analysis/` | Recreate reported summary tables from retained results | Offline aggregate verification |
| `models/`, `experiments/` | Dry-run-by-default model training for six supervised models | Offline synthetic tests and selected real-checkpoint compatibility checks; **new full training not re-executed** |
| `tools/` | Read-only metrics/checkpoint/partial and full inference audits for all 10 models | All specified real-data audits passed on author-server |
| `reference/` | Original Stage 5E.2 manifest and source provenance | SHA256 source-of-truth manifest retained |
| `tests/` | Synthetic corruption, import, architecture and runner tests | Executed locally; does not replace original data verification |

## Fast independent checks (no original datasets or GPU)

Use an existing **trusted** Python environment with NumPy, pandas, SciPy, scikit-learn and PyTorch. The external Time-Series-Library, TSLANet, TSPulse/HART and CrossHAR sources are **not redistributed**.

```bash
python analysis/verify_release.py --export-csv
python tests/test_preprocess_components.py
python tests/test_split_subject_identity.py
python tests/test_stage12_evidence.py
```

Optional model smoke tests (CPU, use modest CPU thread count):

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python tests/test_supervised_models.py
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python tests/test_supervised_runner_synthetic.py
python tests/test_foundation_factory_stubs.py
python tests/test_new_models_runner_synthetic.py
python tests/test_stage11_synthetic.py
```

## Recreate harmonized data

Obtain the original raw public datasets under the directory structure expected by `preprocessing/build_harmonized.py`; **data licenses are the responsibility of the downloader**. Do not commit downloaded datasets or non-public participant information to GitHub.

```bash
export HAR_PROJECT_ROOT=/path/to/local/project_with_original_raw_data
export HAR_REPLAY_ROOT=/path/to/new/isolated/replay
python preprocessing/build_harmonized.py
python preprocessing/verify_preprocessing.py --replay-root "$HAR_REPLAY_ROOT" --mode hash
```

Authors' replay already achieved `ALL CHECKS PASS` for five `X.npy` and five `metadata.csv`. The replay was run against the retained original files, not against an independent re-download by a third party.

## Audits and inference replay (authors' retained project)

For full report commands, see [RELEASE_AUDIT_COMMANDS.md](RELEASE_AUDIT_COMMANDS.md). The reporters are read-only on the existing model artifacts; any JSON output is written outside the original results directories.

**Important inference conditions:** HART is reproduced with CUDA/BF16 and **batch size 1024**, because changing it to 128 produced a numerical mismatch in one full WISDM fold. CrossHAR requires a consistent CUDA device on this author's original installation. The Stage 11 classifier replay uses saved `Z_test` representations; **Stage 13 independently reran the saved CrossHAR encoder from original `X.npy` to `Z_test`** across all 25 folds.

## Training scripts available

`experiments/supervised_clean.py` implements dry-run-by-default CNN1D, ResNet1D, TCN training. `experiments/new_models_clean.py` similarly covers TimesNet, PatchTST and TSLANet. Both require `--train` to actually train. They write into a **new** output location and refuse to overwrite existing run folders.

The original author-side upstream Git commits and package versions are inventoried in [UPSTREAM_PROVENANCE.md](UPSTREAM_PROVENANCE.md); clean-machine installation and third-party redistribution rights remain unverified. The other four models have **inference/audit scripts**, including the author-side CrossHAR encoder replay tool, but do **not** have verified standalone full-training pipelines. Historical notebooks are not a substitute for independent retraining.

## Included author-server evidence (Stage 12)

The supplied [`evidence/HAR_STAGE12_PUBLIC_EVIDENCE.json`](evidence/HAR_STAGE12_PUBLIC_EVIDENCE.json)
records an author-server PASS with nine SHA256 source report digests, 10/10 preprocessing
hash checks, 250/250 score/prediction audits, 50/50 frozen-head replays, and 25/25 each
for HART and CrossHAR. This package also includes a regression test verifying that the
CLI consumes the required `--artifacts-trusted` report; it cannot validate the nine
unbundled source reports on a third-party machine without receiving them separately.

## Publication status

This is a **GitHub release candidate for author-side verified data processing, saved predictions, and saved-checkpoint feature extraction/inference**. It should not be represented as a final *full-training reproducibility* repository. No original raw datasets, .pt/.joblib checkpoints, saved embeddings or confidential/local project configuration are included. Before publishing, choose a repository license and verify third-party license compliance; upstream Git commits have been inventoried in [UPSTREAM_PROVENANCE.md](UPSTREAM_PROVENANCE.md). The verified author-side evidence summary is included under [`evidence/HAR_STAGE12_PUBLIC_EVIDENCE.json`](evidence/HAR_STAGE12_PUBLIC_EVIDENCE.json). The nine underlying source JSON audit reports are not redistributed and the server checks have not been independently rerun in this repository.


## Stage 13 feature-extraction audit (author-side)

**ALL THREE VERIFIED:** MOMENT-1 and MantisV2 reproduced frozen feature embeddings exactly for all five datasets (96,776 rows each). CrossHAR reproduced original saved SSL encoder `Z_test` exactly for all **25 folds** (96,776 test windows, 836,144,640 values). The three replays together compared **1,059,116,544 values** and had max absolute difference **0.0**. No training was rerun.

See [Stage 13 status](STAGE13_STATUS.md), [public Stage 13 evidence](evidence/HAR_STAGE13_AUTHOR_SERVER_FEATURE_EXTRACTION.json), [CrossHAR author-side replay command](CROSSHAR_ENCODER_REPLAY_RUS.md), and [Stage 13 release notes](RELEASE_NOTES_STAGE13_VERIFIED.md). Run `python analysis/verify_stage13_evidence.py` to check the published summary's internal consistency (no GPU required; does not replay source data).
