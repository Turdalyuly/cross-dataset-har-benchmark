# Stage 13 — training and feature-extraction reproducibility (planned)

Status: **not executed**. This is a scope and acceptance plan, not evidence that any new training was rerun.

## Priorities and acceptance gates

1. **Freeze environment and provenance.** Capture an installation snapshot with exact
   Python/CUDA/PyTorch/scikit-learn/transformers versions, upstream repository URLs,
   commit hashes or package revisions, model checkpoint identifiers and licenses;
   record the GPU precision, runtime options, and hardware. Never commit credentials,
   private filesystem paths, or raw personal/participant data. Separate public
   metadata from private machine-specific setup.
2. **MantisV2 and MOMENT-1.** Reconstruct standalone `X.npy -> encoder -> embeddings`
   extraction for each dataset/fold. Compare newly generated embeddings with retained
   arrays (shape, order, label alignment, numerical differences) before replaying the
   frozen StandardScaler + LogisticRegression heads. Do not treat saved embeddings as
   a successful feature-extraction replay.
3. **CrossHAR-adapted.** Reconstruct encoder, contrastive and classifier stages using
   the exact original upstream implementation; run `X.npy -> Z_test` and compare with
   retained Z_test (strict identity/ordering/numeric tolerances established *before*
   testing). Only then repeat classifier replay and report end-to-end results.
4. **HART.** Preserve the already verified original CUDA/BF16, batch size 1024
   inference configuration; add independently runnable training from the exact saved
   subject folds with deterministic seeds, checkpoint save/reload, evaluation and
   provenance output. Check a fresh run statistically: retraining need not be bitwise
   identical, and exact equality must not be claimed without evidence.
5. **Six supervised runners.** Pin TimesNet, PatchTST and TSLANet upstream versions;
   audit original hyperparameters, validation and model selection; perform at least
   one full seed/fold run per architecture before any claim of fresh training
   reproducibility. Mark remaining unrun folds clearly.
6. **Independent clean setup.** With legally obtained datasets and approved model
   licenses, run installation, harmonization, feature extraction, training and test
   prediction verification on a fresh environment. Archive audit logs and scripts,
   not sensitive datasets or disallowed third-party weights.

## Artifacts expected

- `environment.lock` or versioned environment manifests and source-commit inventory.
- Independent, documented `train` and `extract` entrypoints, checkpoint/config
  provenance, deterministic split/seeds, and exact preprocessing compatibility.
- Automated integration tests and explicit per-model audit JSON with `PASS`, `FAIL`,
  or `NOT_VERIFIED`, plus logged tolerances and rerun conditions.
- Updated manuscript Code Availability / Limitations paragraphs reflecting only
  verified paths, not inferred end-to-end reproducibility.

## Non-goals until tested

Do not claim independent full training of 250 runs, complete CrossHAR encoder replay,
new MantisV2/MOMENT-1 embeddings, or clean-machine HART training solely from Stage 12.
