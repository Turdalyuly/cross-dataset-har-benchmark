# Generate a machine-checkable, path-free reproducibility summary

The author has already successfully run the Stage 05–11 tools. This step **does not rerun preprocessing, training or inference**: it reads nine existing JSON reports, checks the claimed counts and numerical accuracy, and writes a public evidence summary with no machine paths or private data.

From the repository root, adjust `PROJECT` to your own existing author-side project:

```bash
PY=python
PROJECT=/path/to/HAR_Q1_2026
REPORTS=/path/to/folder/with/stage_reports

"$PY" tools/validate_repro_evidence.py \
  --preprocessing "$PROJECT/repro_stage5e2_public_v3/REPLAY_VERIFICATION_PROGRESS.json" \
  --supervised "$REPORTS/stage06_1_supervised_audit.json" \
  --newmodels "$REPORTS/stage07_1_checkpoints.json" \
  --remaining "$REPORTS/stage08_artifacts.json" \
  --artifacts "$REPORTS/stage09_safe_artifacts.json" \
  --artifacts-trusted "$REPORTS/stage09_trusted_artifacts.json" \
  --frozen-heads "$REPORTS/stage10_all50_downstream.json" \
  --crosshar "$REPORTS/stage11_crosshar_all25.json" \
  --hart "$REPORTS/stage11_hart_all25.json" \
  --project-root "$PROJECT" \
  --output "$REPORTS/HAR_STAGE12_PUBLIC_EVIDENCE.json"
```

The optional `--project-root` makes the verifier check that **every** HART/CrossHAR replay included the full number of rows present in the original prediction CSV, preventing accidental publication of a successful 256-window-only test as full inference.

Expected summary:

```text
STAGE12 EVIDENCE: PASS
  Preprocessing SHA256: 10/10
  Original test metrics and predictions: 250/250
  Downstream frozen heads: 50/50
  CrossHAR replay: 25/25
  HART replay: 25/25
  HART/CrossHAR full test CSV lengths checked: True
```

A validated author-server summary is provided as `evidence/HAR_STAGE12_PUBLIC_EVIDENCE.json`; it does not contain the nine source reports. Running this command against all author-side inputs independently should reproduce the expected summary. Check the final JSON manually before uploading/publication. The generated file contains no original paths, embeddings, checkpoint bytes, participant metadata or user personal information. Each source JSON's SHA256 digest is included as a provenance fingerprint, not the full input document.

## Standalone audits (optional)

```bash
python tools/audit_original_supervised.py --project-root "$PROJECT" --check-checkpoints sample
python tools/audit_original_foundation.py --project-root "$PROJECT" --check-checkpoints sample
python tools/audit_original_remaining.py --project-root "$PROJECT" --check-artifacts sample
python tools/audit_original_pretrained_artifacts.py --project-root "$PROJECT" --sha256
# ONLY on the authors' own trusted joblib artifacts:
python tools/audit_original_pretrained_artifacts.py --project-root "$PROJECT" --trust-local-joblib
python tools/replay_frozen_heads.py --project-root "$PROJECT" --dataset all --fold all --model all --trust-local-joblib

# Select a GPU appropriate to the machine. The original runs used CUDA and local upstream sources.
CUDA_VISIBLE_DEVICES=2 python tools/replay_crosshar_strict.py --project-root "$PROJECT" --dataset all --fold all --mode replay --device cuda:0
CUDA_VISIBLE_DEVICES=2 python tools/replay_hart_strict.py --project-root "$PROJECT" --dataset all --fold all --mode replay --device cuda:0 --batch-size 1024
```

Full checkpoint loading may import upstream source files. Use only trusted local third-party installations and original author-created checkpoint files; joblib deserialization has code-execution risks and requires explicit opt-in. No `weights_only=False` is used for PyTorch artifacts.


## Stage 13 completed encoder replay (2026-11-08)

Check published author-side summary (requires no original private data or GPU):

```bash
python analysis/verify_stage13_evidence.py
```

For original author-side CrossHAR GPU replay with private original project and source dependencies, see [CROSSHAR_ENCODER_REPLAY_RUS.md](CROSSHAR_ENCODER_REPLAY_RUS.md). This path-free summary records verified **MOMENT-1, MantisV2 and CrossHAR** saved-checkpoint feature extraction.
