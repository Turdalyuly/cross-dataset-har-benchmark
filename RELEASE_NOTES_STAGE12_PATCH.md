# Stage 12 RC — post-audit patch (2026-11-08)

## Source

This package is a correction of `HAR_GITHUB_STAGE12_RELEASE_CANDIDATE(1).zip`.
It includes the author's provided `HAR_STAGE12_PUBLIC_EVIDENCE.json` unchanged.

## Changes

- Fixed `tools/validate_repro_evidence.py`: include `artifacts_trusted` in the
  `paths` mapping before `validate()` consumes the nine required reports.
- Added executable CLI regression covering all nine report SHA256 hashes, the
  `--artifacts-trusted` argument, complete HART/CrossHAR CSV-length checks,
  and path-free output. Missing trusted-artifacts evidence must be refused.
- Replaced author-specific absolute paths in the top-level Russian instructions.
- Added the path-free author-side PASS JSON in `evidence/`, and clarified
  precisely which reported checks could and could not be independently repeated.
- Added a forward-looking `STAGE13_ROADMAP.md`; no Stage 13 results are claimed.

## Verification

- All 32 original Python source files parse (AST).
- `analysis/verify_release.py --export-csv`: PASS (retained result-level checks).
- 13 synthetic/unit test scripts: PASS, including Stage 11 and new Stage 12
  CLI regression checks; these do not independently rerun original model training.
- ZIP contents checked for common author-home-path and credential patterns;
  inspection is heuristic and cannot certify the absence of all secrets.

## Limitations and release blockers

- The authors' actual retained raw data, checkpoints, embeddings and nine
  underlying Stage 05–11 JSON reports are not redistributed.
- This report is verified against the author's supplied PASS JSON, not through
  independent access to the author server.
- Upstream dependency/source versions and all redistribution licenses must still
  be pinned/confirmed before a final public release.
- Stage 13 has not been executed; neither full clean-environment training of
  250 runs nor fresh extraction for the missing models can be claimed.
