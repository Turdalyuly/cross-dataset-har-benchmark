# Stage 13 verified author-side release candidate — 2026-11-08

This package upgrades the Stage 13 *partial* candidate after successful CrossHAR GPU replay and **supersedes its stale CrossHAR-pending statements**. The Stage 12 JSON is retained unchanged as a point-in-time audit with earlier limitations; the Stage 13 JSON documents the subsequent encoder tests.

## Verified report inputs

- MOMENT-1 five-dataset original frozen encoder replay: 96,776 windows, 74,323,968 values, exact match. Private original JSON SHA256: `e8748f8ab97e9bd495667ce426b79df9ac6e6f82915a5e6194e0d6ea14c22e58`.
- MantisV2 five-dataset original frozen encoder replay: 96,776 windows, 148,647,936 values, exact match. Private original JSON SHA256: `8d75bca8d36cdf12d33ec4d82575607460b47c7ff3a0d1797a0a50fdde3cdf6e`.
- CrossHAR original trained SSL encoder `X.npy → Z_test`: **25/25 folds**, 96,776 test windows, 836,144,640 values, exact match. Private original JSON SHA256: `3117181cbd1908e7a80ad06a7fe07757c20759a8cde0d1f365da8b500c6ad673`.
- CrossHAR preflight 25/25 passes, original JSON SHA256: `605a63cf9551a98cb96eae4131bc3a0f8e494ed6896df90ef3dbe939e2373a2f`.
- All reports are author-provided and relate to retained author-side artifacts, not independent clean-room replication.

## Public evidence

`evidence/HAR_STAGE13_AUTHOR_SERVER_FEATURE_EXTRACTION.json` is a path-free summary of these runs and points to exact source-report SHA256 digests. Its CrossHAR numeric summary is supported by 25 unique `(dataset, fold)` records, five complete partitions and exact 0.0 maximum error.

## Scope and release blockers

- Existing original data, 250 neural training runs, robustness/efficiency reruns and third-party clean-machine installation are NOT reproduced in this release.
- Original pretrained models, historical private code, private reports and data are NOT included.
- Verify a suitable **repository license and third-party upstream license compliance** before public publishing. CrossHAR's original checkout has no bundled LICENSE file.
- Do not treat the included Stage 12 document's former “embedding not yet verified” language as the final status; consult Stage 13 and `LIMITATIONS.md`.
