#!/usr/bin/env python3
"""Validate internal consistency of public Stage 13 summary. NOT raw-data verification."""
import json, re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ROWS={'wisdm':52747,'uci_har':9073,'mhealth':2300,'pamap2':8164,'motionsense':24492}
obj=json.loads((ROOT/'evidence/HAR_STAGE13_AUTHOR_SERVER_FEATURE_EXTRACTION.json').read_text(encoding='utf-8'))
assert obj['release_evidence_status']=='AUTHOR_SERVER_FEATURE_EXTRACTION_REPLAY_PASS_THREE_MODELS'
assert obj['original_project_files_modified'] is False
for name,dim in [('moment_1',768),('mantis_v2',1536)]:
 rec=obj[name]
 assert rec['dataset_count']==5 and rec['rows_compared']==sum(ROWS.values())
 assert rec['feature_dim']==dim and rec['values_compared']==sum(ROWS.values())*dim
 assert rec['max_absolute_difference']==0
 assert re.fullmatch('[a-f0-9]{64}',rec['full_private_report_sha256'])
c=obj['crosshar_encoder']; assert c['status']=='EXACT_Z_TEST_MATCH_ALL_25_FOLDS'
assert c['folds_pass']==25 and c['dataset_count']==5 and c['exact_equal'] is True
assert c['test_rows_compared']==sum(ROWS.values()) and c['features_per_test_window']==120*72
assert c['values_compared']==sum(ROWS.values())*120*72 and c['max_absolute_difference']==0
assert set(c['per_dataset'])==set(ROWS)
for ds,n in ROWS.items():
 r=c['per_dataset'][ds]
 assert r['folds']==5 and r['rows']==n and r['values']==n*120*72
assert re.fullmatch('[a-f0-9]{64}',c['full_private_replay_report_sha256'])
assert re.fullmatch('[a-f0-9]{64}',c['full_private_preflight_sha256'])
assert (ROOT/'evidence/HAR_STAGE12_PUBLIC_EVIDENCE.json').is_file()
print('PASS Stage 13 published evidence internal-consistency check (NOT independent GPU replay)')
