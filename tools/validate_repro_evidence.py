#!/usr/bin/env python3
"""Stage 12: verify author-side JSON audits and export a PATH-FREE public evidence summary.

Does NOT train models, perform inference, or treat retained embeddings as newly generated.
All audit JSON inputs are read-only. Reports are only accepted if they contain
all expected distinct datasets/folds, statuses and numerical checks.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

DATASETS = ('wisdm', 'uci_har', 'mhealth', 'pamap2', 'motionsense')
MODELS_HEAD = ('MantisV2', 'MOMENT-1')
MODELS_ARTIFACT = ('MantisV2','MOMENT-1','HART','CrossHAR-adapted')
FOLDS = tuple(range(5))
ROOT = Path(__file__).resolve().parents[1]

class AuditError(ValueError):
    pass

def check(condition, message):
    if not condition:
        raise AuditError(message)

def dict_(item, name):
    check(isinstance(item,dict), f'{name} must be an object')
    return item

def num(item, name):
    check(isinstance(item,(int,float)) and not isinstance(item,bool)
          and math.isfinite(float(item)), f'{name} must be a finite number')
    return float(item)

def same_keys(rows, models):
    return {(m,d,f) for m in models for d in DATASETS for f in FOLDS}

def document(path):
    p = Path(path)
    check(p.is_file(), f'Missing input report: {p}')
    data = json.loads(p.read_text(encoding='utf-8'))
    dict_(data,p.name)
    return data, hashlib.sha256(p.read_bytes()).hexdigest()

def preprocessing(report):
    check(report.get('mode') == 'hash','Preprocessing must use SHA256 hash mode')
    check(report.get('complete_for_selected_datasets') is True,'Preprocessing incomplete')
    reference=json.loads((ROOT/'reference/stage5e2_original_manifest.json').read_text())['datasets']
    ds=dict_(report.get('datasets'),'preprocessing.datasets')
    check(set(ds)==set(DATASETS),'Preprocessing must contain exactly five datasets')
    total_windows=0
    for name in DATASETS:
        e=dict_(ds[name],f'preprocessing.{name}')
        ref=reference[name]
        check(e.get('pass') is True, f'{name}: status not PASS')
        check(e.get('files_exist') is True,f'{name}: files missing')
        check(all(e.get('checks',{}).get(k) is True for k in ('shape','subjects','metadata_rows','class_counts','sha256_X.npy','sha256_metadata.csv')), f'{name}: required structure/hash checks not PASS')
        check(e.get('sha256_match',{}).get('X.npy') is True and e.get('sha256_match',{}).get('metadata.csv') is True, f'{name}: hash mismatched')
        check(e.get('sha256') == ref['sha256'], f'{name}: hashes differ from the published reference manifest')
        check(e.get('shape') == ref['shape'],f'{name}: array shape differs from reference')
        check(e.get('subjects') == ref['subjects'],f'{name}: subject count differs from reference')
        check(e.get('class_counts') == ref['class_counts'],f'{name}: class count differs from reference')
        total_windows += int(ref['shape'][0])
    return {'datasets_pass':5,'sha256_files_match':10,'total_windows':total_windows,
            'scope':'author-server raw-data replay; identical X.npy and metadata.csv outputs'}

def score_audit(report, label, expected, *, checkpoint_key=None, checkpoint_expected=None):
    check(report.get('status')=='PASS',f'{label}: not PASS')
    check(report.get('expected_runs')==expected,f'{label}: expected_runs != {expected}')
    check(report.get('metrics_match')==expected,f'{label}: not all metrics match')
    check(report.get('predictions_match')==expected,f'{label}: not all predictions match')
    check(not report.get('missing') and not report.get('mismatches'),f'{label}: missing or mismatches')
    out={'runs':expected,'metrics_match':expected,'predictions_rescored':expected}
    if checkpoint_key is not None:
        check(report.get(checkpoint_key)==checkpoint_expected,f'{label}: checkpoint compatibility count incorrect')
        ck_checked='checkpoint_checked' if 'checkpoint_checked' in report else 'checkpoints_checked'
        check(report.get(ck_checked)==checkpoint_expected,f'{label}: checkpoint checked count incorrect')
        out['sample_checkpoints_compatible' if 'remaining' not in label else 'artifact_files_present']=checkpoint_expected
    return out

def sample_artifacts(report, trusted_report):
    check(report.get('status')=='PASS','Stage 09: not PASS')
    check(trusted_report.get('status')=='PASS','Stage09 trusted joblib report not PASS')
    trusted_models=dict_(trusted_report.get('models'),'stage09.trusted_models')
    models=dict_(report.get('models'),'stage09.models')
    check(set(models)==set(MODELS_ARTIFACT),'Stage09 model set mismatch')
    check(set(trusted_models)==set(MODELS_ARTIFACT),'Stage09 trusted model set mismatch')
    counts={'MantisV2':2,'MOMENT-1':2,'HART':1,'CrossHAR-adapted':3}
    n=0
    for model,count in counts.items():
        entry=dict_(models[model],model)
        check(entry.get('status')=='PASS',f'{model}: artifact status not PASS')
        files=dict_(entry.get('files'),f'{model}.files')
        check(len(files)==count,f'{model}: wrong artifact file count')
        for name,rec in files.items():
            check(rec.get('status')=='PASS' and int(rec.get('bytes',0))>0, f'{model}.{name}: invalid artifact')
            check('sha256' in rec and len(rec['sha256'])==64,f'{model}.{name}: expected SHA256 of the file')
            n+=1
        if model in MODELS_HEAD:
            check(trusted_models[model].get('status')=='PASS' and 'trusted_joblib_structure' in trusted_models[model],
                  f'{model}: fitted classifier structure was not verified by the trusted report')
    return {'files_verified':n,'trusted_sklearn_heads':2,'scope':'saved file integrity and fitted-object structure, not encoder replay'}

def head_replay(report, *, max_diff=1e-6):
    check(report.get('status')=='PASS','Stage10 status not PASS')
    check(report.get('expected')==50 and report.get('passed')==50,'Stage10 not 50/50')
    check(not report.get('errors') and report.get('failed_checks')==0,'Stage10 contains errors')
    rows=report.get('replayed_folds')
    check(isinstance(rows,list) and len(rows)==50,'Stage10 must contain 50 fold records')
    expected=same_keys(rows,MODELS_HEAD)
    seen=set();worst=0.
    for rec in rows:
        key=(rec.get('model'),rec.get('dataset'),rec.get('fold'))
        check(key in expected and key not in seen,f'Stage10 invalid or repeated key {key}')
        seen.add(key)
        check(rec.get('status')=='PASS' and rec.get('predicted_label_mismatches')==0, f'Stage10 {key} predictions differ')
        check(int(rec.get('windows_test',0))>0,f'Stage10 {key} empty test set')
        value=num(rec.get('max_absolute_probability_error'),f'Stage10 {key} probability')
        check(value<=max_diff,f'Stage10 {key}: probability deviation {value} > {max_diff}')
        worst=max(worst,value)
    check(seen==expected,'Stage10 missing dataset/fold/model combinations')
    return {'folds_pass':50,'label_mismatches':0,'max_probability_difference':worst,
            'scope':'saved embeddings -> StandardScaler + LogisticRegression -> test predictions; no encoder rerun'}

def test_csv_rows(project, model, dataset, fold):
    if model=='HART':
        file=project/'results/stage5g2_hart_5fold'/dataset/f'fold{fold}'/'test_predictions.csv'
    else:
        file=project/'results/stage5l5b_crosshar_5fold'/dataset/f'fold{fold}'/'predictions.csv'
    check(file.is_file(),f'Missing reference predictions for {model}/{dataset}/fold{fold}')
    with file.open(newline='',encoding='utf-8') as f:
        it=csv.reader(f)
        next(it)
        return sum(1 for _ in it)

def torch_replay(report, model, *, project=None, max_diff=1e-6):
    label=f'Stage11 {model}'
    check(report.get('status')=='PASS' and report.get('pass_count')==25 and report.get('total')==25, f'{label}: not 25/25 PASS')
    check(report.get('mode')=='replay', f'{label}: strict-only mode is not inference replay')
    check(not report.get('errors'),f'{label}: errors present')
    rows=report.get('results')
    check(isinstance(rows,list) and len(rows)==25,f'{label}: expected 25 fold records')
    expected={(model,d,f) for d in DATASETS for f in FOLDS}
    seen=set(); worst=0.; total_windows=0
    for rec in rows:
        key=(rec.get('model'),rec.get('dataset'),rec.get('fold'))
        check(key in expected and key not in seen, f'{label}: repeated/invalid {key}')
        seen.add(key)
        check(rec.get('mode')=='replay' and rec.get('strict_checkpoint_load')=='PASS',f'{label} {key}: checkpoint/inference not verified')
        cmp=dict_(rec.get('comparison'),f'{label} {key}.comparison')
        check(rec.get('status')=='PASS' and cmp.get('status')=='PASS',f'{label} {key}: replay not PASS')
        check(cmp.get('label_mismatches')==0,f'{label} {key}: label mismatch')
        size=cmp.get('windows')
        check(isinstance(size,int) and size>0, f'{label} {key}: no full test windows')
        if project is not None:
            n=test_csv_rows(project,model,key[1],key[2])
            check(size==n,f'{label} {key}: PARTIAL replay: tested {size}, expected {n} test rows')
        value=num(cmp.get('max_probability_difference'),f'{label} {key} probability')
        check(value<=max_diff,f'{label} {key}: probability difference {value} > {max_diff}')
        worst=max(worst,value);total_windows+=size
    check(seen==expected,f'{label}: missing datasets/folds')
    return {'folds_pass':25,'label_mismatches':0,'max_probability_difference':worst,
            'total_test_predictions_checked':total_windows,
            'full_csv_length_checked':project is not None,
            'scope':('full HART inference from X.npy using trained weights and author-server dependency' if model=='HART'
                     else 'strict CrossHAR 3-checkpoint loading; classifier inference from retained Z_test representations')}

def validate(paths, project=None):
    reports={};digests={}
    for name,path in paths.items():
        reports[name],digests[name]=document(path)
    result={
        'schema':'HAR-reproducibility-evidence-v1',
        'result':'PASS',
        'evidence':{
            'preprocessing':preprocessing(reports['preprocessing']),
            'supervised_3_models':score_audit(reports['supervised'],'Stage06',75,checkpoint_key='checkpoint_pass',checkpoint_expected=3),
            'timesnet_patchtst_tslanet':score_audit(reports['newmodels'],'Stage07',75,checkpoint_key='checkpoint_pass',checkpoint_expected=3),
            'other_4_models':score_audit(reports['remaining'],'Stage08 remaining',100,checkpoint_key='checkpoints_pass',checkpoint_expected=8),
            'pretrained_artifacts':sample_artifacts(reports['artifacts'],reports['artifacts_trusted']),
            'frozen_heads':head_replay(reports['frozen_heads']),
            'crosshar':torch_replay(reports['crosshar'],'CrossHAR-adapted',project=project),
            'hart':torch_replay(reports['hart'],'HART',project=project),
        },
        'source_report_sha256':digests,
        'boundaries':[
            'Results match saved test predictions, not a new training of all ten models.',
            'Freshly generated MantisV2/MOMENT-1 encoder embeddings have not been verified.',
            'CrossHAR replay uses saved Z_test features; encoder-feature extraction has not been independently rerun.',
            'HART replay requires original environment, trained checkpoints, BF16 GPU and exact batch_size=1024.',
            'Preprocessing exact hashes demonstrated only against retained original author-side raw-data sources.',
            'No independent clean-machine reproduction of training, robustness, or efficiency experiments is claimed.'
        ]
    }
    return result

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('preprocessing','supervised','newmodels','remaining','artifacts','artifacts_trusted','frozen_heads','crosshar','hart'):
        ap.add_argument('--'+name.replace('_','-'),required=True,type=Path)
    ap.add_argument('--project-root',type=Path,help='Optional: prove 25 HART + 25 CrossHAR checks covered ALL saved CSV test rows')
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    paths={name:getattr(args,name) for name in ('preprocessing','supervised','newmodels','remaining','artifacts','artifacts_trusted','frozen_heads','crosshar','hart')}
    try:
        result=validate(paths,args.project_root)
    except (AuditError,ValueError,KeyError,TypeError,OSError) as e:
        print(f'STAGE12 EVIDENCE: INCOMPLETE — {e}')
        return 2
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('STAGE12 EVIDENCE: PASS')
    print('  Preprocessing SHA256: 10/10')
    print('  Original test metrics and predictions: 250/250')
    print('  Downstream frozen heads: 50/50')
    print('  CrossHAR replay: 25/25')
    print('  HART replay: 25/25')
    print('  HART/CrossHAR full test CSV lengths checked:', bool(args.project_root))
    print('  Path-free evidence summary:',args.output)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
