#!/usr/bin/env python3
"""Stage 09: read-only inspection of original MantisV2, MOMENT-1, HART and CrossHAR artifacts.

Default: only authors' local PyTorch .pt files are deserialized, using weights_only=True;
joblib files are never deserialized unless the author explicitly passes --trust-local-joblib.
No GPU, training, file mutation, imports from external_models, or fallback to weights_only=False.

The output verifies artifact STRUCTURE, not encoder compatibility or reproduced predictions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path

ARTIFACTS = {
    'MantisV2': ('stage5g1_mantisv2_hart_5fold', ['scaler.joblib', 'logreg.joblib'], 1536),
    'MOMENT-1': ('stage5l4_moment_5fold', ['scaler.joblib', 'logreg.joblib'], 768),
    'HART': ('stage5g2_hart_5fold', ['best.pt'], None),
    'CrossHAR-adapted': ('stage5l5b_crosshar_5fold', ['encoder_best.pt', 'contrastive_best.pt', 'classifier_best.pt'], None),
}
DATASETS = ('wisdm', 'uci_har', 'mhealth', 'pamap2', 'motionsense')


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def inspect_pytorch(path: Path) -> dict:
    import torch
    obj = torch.load(str(path), map_location='cpu', weights_only=True)
    # Support common save formats: raw state_dict or dict with a nested state dictionary.
    if not isinstance(obj, Mapping):
        raise ValueError('Checkpoint root is not a dictionary')
    root_keys = [str(x) for x in list(obj)[:25]]
    possible = ['state_dict', 'model_state_dict', 'model', 'model_state', 'net', 'encoder']
    state = obj
    for k in possible:
        if isinstance(obj.get(k), Mapping) and any(torch.is_tensor(v) for v in obj[k].values()):
            state = obj[k]
            break
    tensor_items = [(str(k), v) for k,v in state.items() if torch.is_tensor(v)]
    if not tensor_items:
        raise ValueError(f'No tensor weights in checkpoint: root keys {root_keys}')
    n_parameters = sum(int(v.numel()) for _, v in tensor_items)
    if not n_parameters:
        raise ValueError('No elements in tensor weights')
    invalid = [k for k,v in tensor_items if (v.is_floating_point() or v.is_complex()) and not bool(torch.isfinite(v).all().item())]
    if invalid:
        raise ValueError(f'Non-finite tensors: {invalid[:8]}')
    return {
        'format': 'raw state_dict' if state is obj else 'nested state_dict',
        'root_keys_sample': root_keys,
        'tensor_keys': len(tensor_items),
        'tensor_elements_including_buffers': n_parameters,
        'tensor_shapes_sample': {k:list(v.shape) for k,v in tensor_items[:12]},
        'all_tensors_finite': True,
        'warnings': ['Tensor-element count may include buffers; does not establish architecture compatibility.'],
    }


def inspect_trusted_joblib(scaler_path: Path, classifier_path: Path, n_features: int) -> dict:
    # Call only on files created locally by the researcher and explicitly trusted.
    import joblib
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    scaler = joblib.load(scaler_path)
    classifier = joblib.load(classifier_path)
    if not isinstance(scaler, StandardScaler):
        raise ValueError(f'Expected StandardScaler, got {type(scaler).__name__}')
    if not isinstance(classifier, LogisticRegression):
        raise ValueError(f'Expected LogisticRegression, got {type(classifier).__name__}')
    sc_nf = int(scaler.n_features_in_)
    cl_nf = int(classifier.n_features_in_)
    if sc_nf != n_features or cl_nf != n_features:
        raise ValueError(f'Feature dimensions: scaler={sc_nf}, classifier={cl_nf}, expected={n_features}')
    classes = np.asarray(classifier.classes_).tolist()
    if classes != [0,1,2,3]:
        raise ValueError(f'Classifier classes {classes}; expected [0,1,2,3]')
    c = np.asarray(classifier.coef_)
    if c.shape != (4,n_features) or not np.isfinite(c).all():
        raise ValueError(f'Bad classifier coefficients: {c.shape}')
    for key in ['mean_', 'scale_', 'var_']:
        arr = np.asarray(getattr(scaler,key))
        if arr.shape != (n_features,) or not np.isfinite(arr).all():
            raise ValueError(f'Invalid scaler {key} shape/values')
    if (np.asarray(scaler.scale_) <= 0).any():
        raise ValueError('Scaler has non-positive scales')
    # No use of trained classifier to make predictions or run arbitrary third-party encoders.
    return {'scaler': 'StandardScaler', 'classifier': 'LogisticRegression',
            'n_features':n_features, 'classes':classes, 'coef_shape':list(c.shape),
            'validation': 'fitted-object structural compatibility only; no encoder inference'}


def run(project: Path, dataset: str, fold: int, trust_joblib: bool, hash_files: bool) -> dict:
    project = Path(project).resolve()
    report = {'project_root': str(project), 'dataset':dataset, 'fold':fold, 'trust_local_joblib':trust_joblib,
              'models':{},'errors':[], 'warnings':[],
              'verification_scope': ['Existing files', 'SHA256 when requested',
                       'Safe PyTorch checkpoint tensor content',
                       'Optional explicitly trusted joblib fitted-object structures',
                       'Does not check third-party encoder weights or repeat model inference']}
    for model,(sub, names, dims) in ARTIFACTS.items():
        base = project/'checkpoints'/sub/dataset/f'fold{fold}'
        model_record = {'directory':str(base), 'files':{}, 'status':'PASS'}
        for name in names:
            path=base/name
            entry={'path':str(path)}
            try:
                if not path.is_file(): raise FileNotFoundError(str(path))
                entry['bytes']=path.stat().st_size
                if not entry['bytes']: raise ValueError('Empty file')
                if hash_files: entry['sha256']=sha256_file(path)
                if path.suffix == '.pt':
                    entry['torch_tensors']=inspect_pytorch(path)
                else:
                    entry['inspection']='presence only, joblib deserialization disabled'
                entry['status']='PASS'
            except Exception as exc:
                entry['status']='ERROR'
                entry['error']=f'{type(exc).__name__}: {exc}'
                report['errors'].append(f'{model}/{name}: {entry["error"]}')
                model_record['status']='INCOMPLETE'
            model_record['files'][name]=entry
        if dims is not None:
            if trust_joblib and model_record['status']=='PASS':
                try:
                    model_record['trusted_joblib_structure']=inspect_trusted_joblib(base/'scaler.joblib',base/'logreg.joblib',dims)
                except Exception as exc:
                    report['errors'].append(f'{model}: {type(exc).__name__}: {exc}')
                    model_record['status']='INCOMPLETE'
            elif not trust_joblib:
                model_record['joblib_validation']='SKIPPED: provide --trust-local-joblib ONLY for checkpoints you created and trust'
        report['models'][model]=model_record
    report['status']='PASS' if not report['errors'] else 'INCOMPLETE'
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',required=True)
    p.add_argument('--dataset',choices=DATASETS,default='wisdm')
    p.add_argument('--fold',type=int,choices=range(5),default=0)
    p.add_argument('--trust-local-joblib',action='store_true',help='OPT-IN unsafe pickle deserialization. Only for your own trusted artifacts.')
    p.add_argument('--sha256',action='store_true')
    p.add_argument('--output')
    a=p.parse_args()
    if a.trust_local_joblib:
        print('WARNING: joblib.load is unsafe for untrusted files; ONLY proceed with your own original trusted artifacts.', flush=True)
    report=run(Path(a.project_root),a.dataset,a.fold,a.trust_local_joblib,a.sha256)
    for model,info in report['models'].items():
        print(f'\n[{model}] {info["status"]}')
        for filename, result in info['files'].items():
            text=f'  {filename}: {result["status"]}'
            if 'bytes' in result:text+=f' ({result["bytes"]:,} bytes)'
            if 'torch_tensors' in result:text+=f' | {result["torch_tensors"]["tensor_keys"]} tensor keys'
            print(text)
            if 'error' in result:print('    '+result['error'])
        if 'trusted_joblib_structure' in info:print('  Trusted joblib: FITTED STRUCTURES PASS')
        if 'joblib_validation' in info:print('  Joblib: NOT DESERIALIZED')
    print('\nARTIFACT STRUCTURE AUDIT:',report['status'])
    print('NOTE: This does NOT prove model-weight compatibility or reproduce predictions.')
    if a.output:
        dest=Path(a.output);dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
        print('SAVED:',dest)
    return 0 if report['status']=='PASS' else 2

if __name__=='__main__':sys.exit(main())
