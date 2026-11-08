"""Shared read-only verification helpers for HAR Stage 11."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

DATASETS = ('wisdm', 'uci_har', 'mhealth', 'pamap2', 'motionsense')
CLASSES = ('walking','sitting','standing','stairs')


def load_split(project: Path, dataset: str, fold: int):
    p = project/'splits'/'harmonized_hart_v1'/f'{dataset}_5fold.json'
    with p.open(encoding='utf-8') as fh: item=json.load(fh)
    matches=[r for r in item['folds'] if int(r['fold'])==int(fold)]
    if len(matches)!=1: raise ValueError(f'Expected one fold {fold} in {p}, found {len(matches)}')
    f=matches[0]
    groups=[set(map(str,f[k])) for k in ['train_subjects','val_subjects','test_subjects']]
    if any(groups[i]&groups[j] for i in range(3) for j in range(i+1,3)):
        raise ValueError(f'Subject leakage in {p} fold{fold}')
    return f


def test_indices(meta: pd.DataFrame, split: dict):
    if 'subject' not in meta or 'label_id' not in meta:
        raise ValueError('metadata.csv missing subject / label_id')
    subject=meta['subject'].astype(str).to_numpy()
    return np.flatnonzero(np.isin(subject, list(map(str,split['test_subjects']))))


def compare_saved(df: pd.DataFrame, predicted: np.ndarray, probability: np.ndarray,
                  expected_idx: np.ndarray|None=None, actual_ids: np.ndarray|None=None,
                  max_prob_diff=1e-4, require_probs=True) -> dict:
    predicted=np.asarray(predicted,dtype=np.int64)
    probability=np.asarray(probability,dtype=np.float64)
    if predicted.ndim!=1 or probability.shape!=(len(predicted),4):
        raise ValueError(f'Invalid output dimensions: labels {predicted.shape}, probability {probability.shape}')
    if len(df)!=len(predicted):
        raise ValueError(f'Row count: saved {len(df)}, inference {len(predicted)}')
    if not np.isfinite(probability).all(): raise ValueError('Nonfinite inference probability')
    if not np.allclose(probability.sum(axis=1),1,atol=1e-4):
        raise ValueError('Probabilities do not sum to 1')
    if expected_idx is not None:
        if 'global_idx' in df:
            if not np.array_equal(df['global_idx'].to_numpy(dtype=np.int64),expected_idx):
                raise ValueError('Saved CSV global_idx differs from replay indices')
        elif actual_ids is not None and 'window_id' in df:
            if not np.array_equal(df['window_id'].astype(str).to_numpy(),np.asarray(actual_ids).astype(str)):
                raise ValueError('Saved CSV window_id differs from replay window_id')
        else:
            raise ValueError('Cannot verify test-window identity in saved CSV')
    if 'y_pred' not in df or 'y_true' not in df: raise ValueError('CSV missing y_pred or y_true')
    orig=df['y_pred'].to_numpy(dtype=np.int64)
    mismatches=int(np.sum(predicted!=orig))
    cols=[f'prob_{c}' for c in CLASSES]
    if require_probs and not all(x in df.columns for x in cols):
        raise ValueError(f'CSV missing probability columns: {set(cols)-set(df.columns)}')
    if all(x in df.columns for x in cols):
        reference=df[cols].to_numpy(dtype=np.float64)
        if not np.isfinite(reference).all(): raise ValueError('Nonfinite saved probability')
        max_difference=float(np.max(np.abs(reference-probability))) if len(df) else 0.
    else:
        max_difference=None
    ok=(mismatches==0 and (max_difference is not None and max_difference<=max_prob_diff))
    return {'status':'PASS' if ok else 'MISMATCH', 'windows':len(df),
            'label_mismatches':mismatches,'max_probability_difference':max_difference,
            'probability_tolerance':max_prob_diff,
            'note':'Exact labels and tolerance-based probabilities; not a retraining test.'}


def dump_report(output: str|None, report: dict):
    if not output:return
    p=Path(output); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    print('SAVED:',p)
