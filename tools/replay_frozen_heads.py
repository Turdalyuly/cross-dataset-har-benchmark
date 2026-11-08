#!/usr/bin/env python3
"""Stage 10: replay MantisV2/MOMENT-1 *downstream heads* from stored embeddings.

Read-only: original embedding .npy, harmonized metadata, subject-split JSON, trusted
local scaler/logreg joblib and recorded predictions. No GPUs, encoder inference,
training, checkpoint writes, or downloads. The original embeddings must exist.

SECURITY: joblib.load is pickle-based; only run with --trust-local-joblib for local,
original artifacts you created and trust. Never load arbitrary downloaded joblib.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

DATASETS = ('wisdm', 'uci_har', 'mhealth', 'pamap2', 'motionsense')
FOLDS = range(5)
LABELS = (0, 1, 2, 3)
CLASS_NAMES = ('walking', 'sitting', 'standing', 'stairs')
SPEC = {
    'MantisV2': {
        'feature_dim': 1536,
        'embedding_root': 'results/embeddings/mantisv2_hart',
        'checkpoint_root': 'checkpoints/stage5g1_mantisv2_hart_5fold',
        'prediction_root': 'results/stage5g1_mantisv2_hart_5fold',
        'prediction_file': 'test_predictions.csv',
        'prediction_rule': 'clf.predict',
        'post_scaler_cast': 'float32',
    },
    'MOMENT-1': {
        'feature_dim': 768,
        'embedding_root': 'results/stage5l4_moment_5fold/embeddings',
        'checkpoint_root': 'checkpoints/stage5l4_moment_5fold',
        'prediction_root': 'results/stage5l4_moment_5fold/folds',
        'prediction_file': 'predictions.csv',
        'prediction_rule': 'argmax(predict_proba)',
        'post_scaler_cast': None,
    },
}


def read_split(project: Path, dataset: str, fold: int) -> dict:
    path = project / 'splits' / 'harmonized_hart_v1' / f'{dataset}_5fold.json'
    splits = json.loads(path.read_text(encoding='utf-8'))['folds']
    matches = [s for s in splits if int(s['fold']) == fold]
    if len(matches) != 1:
        raise ValueError(f'Expected one split for {dataset} fold{fold}, got {len(matches)}')
    return matches[0]


def _assert_partition(subjects: np.ndarray, split: dict) -> np.ndarray:
    sets = [set(int(v) for v in split[f'{part}_subjects']) for part in ('train', 'val', 'test')]
    if any(not s for s in sets):
        raise ValueError('Empty subject split')
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise ValueError('Subject-leakage / overlapping split')
    observed = set(int(x) for x in subjects)
    if not sets[2] <= observed:
        raise ValueError(f'Test subjects absent from metadata: {sets[2] - observed}')
    if set.union(*sets) != observed:
        raise ValueError(f'Split subjects mismatch: absent={observed-set.union(*sets)}, extra={set.union(*sets)-observed}')
    test_idx = np.flatnonzero(np.isin(subjects, sorted(sets[2])))
    if len(test_idx) == 0:
        raise ValueError('No test windows')
    return test_idx


def replay_fold(project: Path, model: str, dataset: str, fold: int,
                *, prob_atol: float = 1e-5, feature_atol: float = 1e-5) -> dict[str, Any]:
    """Run classification from stored embeddings, check every original CSV row."""
    # feature_atol reserved for diagnostic compare of stored scaler features if added later.
    del feature_atol
    import joblib  # Import only where explicit opt-in is enforced by main().
    spec = SPEC[model]
    p = project.resolve()
    efile = p / spec['embedding_root'] / dataset / 'embeddings.npy'
    mfile = p / 'datasets' / 'processed' / 'harmonized_hart_v1' / dataset / 'metadata.csv'
    ckpts = p / spec['checkpoint_root'] / dataset / f'fold{fold}'
    predfile = p / spec['prediction_root'] / dataset / f'fold{fold}' / spec['prediction_file']

    # Explicit paths are part of the protocol; never search for an alternative run.
    for required in (efile, mfile, ckpts/'scaler.joblib', ckpts/'logreg.joblib', predfile):
        if not required.is_file():
            raise FileNotFoundError(str(required))

    meta = pd.read_csv(mfile)
    if not {'subject', 'label_id'} <= set(meta.columns):
        raise ValueError('Harmonized metadata requires subject and label_id columns')
    y = meta.label_id.to_numpy(dtype=np.int64)
    subjects = meta.subject.to_numpy(dtype=np.int64)
    if not np.isin(y, LABELS).all():
        raise ValueError('Unexpected harmonized class IDs')
    z = np.load(efile, mmap_mode='r', allow_pickle=False)
    if z.ndim != 2 or z.shape != (len(meta), spec['feature_dim']):
        raise ValueError(f'Embedding shape {z.shape} != {(len(meta), spec["feature_dim"])}')
    test_idx = _assert_partition(subjects, read_split(p, dataset, fold))
    data = np.asarray(z[test_idx], dtype=np.float32)
    del z
    if not np.isfinite(data).all():
        raise ValueError('Non-finite test embeddings')

    scaler = joblib.load(ckpts / 'scaler.joblib')
    clf = joblib.load(ckpts / 'logreg.joblib')
    if type(scaler) is not StandardScaler or type(clf) is not LogisticRegression:
        raise TypeError('Unexpected trusted joblib object type')
    if int(scaler.n_features_in_) != spec['feature_dim'] or int(clf.n_features_in_) != spec['feature_dim']:
        raise ValueError('Scaler or classifier feature dimensions do not match embeddings')
    classes = np.asarray(clf.classes_, dtype=int)
    if classes.tolist() != list(LABELS):
        raise ValueError(f'Expected classes {LABELS}, got {classes.tolist()}')
    transformed = scaler.transform(data)
    if spec['post_scaler_cast'] == 'float32':
        transformed = transformed.astype(np.float32)
    raw_prob = clf.predict_proba(transformed)
    probability = np.zeros((len(data), len(LABELS)), dtype=np.float64)
    for j, cl in enumerate(classes):
        probability[:, int(cl)] = raw_prob[:, j]
    if spec['prediction_rule'] == 'clf.predict':
        predicted = clf.predict(transformed).astype(np.int64)
    else:
        predicted = np.argmax(probability, axis=1).astype(np.int64)

    saved = pd.read_csv(predfile)
    if len(saved) != len(test_idx):
        raise ValueError(f'Prediction count {len(saved)} != test windows {len(test_idx)}')
    for col in ('y_true', 'y_pred'):
        if col not in saved.columns:
            raise ValueError(f'Missing column {col} in original predictions')
    yt = y[test_idx]
    original_y = saved.y_true.to_numpy(dtype=np.int64)
    original_pred = saved.y_pred.to_numpy(dtype=np.int64)
    if not np.array_equal(original_y, yt):
        raise ValueError('Original y_true differs from metadata test order')
    if 'global_idx' in saved and not np.array_equal(saved.global_idx.to_numpy(dtype=np.int64), test_idx):
        raise ValueError('Original global_idx differs from original subject test order')
    if 'subject' in saved and not np.array_equal(saved.subject.to_numpy(dtype=np.int64), subjects[test_idx]):
        raise ValueError('Original subject column differs from metadata test order')
    disagreement = np.flatnonzero(original_pred != predicted)
    probs_missing = [f'prob_{c}' for c in CLASS_NAMES if f'prob_{c}' not in saved.columns]
    prob_error = None
    if not probs_missing:
        saved_prob = saved[[f'prob_{c}' for c in CLASS_NAMES]].to_numpy(dtype=np.float64)
        if saved_prob.shape != probability.shape or not np.isfinite(saved_prob).all():
            raise ValueError('Invalid original test probabilities')
        prob_error = float(np.max(np.abs(probability-saved_prob)))
    f1_original = float(f1_score(yt, original_pred, labels=LABELS, average='macro', zero_division=0))
    f1_replay = float(f1_score(yt, predicted, labels=LABELS, average='macro', zero_division=0))
    ok = len(disagreement) == 0 and (prob_error is not None and prob_error <= prob_atol)
    return {
        'model': model, 'dataset': dataset, 'fold': fold, 'status': 'PASS' if ok else 'FAIL',
        'windows_test': len(test_idx), 'embeddings': str(efile.relative_to(p)),
        'checkpoint_folder': str(ckpts.relative_to(p)),
        'original_predictions': str(predfile.relative_to(p)),
        'predicted_label_mismatches': int(len(disagreement)),
        'first_mismatch_local_rows': disagreement[:10].tolist(),
        'max_absolute_probability_error': prob_error,
        'missing_probability_columns': probs_missing,
        'probability_tolerance': prob_atol,
        'original_macro_f1': f1_original, 'replayed_macro_f1': f1_replay,
        'verification_scope': 'Frozen downstream head from RETAINED embeddings; no encoder inference',
    }


def run(project: Path, models: list[str], datasets: list[str], folds: list[int],
        *, trust_local_joblib: bool, prob_atol: float = 1e-5) -> dict[str, Any]:
    if not trust_local_joblib:
        raise PermissionError('joblib.load requires explicit --trust-local-joblib for original trusted artifacts')
    rows = []
    errors = []
    for model in models:
        for dataset in datasets:
            for fold in folds:
                try:
                    result = replay_fold(project, model, dataset, fold, prob_atol=prob_atol)
                    rows.append(result)
                    print(f'{model:10} {dataset:12} fold{fold}: {result["status"]} '
                          f'labels={result["predicted_label_mismatches"]} mismatches '
                          f'max|Δprob|={result["max_absolute_probability_error"]}', flush=True)
                except Exception as exc:
                    msg = f'{model}/{dataset}/fold{fold}: {type(exc).__name__}: {exc}'
                    errors.append(msg)
                    print('ERROR', msg, flush=True)
    passed = sum(r['status'] == 'PASS' for r in rows)
    expected = len(models)*len(datasets)*len(folds)
    report = {
        'status': 'PASS' if passed == expected and not errors else 'INCOMPLETE',
        'project_root': str(project.resolve()), 'expected': expected,
        'passed': passed, 'failed_checks': expected-passed,
        'replayed_folds': rows, 'errors': errors,
        'scope': [
            'Reproduces saved LogisticRegression head predictions from saved pretrained embeddings',
            'Checks labels, original row order, probabilities and fixed subject splits',
            'Does not rerun MantisV2/MOMENT-1 embedding extraction',
            'Does not establish independent reproducibility from raw data or the entire original training protocol',
            'Original trusted joblib objects loaded explicitly by user choice',
        ],
    }
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root', required=True, type=Path)
    ap.add_argument('--model', choices=['all', *SPEC], default='all')
    ap.add_argument('--dataset', choices=['all', *DATASETS], default='wisdm')
    ap.add_argument('--fold', choices=['all', *[str(x) for x in FOLDS]], default='0')
    ap.add_argument('--trust-local-joblib', action='store_true', help='Only for your original locally created checkpoints')
    ap.add_argument('--prob-atol', type=float, default=1e-5)
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    if args.prob_atol < 0:
        ap.error('--prob-atol must be nonnegative')
    if not args.trust_local_joblib:
        ap.error('Refusing to deserialize joblib without --trust-local-joblib (ONLY original trusted artifacts)')
    models = list(SPEC) if args.model == 'all' else [args.model]
    datasets = list(DATASETS) if args.dataset == 'all' else [args.dataset]
    folds = list(FOLDS) if args.fold == 'all' else [int(args.fold)]
    report = run(args.project_root, models, datasets, folds,
                 trust_local_joblib=args.trust_local_joblib, prob_atol=args.prob_atol)
    print(f'\nDOWNSTREAM INFERENCE REPLAY: {report["status"]} ({report["passed"]}/{report["expected"]})', flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print('SAVED:', args.output)
    return 0 if report['status'] == 'PASS' else 2


if __name__ == '__main__':
    sys.exit(main())
