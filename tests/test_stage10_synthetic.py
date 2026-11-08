"""Recreate two tiny original-like MantisV2/MOMENT-1 folders and test full replay."""
from __future__ import annotations
import importlib.util
import json
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

SCRIPT = Path(__file__).resolve().parents[1] / 'tools' / 'replay_frozen_heads.py'
spec = importlib.util.spec_from_file_location('stage10', SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

rng = np.random.default_rng(2026)
with tempfile.TemporaryDirectory() as tmp:
    project = Path(tmp)
    dataset = 'mhealth'
    n = 60
    subjects = np.repeat(np.arange(1,7),10)
    labels = np.tile(np.arange(4),15)
    metadata = pd.DataFrame({'subject': subjects, 'label_id': labels, 'window_id': np.arange(n)})
    mfile = project / 'datasets/processed/harmonized_hart_v1' / dataset / 'metadata.csv'
    mfile.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(mfile, index=False)
    split_file = project / 'splits/harmonized_hart_v1' / f'{dataset}_5fold.json'
    split_file.parent.mkdir(parents=True, exist_ok=True)
    split_file.write_text(json.dumps({'folds':[{'fold':0,'train_subjects':[1,2,3],
        'val_subjects':[4], 'test_subjects':[5,6]}]}),encoding='utf-8')
    test_idx = np.flatnonzero(np.isin(subjects,[5,6]))
    train_idx = np.flatnonzero(np.isin(subjects,[1,2,3]))
    for model, conf in mod.SPEC.items():
        Z = rng.normal(0, 1, size=(n, conf['feature_dim'])).astype(np.float32)
        # Use a stable four-class signal.
        Z[:, :4] += 4*np.eye(4,dtype=np.float32)[labels]
        emb = project/conf['embedding_root']/dataset/'embeddings.npy'
        emb.parent.mkdir(parents=True, exist_ok=True)
        np.save(emb,Z)
        scaler = StandardScaler().fit(Z[train_idx])
        Xtr=scaler.transform(Z[train_idx])
        Xte=scaler.transform(Z[test_idx])
        if model == 'MantisV2':
            Xtr=Xtr.astype(np.float32)
            Xte=Xte.astype(np.float32)
        clf = LogisticRegression(max_iter=150, random_state=2026).fit(Xtr,labels[train_idx])
        ckpt = project/conf['checkpoint_root']/dataset/'fold0'
        ckpt.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler,ckpt/'scaler.joblib')
        joblib.dump(clf,ckpt/'logreg.joblib')
        prob=clf.predict_proba(Xte)
        pred=clf.predict(Xte) if model == 'MantisV2' else np.argmax(prob,axis=1)
        csv=project/conf['prediction_root']/dataset/'fold0'/conf['prediction_file']
        csv.parent.mkdir(parents=True,exist_ok=True)
        df=metadata.iloc[test_idx].reset_index(drop=True).copy()
        if model == 'MOMENT-1':df['global_idx']=test_idx
        df['y_true']=labels[test_idx]
        df['y_pred']=pred
        for i, c in enumerate(mod.CLASS_NAMES):df[f'prob_{c}']=prob[:,i]
        df.to_csv(csv,index=False)
    r=mod.run(project,list(mod.SPEC),[dataset],[0],trust_local_joblib=True)
    assert r['status']=='PASS' and r['passed']==2, r
    print('PASS: 2/2 synthetic downstream classifier replays; labels, order, probability values')

    # Detect corrupted predicted labels, not just matching overall accuracy.
    conf=mod.SPEC['MantisV2']
    csv=project/conf['prediction_root']/dataset/'fold0'/conf['prediction_file']
    df=pd.read_csv(csv)
    df.loc[0,'y_pred']=(int(df.loc[0,'y_pred'])+1)%4
    df.to_csv(csv,index=False)
    r=mod.run(project,['MantisV2'],[dataset],[0],trust_local_joblib=True)
    assert r['status']=='INCOMPLETE' and r['replayed_folds'][0]['predicted_label_mismatches']==1
    print('PASS: synthetic incorrect prediction detected')
    try:
        mod.run(project,['MantisV2'],[dataset],[0],trust_local_joblib=False)
    except PermissionError:
        print('PASS: trusted-joblib opt-in enforced')
    else:
        raise AssertionError('Unsafe joblib used without opt-in')
print('STAGE10 SYNTHETIC TESTS PASS')
