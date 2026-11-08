#!/usr/bin/env python3
"""Strict CrossHAR 3-checkpoint load and classifier replay on saved representations.

No training, no downloads, no replacement of original artifacts. Requires the local
original CrossHAR source; loading its Python modules executes third-party Python code.
All .pt checkpoints load with torch.load(weights_only=True) only.
"""
from __future__ import annotations
import argparse
import importlib
import json
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from stage11_common import DATASETS,load_split,test_indices,compare_saved,dump_report

ENCODER_CFG=SimpleNamespace(feature_num=3,hidden=72,hidden_ff=144,n_layers=1,n_heads=4,seq_len=120,emb_norm=True)
CLASSIFIER_CFG=SimpleNamespace(input_size=72,hidden_size=100,num_layers=1,num_heads=4,dropout_ratio=0.1,dim_feedforward=2048)


def scoped_crosshar_import(root:Path):
    """Reproduce the original notebook's generic-name import fix, check origins."""
    root=root.resolve()
    if not root.is_dir():raise FileNotFoundError(f'Original CrossHAR repository not found: {root}')
    names=('config','augmentations','utils','models','Contrastive')
    saved={k:sys.modules.pop(k) for k in names if k in sys.modules}
    oldpath=list(sys.path)
    try:
        sys.path.insert(0,str(root))
        importlib.invalidate_caches()
        modules={k:importlib.import_module(k) for k in names}
        for key,m in modules.items():
            source=Path(getattr(m,'__file__','/missing')).resolve()
            if root not in source.parents:raise RuntimeError(f'CrossHAR module {key} loaded from {source}, expected inside {root}')
        return modules['models'].MaskedModel4Pretrain, modules['models'].Transformer_ft, modules['Contrastive'].Contrastive
    finally:
        sys.path[:]=oldpath
        for k in names: sys.modules.pop(k,None)
        sys.modules.update(saved)


def select_device(requested):
    if requested=='auto':requested='cuda' if torch.cuda.is_available() else 'cpu'
    if requested.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
    return torch.device(requested)


def check_one(project:Path,dataset:str,fold:int,mode:str,device:str,batch_size:int,tol:float):
    modelroot=project/'external_models'/'CrossHAR'
    ckpt=project/'checkpoints'/'stage5l5b_crosshar_5fold'/dataset/f'fold{fold}'
    run=project/'results'/'stage5l5b_crosshar_5fold'/dataset/f'fold{fold}'
    enc_type,clf_type,con_type=scoped_crosshar_import(modelroot)
    dev=select_device(device)
    models={'encoder':enc_type(ENCODER_CFG,output_embed=False),
            'contrastive':con_type(),
            'classifier':clf_type(CLASSIFIER_CFG,output_size=4)}
    files={'encoder':'encoder_best.pt','contrastive':'contrastive_best.pt','classifier':'classifier_best.pt'}
    keys={}
    for name,model in models.items():
        path=ckpt/files[name]
        state=torch.load(path,map_location='cpu',weights_only=True)
        if not isinstance(state,dict) or not all(torch.is_tensor(v) for v in state.values()):
            raise ValueError(f'{path} not a raw tensor-only state_dict')
        model.load_state_dict(state,strict=True)
        model.to(dev).eval()
        keys[name]=len(state)
    result={'model':'CrossHAR-adapted','dataset':dataset,'fold':fold,'mode':mode,
            'strict_checkpoint_load':'PASS','tensor_keys':keys,
            'verification_scope':'3 checkpoints strict state_dict load; optional classifier-only replay from retained representations'}
    if mode=='strict':return result
    if batch_size<=0:raise ValueError('batch_size must be positive')
    repfile=run/'representations.npz'
    with np.load(repfile,allow_pickle=False) as reps:
        if not {'Z_test','test_idx','y_test'}.issubset(reps.files):
            # Original Stage 5L.5 saves only indices + Z arrays; y_test is in metadata.csv.
            if not {'Z_test','test_idx'}.issubset(reps.files):
                raise ValueError(f'{repfile} missing Z_test/test_idx')
        z=np.asarray(reps['Z_test'],dtype=np.float32)
        indices=np.asarray(reps['test_idx'],dtype=np.int64)
    if len(z)!=len(indices):raise ValueError('Representations and test indices length mismatch')
    meta=pd.read_csv(project/'datasets'/'processed'/'harmonized_hart_v1'/dataset/'metadata.csv')
    split=load_split(project,dataset,fold)
    expected=test_indices(meta,split)
    if not np.array_equal(indices,expected):
        raise ValueError(f'Representation test indices disagree with subject-disjoint split for {dataset}/fold{fold}')
    saved=pd.read_csv(run/'predictions.csv')
    if not np.array_equal(saved['y_true'].to_numpy(dtype=np.int64),meta.iloc[expected]['label_id'].to_numpy(dtype=np.int64)):
        raise ValueError('Saved ground-truth labels differ from metadata for selected test subjects')
    allp=[]
    with torch.inference_mode():
        for start in range(0,len(z),batch_size):
            inp=torch.from_numpy(z[start:start+batch_size].copy()).to(dev)
            logits=models['classifier'](inp)
            if isinstance(logits,(tuple,list)):logits=logits[0]
            if logits.shape != (len(inp),4):raise ValueError(f'Classifier logits shape {tuple(logits.shape)}')
            allp.append(torch.softmax(logits.float(),dim=1).cpu().numpy())
    prob=np.concatenate(allp,axis=0)
    pred=np.argmax(prob,axis=1)
    result['comparison']=compare_saved(saved,pred,prob,expected_idx=expected,max_prob_diff=tol)
    result['status']=result['comparison']['status']
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',required=True)
    p.add_argument('--dataset',choices=(*DATASETS,'all'),default='wisdm')
    p.add_argument('--fold',choices=(*map(str,range(5)),'all'),default='0')
    p.add_argument('--mode',choices=['strict','replay'],default='replay')
    p.add_argument('--device',default='cpu',help='cpu (default), cuda, cuda:0, or auto')
    p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--prob-tolerance',type=float,default=1e-4)
    p.add_argument('--output')
    a=p.parse_args()
    project=Path(a.project_root).resolve()
    ds=DATASETS if a.dataset=='all' else [a.dataset]
    folds=range(5) if a.fold=='all' else [int(a.fold)]
    records=[];failed=[]
    for d in ds:
        for fold in folds:
            try:
                rec=check_one(project,d,fold,a.mode,a.device,a.batch_size,a.prob_tolerance)
                status=rec.get('status','PASS')
                diff=rec.get('comparison',{}).get('max_probability_difference','')
                print(f'CrossHAR {d:12s} fold{fold}: {status} | strict 3/3 | Δprob={diff}',flush=True)
                if status!='PASS':failed.append(f'{d}/fold{fold}: probability/label mismatch')
            except Exception as exc:
                rec={'model':'CrossHAR-adapted','dataset':d,'fold':fold,'mode':a.mode,'status':'ERROR','error':f'{type(exc).__name__}: {exc}'}
                failed.append(f'{d}/fold{fold}: {rec["error"]}')
                print(f'CrossHAR {d:12s} fold{fold}: ERROR: {rec["error"]}',flush=True)
            records.append(rec)
    report={'status':'PASS' if not failed else 'INCOMPLETE','mode':a.mode,'pass_count':len(records)-len(failed),
            'total':len(records),'errors':failed,'results':records}
    dump_report(a.output,report)
    print(f'CROSSHAR STAGE11: {report["status"]} ({report["pass_count"]}/{report["total"]})')
    return 0 if report['status']=='PASS' else 2

if __name__=='__main__':sys.exit(main())
