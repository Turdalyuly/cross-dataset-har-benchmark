#!/usr/bin/env python3
"""Strict HART checkpoint load and optional inference comparison on original X.npy.

Requires the local authors' tsfm_public/tspulse installation and original pretrained HART
checkpoint directory. Uses no training, no downloads and torch.load(weights_only=True).
For CUDA inference, matches the original bfloat16 autocast setting when supported.
"""
from __future__ import annotations
import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from stage11_common import DATASETS,load_split,test_indices,compare_saved,dump_report


def choose_device(value:str):
    if value=='auto': value='cuda' if torch.cuda.is_available() else 'cpu'
    if value.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
    return torch.device(value)


def check_one(project:Path,dataset:str,fold:int,mode:str,device:str,batch_size:int,tol:float,max_windows:int):
    import stage11_hart_original as architecture
    pretrained=project/'external_models'/'HART'/'finetuning_HART'/'pretrained_checkpoint'
    ckpt=project/'checkpoints'/'stage5g2_hart_5fold'/dataset/f'fold{fold}'/'best.pt'
    run=project/'results'/'stage5g2_hart_5fold'/dataset/f'fold{fold}'
    if not pretrained.is_dir():raise FileNotFoundError(f'Local HART pretrained base not found: {pretrained}')
    obj=torch.load(ckpt,map_location='cpu',weights_only=True)
    if not isinstance(obj,dict) or 'state_dict' not in obj:
        raise ValueError(f'{ckpt} not a HART checkpoint dict containing state_dict')
    state=obj['state_dict']
    if not isinstance(state,dict) or not all(torch.is_tensor(x) for x in state.values()):
        raise ValueError('Checkpoint state_dict is not tensor-only')
    weights=state.get('class_weights')
    if weights is None:
        cfg=obj.get('config',{})
        weights=torch.tensor(cfg['class_weights'],dtype=torch.float32) if 'class_weights' in cfg else None
    if weights is None or tuple(weights.shape)!=(4,):
        raise ValueError('Cannot recover 4 class_weights from original checkpoint')
    model=architecture.HARTForHAR(pretrained,4,weights.cpu().float())
    model.load_state_dict(state,strict=True)
    dev=choose_device(device)
    model=model.to(dev).eval()
    result={'model':'HART','dataset':dataset,'fold':fold,'mode':mode,
            'strict_checkpoint_load':'PASS','tensor_keys':len(state),
            'verification_scope':'strict HARTForHAR load; optional forward pass on original test X.npy'}
    if mode=='strict':return result
    if batch_size<=0:raise ValueError('batch_size must be > 0')
    X=np.load(project/'datasets'/'processed'/'harmonized_hart_v1'/dataset/'X.npy',mmap_mode='r')
    meta=pd.read_csv(project/'datasets'/'processed'/'harmonized_hart_v1'/dataset/'metadata.csv')
    split=load_split(project,dataset,fold)
    idx=test_indices(meta,split)
    saved=pd.read_csv(run/'test_predictions.csv')
    if len(saved)!=len(idx):raise ValueError('Prediction CSV test rows differ from original subject-disjoint test selection')
    actual_y=meta.iloc[idx]['label_id'].to_numpy(dtype=np.int64)
    if not np.array_equal(saved['y_true'].to_numpy(dtype=np.int64),actual_y):
        raise ValueError('Saved labels differ from metadata for the original test split')
    normalized=obj.get('normalization')
    if not isinstance(normalized,dict):
        p=run/'normalization.json'
        if not p.exists():raise ValueError('No checkpoint normalization or normalization.json')
        normalized=json.loads(p.read_text(encoding='utf-8'))
    mean=np.asarray(normalized['mean'],dtype=np.float32).reshape((1,3))
    std=np.asarray(normalized['std'],dtype=np.float32).reshape((1,3))
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std<=0):
        raise ValueError('Invalid train-only normalization stats')
    # Independent check that the checkpoint normalization is the same as saved preprocessing of this fold.
    if (run/'normalization.json').exists():
        previous=json.loads((run/'normalization.json').read_text(encoding='utf-8'))
        for key,v in [('mean',mean),('std',std)]:
            if not np.allclose(np.asarray(previous[key],dtype=np.float32),v.reshape(-1),rtol=0,atol=1e-7):
                raise ValueError(f'Checkpoint {key} differs from run normalization.json')
    if max_windows>0:idx=idx[:max_windows]; saved=saved.iloc[:len(idx)].copy()
    if len(idx)==0:raise ValueError('No test windows selected')
    probs=[]
    bf16=(dev.type=='cuda' and torch.cuda.is_bf16_supported())
    with torch.inference_mode():
        for start in range(0,len(idx),batch_size):
            ix=idx[start:start+batch_size]
            batch=np.asarray(X[ix],dtype=np.float32)
            batch=(batch-mean)/std
            data=torch.from_numpy(np.ascontiguousarray(batch)).to(dev)
            context=torch.autocast('cuda',dtype=torch.bfloat16) if bf16 else nullcontext()
            with context:
                out=model(past_values=data,target_values=None)
            logits=out.logits.float()
            if logits.shape!=(len(ix),4):raise ValueError(f'Unexpected HART output {tuple(logits.shape)}')
            probs.append(torch.softmax(logits,dim=-1).cpu().numpy())
    prob=np.concatenate(probs,axis=0)
    pred=np.argmax(prob,axis=1)
    window_id=meta.iloc[idx]['window_id'].to_numpy() if 'window_id' in meta else None
    result['comparison']=compare_saved(saved,pred,prob,expected_idx=idx,actual_ids=window_id,max_prob_diff=tol)
    result['autocast_bfloat16']=bool(bf16)
    result['status']=result['comparison']['status']
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',required=True)
    p.add_argument('--dataset',choices=(*DATASETS,'all'),default='wisdm')
    p.add_argument('--fold',choices=(*map(str,range(5)),'all'),default='0')
    p.add_argument('--mode',choices=['strict','replay'],default='strict')
    p.add_argument('--device',default='auto',help='cuda (recommended), cuda:0, cpu or auto')
    p.add_argument('--batch-size',type=int,default=1024,help='Original Stage 5E.3 evaluation batch size; preserves exact BF16 replay')
    p.add_argument('--prob-tolerance',type=float,default=5e-3,help='Difference tolerance accounts for GPU BF16 inference')
    p.add_argument('--max-windows',type=int,default=0,help='Replay only first N test windows; 0 means all')
    p.add_argument('--output')
    a=p.parse_args()
    project=Path(a.project_root).resolve()
    ds=DATASETS if a.dataset=='all' else [a.dataset]
    folds=range(5) if a.fold=='all' else [int(a.fold)]
    records=[];failed=[]
    for d in ds:
        for fold in folds:
            try:
                rec=check_one(project,d,fold,a.mode,a.device,a.batch_size,a.prob_tolerance,a.max_windows)
                status=rec.get('status','PASS')
                diff=rec.get('comparison',{}).get('max_probability_difference','')
                print(f'HART     {d:12s} fold{fold}: {status} | strict load PASS | Δprob={diff}',flush=True)
                if status!='PASS':failed.append(f'{d}/fold{fold}: labels/probability mismatch')
            except Exception as exc:
                rec={'model':'HART','dataset':d,'fold':fold,'mode':a.mode,'status':'ERROR',
                     'error':f'{type(exc).__name__}: {exc}'}
                failed.append(f'{d}/fold{fold}: {rec["error"]}')
                print(f'HART     {d:12s} fold{fold}: ERROR: {rec["error"]}',flush=True)
            records.append(rec)
    report={'status':'PASS' if not failed else 'INCOMPLETE','mode':a.mode,'pass_count':len(records)-len(failed),
            'total':len(records),'errors':failed,'results':records}
    dump_report(a.output,report)
    print(f'HART STAGE11: {report["status"]} ({report["pass_count"]}/{report["total"]})')
    return 0 if report['status']=='PASS' else 2

if __name__=='__main__':sys.exit(main())
