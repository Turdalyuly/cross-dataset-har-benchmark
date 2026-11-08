#!/usr/bin/env python3
"""CrossHAR adapted Stage 5L.5A/5L.5B: read-only GPU Z_test encoder replay.

From original harmonized X.npy -> train-only normalization -> saved SSL encoder
-> output_embed=True -> comparison against original representations.npz/Z_test.

Never trains or changes original project data. Uses only weights_only=True.
Requires author-side original source and checkpoints. This is not clean-room
independent reproduction, and does not reproduce SSL/classifier training.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

# Avoid creating __pycache__ inside upstream repositories during import.
sys.dont_write_bytecode = True

import numpy as np

DATASETS = ('wisdm','uci_har','mhealth','pamap2','motionsense')
ROWS = {'wisdm':52747,'uci_har':9073,'mhealth':2300,'pamap2':8164,'motionsense':24492}
EXPECTED_SOURCE_SHA256 = {
    'external_models/CrossHAR/models.py':'644af876189943023f7c28116abfecd4408b8790eca93cce07956a45e449ab13',
    'external_models/CrossHAR/utils.py':'c86ed75045874f3c282f71cfd7afcb2fbbe3f8ec81c91b4f2bf8d7e2a0742f6e',
    'external_models/CrossHAR/config.py':'a02360dc7cd31798f87f0c59cdf85d5a7c445dda6e69c6e8f952693b93d46266',
}
SCHEMA='HAR-crosshar-encoder-25fold-replay-v1'


def digest(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def artifact_paths(root:Path,dataset:str,fold:int)->tuple[Path,Path]:
    suffix=f'{dataset}/fold{fold}'
    if fold==0:
        return (root/'results/stage5l5a_crosshar_fold0'/suffix,
                root/'checkpoints/stage5l5a_crosshar_fold0'/suffix)
    return (root/'results/stage5l5b_crosshar_5fold'/suffix,
            root/'checkpoints/stage5l5b_crosshar_5fold'/suffix)


def source_proof(root:Path)->dict:
    checks={}
    for rel,expected in EXPECTED_SOURCE_SHA256.items():
        p=root/rel
        if not p.is_file(): raise FileNotFoundError('CrossHAR source missing: '+rel)
        actual=digest(p)
        checks[rel]={'sha256_matches_historical':actual==expected}
        if actual!=expected: raise RuntimeError('Source SHA256 mismatch: '+rel)
    return checks


def verify_preflight(preflight:dict)->None:
    if preflight.get('schema')!='HAR-stage13K-crosshar-encoder-preflight-v1' or preflight.get('status')!='READY_FOR_ENCODER_REPLAY_PREFLIGHT_ONLY':
        raise ValueError('Stage13K preflight status/schema invalid')
    if preflight.get('folds_checked')!=25 or preflight.get('folds_ready')!=25:
        raise ValueError('Stage13K did not pass all 25 folds')
    folds=preflight.get('folds')
    if not isinstance(folds,list) or len(folds)!=25:
        raise ValueError('Stage13K fold list incomplete')
    seen=set()
    for record in folds:
        key=(record.get('dataset'),record.get('fold'))
        if key in seen or key[0] not in DATASETS or key[1] not in range(5):
            raise ValueError('Stage13K duplicate or unknown fold')
        seen.add(key)
        if record.get('status')!='READY_FOR_ENCODER_REPLAY_PREFLIGHT_ONLY' or record.get('issues'):
            raise ValueError(f'Stage13K not ready: {key}')
        if record.get('train_norm_exact') is not True or not all(record.get('index_order_matches',{}).get(s) is True for s in ('train','val','test')):
            raise ValueError(f'Stage13K normalization/order not exact: {key}')
    if len(seen)!=25:
        raise ValueError('Stage13K fold coverage incomplete')
    for ds in DATASETS:
        cov=preflight.get('fivefold_test_coverage',{}).get(ds,{})
        if cov.get('pass') is not True or cov.get('dataset_rows')!=ROWS[ds] or cov.get('test_rows_with_zero_coverage')!=0 or cov.get('test_rows_with_multiple_coverage')!=0:
            raise ValueError('Stage13K partition invalid: '+ds)


def import_original_encoder(root:Path):
    """Match historical isolated import so CrossHAR models imports its own utils."""
    base=root/'external_models'/'CrossHAR'
    up=base/'utils.py'
    mp=base/'models.py'
    old=sys.modules.get('utils')
    try:
        spec=importlib.util.spec_from_file_location('stagefinal_crosshar_utils',up)
        if not spec or not spec.loader: raise ImportError('Cannot load CrossHAR/utils.py')
        um=importlib.util.module_from_spec(spec)
        # Some libraries expect the named module to be registered during exec.
        sys.modules[spec.name]=um
        spec.loader.exec_module(um)
        sys.modules['utils']=um
        mspec=importlib.util.spec_from_file_location('stagefinal_crosshar_models',mp)
        if not mspec or not mspec.loader: raise ImportError('Cannot load CrossHAR/models.py')
        mm=importlib.util.module_from_spec(mspec)
        sys.modules[mspec.name]=mm
        mspec.loader.exec_module(mm)
    finally:
        if old is None: sys.modules.pop('utils',None)
        else: sys.modules['utils']=old
    return mm.MaskedModel4Pretrain


def compare_arrays(replayed:np.ndarray,reference:np.ndarray,atol:float,rtol:float):
    if replayed.shape!=reference.shape or replayed.dtype!=np.float32 or reference.dtype!=np.float32:
        raise ValueError(f'Embedding shape/dtype mismatch: replay={replayed.shape}/{replayed.dtype}, reference={reference.shape}/{reference.dtype}')
    if not np.isfinite(replayed).all() or not np.isfinite(reference).all():
        raise ValueError('NaN/Inf in embeddings')
    absdiff=np.abs(replayed.astype(np.float64)-reference.astype(np.float64))
    per_row_exact=np.all(replayed==reference,axis=(1,2))
    per_row_close=np.all(absdiff<=atol+rtol*np.abs(reference.astype(np.float64)),axis=(1,2))
    return {'max_absolute_difference':float(absdiff.max(initial=0)),
            'difference_sum':float(absdiff.sum()),
            'squared_error_sum':float(np.square(absdiff).sum()),
            'exact_rows':int(per_row_exact.sum()),'allclose_rows':int(per_row_close.sum()),
            'first_mismatch_index_in_batch':int(np.where(~per_row_close)[0][0]) if not per_row_close.all() else None}


def run_fold(root:Path,ds:str,fold:int,model_cls,device,torch,batch_size:int,atol:float,rtol:float)->dict:
    result_dir, ckpt_dir=artifact_paths(root,ds,fold)
    x_path=root/'datasets/processed/harmonized_hart_v1'/ds/'X.npy'
    ref_path=result_dir/'representations.npz'
    norm_path=result_dir/'normalization.json'
    weight_path=ckpt_dir/'encoder_best.pt'
    for p in (x_path,ref_path,norm_path,weight_path):
        if not p.is_file(): raise FileNotFoundError('Missing original artifact: '+p.name)
    norm=json.loads(norm_path.read_text(encoding='utf-8'))
    if norm.get('fitted_on')!='train subjects only': raise ValueError('Normalizer not train-only')
    mean=np.asarray(norm['mean'],dtype=np.float32).reshape(1,1,3)
    std=np.asarray(norm['std'],dtype=np.float32).reshape(1,1,3)
    if np.any(std<=0) or not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError('Normalization invalid')
    X=np.load(x_path,allow_pickle=False,mmap_mode='r')
    if X.shape!=(ROWS[ds],120,3) or X.dtype!=np.float32:
        raise ValueError(f'Input dimensions wrong: {X.shape},{X.dtype}')
    with np.load(ref_path,allow_pickle=False) as arc:
        test_idx=arc['test_idx']
        if test_idx.dtype!=np.int64 or test_idx.ndim!=1 or not len(test_idx) or np.any(test_idx<0) or np.any(test_idx>=len(X)) or len(np.unique(test_idx))!=len(test_idx):
            raise ValueError('Invalid test indices')
        Z_ref=arc['Z_test'] # one fold in memory; no modifications
        if Z_ref.shape!=(len(test_idx),120,72) or Z_ref.dtype!=np.float32:
            raise ValueError(f'Bad Z_test shape/dtype {Z_ref.shape}/{Z_ref.dtype}')

    cfg=SimpleNamespace(feature_num=3,hidden=72,hidden_ff=144,n_layers=1,n_heads=4,seq_len=120,emb_norm=True)
    net=model_cls(cfg,output_embed=False).to(device)
    # Trust only project-author original weights. Do not execute untrusted checkpoints.
    weights=torch.load(weight_path,map_location='cpu',weights_only=True)
    net.load_state_dict(weights,strict=True)
    del weights
    net.eval()
    net.output_embed=True
    count=exact=close=0
    maxdiff=diffsum=sqsum=0.0
    first_bad=None
    with torch.inference_mode():
        for start in range(0,len(test_idx),batch_size):
            ids=test_idx[start:start+batch_size]
            # Match historical SequenceDataset.__getitem__: float32 then (x - mean) / std.
            arr=np.asarray(X[ids],dtype=np.float32)
            arr=np.ascontiguousarray((arr-mean)/std,dtype=np.float32)
            z=net(torch.from_numpy(arr).to(device)).float().cpu().numpy()
            ref_batch=Z_ref[start:start+len(ids)]
            stats=compare_arrays(z,ref_batch,atol,rtol)
            count+=len(ids)
            exact+=stats['exact_rows'];close+=stats['allclose_rows']
            maxdiff=max(maxdiff,stats['max_absolute_difference'])
            diffsum+=stats['difference_sum'];sqsum+=stats['squared_error_sum']
            if stats['first_mismatch_index_in_batch'] is not None and first_bad is None:
                first_bad=start+stats['first_mismatch_index_in_batch']
                # Stop immediately; retain partial result and avoid wasted GPU.
                break
    del Z_ref, net, X
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    total_values=count*120*72
    result={'dataset':ds,'fold':fold,'status':'NUMERICAL_MATCH_FULL_FOLD' if count==len(test_idx) and close==len(test_idx) else 'NUMERICAL_MISMATCH_OR_INCOMPLETE',
            'rows_compared':count,'expected_test_rows':int(len(test_idx)),
            'features_per_row':8640,'values_compared':total_values,
            'exact_rows':exact,'allclose_rows':close,
            'exact_equal':bool(count==len(test_idx) and exact==len(test_idx)),
            'allclose':bool(count==len(test_idx) and close==len(test_idx)),
            'max_absolute_difference':maxdiff,
            'mean_absolute_difference':diffsum/total_values if total_values else None,
            'rmse':(sqsum/total_values)**.5 if total_values else None,
            'first_tolerance_mismatch_row_zero_based':first_bad,
            'reference_file':'representations.npz/Z_test',
            'checkpoint':'encoder_best.pt'}
    return result


def write_report(report:dict,out:Path):
    out.parent.mkdir(parents=True,exist_ok=True)
    # Always write only the passed output path, outside the original project.
    tmp=out.with_name(out.name+'.partial')
    tmp.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    os.replace(tmp,out)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root',type=Path,required=True)
    ap.add_argument('--preflight',type=Path,required=True,help='HAR_STAGE13K_CROSSHAR_PREFLIGHT_ALL25.json')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--batch-size',type=int,default=512)
    ap.add_argument('--atol',type=float,default=1e-5)
    ap.add_argument('--rtol',type=float,default=1e-4)
    ap.add_argument('--only-fold',type=str,default=None,help='Optional test: wisdm:0. Default ALL 25.')
    args=ap.parse_args()
    root=args.project_root.resolve(); out=args.output.resolve()
    if out==root or root in out.parents: ap.error('--output must be outside original project tree')
    if not 1<=args.batch_size<=512: ap.error('--batch-size must be 1..512; use 512 to reproduce historical batches')
    if args.atol<0 or args.rtol<0: ap.error('Tolerances must be nonnegative')
    report={'schema':SCHEMA,'status':'IN_PROGRESS','generated_utc':datetime.now(timezone.utc).isoformat(),
        'claim_scope':'author-side saved-checkpoint CrossHAR SSL encoder -> original Z_test; no training; not clean-room reproduction',
        'training_performed':False,'original_project_files_modified':False,'independent_clean_machine_reproduction':False,
        'historical_encoder_cfg':{'feature_num':3,'hidden':72,'hidden_ff':144,'n_layers':1,'n_heads':4,'seq_len':120,'emb_norm':True},
        'historical_extraction':{'batch_size':512,'output_embed':True,'shuffle':False,'normalization':'float32 train-only per fold','autocast':False},
        'numerical_tolerance':{'atol':args.atol,'rtol':args.rtol},'folds':[]}
    try:
        source_proof(root)
        p=json.loads(args.preflight.read_text(encoding='utf-8'))
        verify_preflight(p)
        report['preflight_sha256']=digest(args.preflight)
        report['source_sha256_checked']=True
        import torch
        if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU required for historical inference')
        if os.environ.get('CUDA_VISIBLE_DEVICES') is None:
            raise RuntimeError('Set CUDA_VISIBLE_DEVICES=1 (historical author-server GPU)')
        device=torch.device('cuda:0')
        report['environment']={'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),
                               'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES')}
        model_cls=import_original_encoder(root)
        targets=[(d,f) for d in DATASETS for f in range(5)]
        if args.only_fold:
            try:
                ds,fold_s=args.only_fold.split(':')
                fold=int(fold_s)
                if (ds,fold) not in targets: raise ValueError()
            except ValueError: raise ValueError('--only-fold format wisdm:0, valid dataset and fold 0..4')
            targets=[(ds,fold)]
        report['scope_folds_expected']=len(targets)
        for i,(ds,fold) in enumerate(targets,1):
            print(f'[{i}/{len(targets)}] {ds} fold{fold}',flush=True)
            record=run_fold(root,ds,fold,model_cls,device,torch,args.batch_size,args.atol,args.rtol)
            report['folds'].append(record)
            report['folds_completed']=len(report['folds'])
            write_report(report,out)
            print('    '+record['status']+f"; rows={record['rows_compared']}; max_abs={record['max_absolute_difference']:.9g}",flush=True)
            if record['status']!='NUMERICAL_MATCH_FULL_FOLD':
                raise RuntimeError(f'Numerical comparison failed at {ds}/fold{fold}; see output report')
        report['total_test_rows_compared']=sum(v['rows_compared'] for v in report['folds'])
        report['total_values_compared']=sum(v['values_compared'] for v in report['folds'])
        report['folds_pass']=len(report['folds'])
        report['max_absolute_difference']=max(v['max_absolute_difference'] for v in report['folds'])
        report['all_folds_exact_equal']=all(v['exact_equal'] for v in report['folds'])
        report['status']=('NUMERICAL_MATCH_ALL_25_FOLDS' if len(targets)==25 and
                          report['total_test_rows_compared']==96776 else 'NUMERICAL_MATCH_SELECTED_FOLDS')
        write_report(report,out)
        print('CROSSHAR:',report['status'],flush=True)
    except Exception as e:
        report['status']='FAILED_OR_BLOCKED'
        report['failure']=f'{type(e).__name__}: {str(e)[:350]}'
        write_report(report,out)
        print('CROSSHAR:',report['status'],'—',report['failure'],file=sys.stderr)
        raise SystemExit(1)

if __name__=='__main__':main()
