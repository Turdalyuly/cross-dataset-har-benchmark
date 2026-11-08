#!/usr/bin/env python3
"""One-fold clean TimesNet/PatchTST/TSLANet reproduction; NO TRAINING unless --train.

Reconstructed from corrected Stage 5L.2/5L.3; external original model implementations required.
The published results were produced by historical Stage 5L.2 (fold 0) + Stage 5L.3
(folds 1–4). This runner preserves their mathematical protocol, but is not claimed
bitwise-equivalent on other machines (nondeterministic backend, worker scheduling).
"""
import argparse
import csv
import json
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from sklearn.metrics import f1_score
except ImportError as e:
    raise SystemExit(f"Missing dependency: {e}. pip install -r requirements-supervised.txt")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.foundation_baselines import build_model

DATASETS = ('wisdm','uci_har','mhealth','pamap2','motionsense')
MODELS = ('timesnet','patchtst','tslanet')


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalization(X, indices, chunk_size=5000):
    # Same float64 streaming statistics as final Stage 5G.3.
    total=np.zeros(3,dtype=np.float64)
    squares=np.zeros(3,dtype=np.float64)
    count=0
    for start in range(0,len(indices),chunk_size):
        part=np.asarray(X[indices[start:start+chunk_size]], dtype=np.float64)
        total+=part.sum(axis=(0,1))
        squares+=np.square(part).sum(axis=(0,1))
        count+=part.shape[0]*part.shape[1]
    if count==0:
        raise ValueError('Empty training set')
    mean=total/count
    sd=np.sqrt(np.maximum(squares/count-mean**2,1e-12))
    return mean.astype('float32'), sd.astype('float32')


class WindowDataset(Dataset):
    def __init__(self, X, y, indices, mean, sd):
        self.X=X
        self.y=y
        self.indices=np.asarray(indices,dtype=np.int64)
        self.mean=np.asarray(mean,dtype=np.float32).reshape(1,3)
        self.sd=np.asarray(sd,dtype=np.float32).reshape(1,3)
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        idx=int(self.indices[i])
        x=np.array(self.X[idx],dtype=np.float32,copy=True)
        x=((x-self.mean)/self.sd).copy()
        return torch.from_numpy(x), torch.tensor(int(self.y[idx]),dtype=torch.long), idx


def prepare_data(project_root, dataset, fold, data_root=None, split_root=None):
    project_root=Path(project_root)
    data=Path(data_root) if data_root else project_root/'datasets/processed/harmonized_hart_v1'
    splits=Path(split_root) if split_root else project_root/'splits/harmonized_hart_v1'
    X=np.load(data/dataset/'X.npy',mmap_mode='r')
    meta=pd.read_csv(data/dataset/'metadata.csv')
    assert X.ndim==3 and X.shape[1:]==(120,3) and len(X)==len(meta), (X.shape,len(meta))
    assert {'label_id','subject'}.issubset(meta.columns), meta.columns.tolist()
    y=meta['label_id'].to_numpy(dtype=np.int64)
    subjects=meta['subject'].to_numpy(dtype=np.int64)
    assert set(y)=={0,1,2,3},set(y)
    obj=json.loads((splits/f'{dataset}_5fold.json').read_text())
    entry=next((r for r in obj['folds'] if r['fold']==fold),None)
    if entry is None: raise ValueError(f'Missing fold {fold}')
    train=set(entry['train_subjects']); val=set(entry['val_subjects']); test=set(entry['test_subjects'])
    if train & val or train & test or val & test: raise ValueError('Subject leakage')
    if train|val|test != set(subjects):raise ValueError('Subjects do not partition the dataset')
    ix={k:np.flatnonzero(np.isin(subjects,list(v))) for k,v in [('train',train),('val',val),('test',test)]}
    if any(len(v)==0 for v in ix.values()):raise ValueError('Empty train/val/test indices')
    mean,sd=normalization(X,ix['train'])
    counts=np.bincount(y[ix['train']],minlength=4)
    weights=(counts.sum()/(4*np.maximum(counts,1))).astype('float32')
    return X,y,ix,mean,sd,counts,weights


def eval_loader(model,loader,device):
    model.eval()
    truth=[]; pred=[]; ix=[]; prob=[]
    with torch.inference_mode():
        for x,y,ids in loader:
            ctx=torch.autocast(device_type='cuda',dtype=torch.bfloat16) if device.startswith('cuda') else nullcontext()
            with ctx:
                logits=model(x.to(device))
            probabilities=torch.softmax(logits.float(),dim=1).cpu().numpy()
            truth.extend(y.numpy().tolist()); pred.extend(probabilities.argmax(1).tolist())
            prob.extend(probabilities.tolist()); ix.extend(ids.numpy().tolist())
    return {'macro_f1':float(f1_score(truth,pred,average='macro',zero_division=0)),
            'y_true':np.array(truth),'y_pred':np.array(pred),'idx':np.array(ix),'probs':np.array(prob)}


def run(args):
    set_seed(args.seed)
    X,y,ix,mean,sd,counts,weights=prepare_data(args.project_root,args.dataset,args.fold,args.data_root,args.split_root)
    sizes={k:len(v) for k,v in ix.items()}
    print('DATA PASS',args.dataset, 'fold',args.fold, 'sizes',sizes, 'class counts',counts.tolist())
    model=build_model(args.model,args.project_root)
    params=sum(x.numel() for x in model.parameters())
    model.eval()
    with torch.inference_mode():
        out=model(torch.randn(2,120,3))
    assert out.shape==(2,4)
    print('MODEL PASS',args.model,'parameters',params,'logits',(2,4))
    if not args.train:
        print('DRY RUN PASS. NO TRAINING PERFORMED.')
        return
    device='cuda' if args.device=='auto' and torch.cuda.is_available() else ('cpu' if args.device=='auto' else args.device)
    if device.startswith('cuda') and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    folder=Path(args.output_root)/args.dataset/args.model/f'fold{args.fold}'
    if folder.exists():raise FileExistsError(f'Output exists; refusing overwrite: {folder}')
    folder.mkdir(parents=True)
    model=model.to(device)
    kwargs={'batch_size':args.batch_size,'num_workers':args.workers,'pin_memory':device.startswith('cuda'),'persistent_workers':args.workers>0}
    loaders={name:DataLoader(WindowDataset(X,y,ids,mean,sd),shuffle=(name=='train'),
           generator=(torch.Generator().manual_seed(args.seed) if name=='train' else None),**kwargs)
           for name,ids in ix.items()}
    criterion=nn.CrossEntropyLoss(weight=torch.tensor(weights,device=device))
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',factor=0.5,patience=3,min_lr=1e-5)
    best_f1=-1.0; bad=0; best_epoch=None; rows=[]
    config={'dataset':args.dataset,'model':args.model,'fold':args.fold,'seed':args.seed,
            'batch_size':args.batch_size,'max_epochs':args.epochs,'patience':args.patience,
            'learning_rate':args.lr,'weight_decay':args.weight_decay,'parameters':params,
            'train_subjects':sizes['train'],'val_subjects':sizes['val'],'test_subjects':sizes['test'],
            'train_class_counts':counts.tolist(),'train_class_weights':weights.tolist(),
            'normalization_mean':mean.tolist(),'normalization_std':sd.tolist(),
            'worker_count':args.workers,'device':device,'source_note':'Reconstructed Stage5L.2/5L.3 runner; model-specific upstream sources required'}
    (folder/'config.json').write_text(json.dumps(config,indent=2))
    t0=time.perf_counter()
    for epoch in range(1,args.epochs+1):
        model.train(); losses=[]
        for batch_i,(x,t,_) in enumerate(loaders['train']):
            x=x.to(device);t=t.to(device)
            opt.zero_grad(set_to_none=True)
            if device.startswith('cuda'):
                ctx=torch.autocast(device_type='cuda',dtype=torch.bfloat16)
            else: ctx=nullcontext()
            with ctx:
                loss=criterion(model(x),t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.)
            opt.step();losses.append(float(loss.detach().item()))
            if args.max_batches and batch_i+1>=args.max_batches:break
        val=eval_loader(model,loaders['val'],device)
        val_f1=val['macro_f1'];scheduler.step(val_f1)
        improved=val_f1>best_f1+1e-5
        if improved:
            best_f1=val_f1;best_epoch=epoch;bad=0
            torch.save({'state_dict':model.state_dict(),'mean':mean.tolist(),'std':sd.tolist(),'config':config,
                        'epoch':epoch,'val_macro_f1':best_f1},folder/'best.pt')
        else:bad+=1
        rows.append({'epoch':epoch,'train_loss':float(np.mean(losses)),
                     'val_macro_f1':val_f1,'lr':opt.param_groups[0]['lr'],'best':improved})
        print(f'Epoch {epoch}: val Macro-F1={val_f1:.6f}'+' *BEST*' if improved else f'Epoch {epoch}: val Macro-F1={val_f1:.6f}',flush=True)
        if bad>=args.patience:break
    pd.DataFrame(rows).to_csv(folder/'history.csv',index=False)
    ckpt=torch.load(folder/'best.pt',map_location=device,weights_only=True)
    model.load_state_dict(ckpt['state_dict'],strict=True)
    test=eval_loader(model,loaders['test'],device)
    df=pd.DataFrame({'index':test['idx'],'y_true':test['y_true'],'y_pred':test['y_pred']})
    for i,c in enumerate(['walking','sitting','standing','stairs']):df[f'prob_{c}']=test['probs'][:,i]
    df.to_csv(folder/'test_predictions.csv',index=False)
    metrics={'macro_f1':test['macro_f1'],'best_epoch':best_epoch,'best_val_macro_f1':best_f1,
             'n_test':len(test['idx']),'duration_sec':time.perf_counter()-t0}
    (folder/'metrics.json').write_text(json.dumps(metrics,indent=2))
    print('TEST',metrics,'OUTPUT',folder)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root',required=True)
    ap.add_argument('--dataset',choices=DATASETS,required=True)
    ap.add_argument('--model',choices=MODELS,required=True)
    ap.add_argument('--fold',type=int,choices=range(5),required=True)
    ap.add_argument('--data-root',default=None)
    ap.add_argument('--split-root',default=None)
    ap.add_argument('--device',default='auto')
    ap.add_argument('--train',action='store_true',help='Explicitly allow training; absent => dry-run only')
    ap.add_argument('--output-root',default=str(Path.cwd()/'runs'/'new_models_reproduction'))
    ap.add_argument('--epochs',type=int,default=60)
    ap.add_argument('--patience',type=int,default=10)
    ap.add_argument('--batch-size',type=int,default=512)
    ap.add_argument('--workers',type=int,default=0)
    ap.add_argument('--seed',type=int,default=2026)
    ap.add_argument('--lr',type=float,default=1e-3)
    ap.add_argument('--weight-decay',type=float,default=1e-4)
    ap.add_argument('--max-batches',type=int,default=0,help='Non-paper smoke test limit only')
    a=ap.parse_args()
    if a.epochs<1 or a.batch_size<1:ap.error('epochs and batch-size must be >= 1')
    run(a)
