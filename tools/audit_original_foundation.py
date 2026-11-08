#!/usr/bin/env python3
"""Read-only audit of Stage5L3 TimesNet/PatchTST/TSLANet, 75 runs.

- Recomputes test Macro-F1 from Stage5L.3 predictions.csv, compares to 250-row CSV.
- Inspects sample/all original checkpoint with narrowly restricted weights_only=True.
- Loads exact original architecture from external_models and strict state dict.
- DOES NOT retrain, change datasets, or overwrite source experiments.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
MODELS=('timesnet','patchtst','tslanet')
DATASETS=('wisdm','uci_har','mhealth','pamap2','motionsense')


def _load_checkpoint(path):
    import torch
    from tools.safe_numpy_checkpoint import _restricted_numpy_globals
    allow = _restricted_numpy_globals(path)
    with torch.serialization.safe_globals(allow):
        obj=torch.load(str(path),map_location='cpu',weights_only=True)
    if not isinstance(obj,dict) or not isinstance(obj.get('state_dict'),dict):
        raise ValueError('Checkpoint lacks state_dict dictionary')
    return obj


def _check_checkpoint(path,project_root,model,dataset,fold):
    import torch
    from models.foundation_baselines import build_model
    obj=_load_checkpoint(path)
    if str(obj.get('model_name','')).lower()!=model:
        raise ValueError('Checkpoint model_name mismatch')
    if str(obj.get('dataset','')).lower()!=dataset or int(obj.get('fold',-1))!=fold:
        raise ValueError('Checkpoint dataset/fold mismatch')
    norm=obj.get('normalization',{})
    for field in ('mean','std'):
        a=np.asarray(norm.get(field,[]),dtype='float64')
        if a.shape!=(3,) or not np.isfinite(a).all() or (field=='std' and not (a>0).all()):
            raise ValueError(f'Invalid normalization.{field}: {a}')
    net=build_model(model,project_root)
    net.load_state_dict(obj['state_dict'],strict=True)
    net.eval()
    with torch.inference_mode():
        output=net(torch.randn(2,120,3,dtype=torch.float32))
    if tuple(output.shape)!=(2,4) or not torch.isfinite(output).all():
        raise ValueError(f'Output shape or numerics invalid: {output.shape}')
    return {'param_count':sum(p.numel() for p in net.parameters()),
            'best_epoch':obj.get('best_epoch')}


def run(project_root, public_csv, mode):
    root=Path(project_root)
    df=pd.read_csv(public_csv)
    df['m']=df.model.astype(str).str.lower()
    df['d']=df.dataset.astype(str).str.lower()
    filtered=df[df.m.isin(MODELS)]
    if len(filtered)!=75:
        raise ValueError(f'Expected exactly 75 rows for 3 models; found {len(filtered)}')
    index={(r.m,r.d,int(r.fold)):float(r.macro_f1) for r in filtered.itertuples()}
    if len(index)!=75:raise ValueError('Duplicate model/dataset/fold in 250-row CSV')
    report={'expected_runs':75,'metrics_match':0,'predictions_match':0,'checkpoint_checked':0,
       'checkpoint_pass':0,'missing':[],'mismatches':[],'warnings':[],'runs':[]}
    checked=set()
    for model in MODELS:
      for dataset in DATASETS:
       for fold in range(5):
        rel=Path('results/stage5l3_top10_newmodels_5fold')/model/dataset/f'fold{fold}'
        directory=root/rel
        row={'model':model,'dataset':dataset,'fold':fold}
        for f in ('COMPLETE','metrics.json','predictions.csv'):
            if not (directory/f).is_file():report['missing'].append(str(rel/f))
        if all((directory/f).is_file() for f in ('COMPLETE','metrics.json','predictions.csv')):
            try:
                meta=json.loads((directory/'metrics.json').read_text())
                score=float(meta.get('test_macro_f1',meta.get('macro_f1',float('nan'))))
                exp=index[model,dataset,fold]
                if np.isfinite(score) and abs(score-exp)<=2e-4:report['metrics_match']+=1
                else:report['mismatches'].append(f'{rel}: metrics score {score}, published {exp}')
                preds=pd.read_csv(directory/'predictions.csv',usecols=['y_true','y_pred'])
                y_true=preds.y_true.astype(int).to_numpy();y_pred=preds.y_pred.astype(int).to_numpy()
                actual=float(f1_score(y_true,y_pred,labels=[0,1,2,3],average='macro',zero_division=0))
                if abs(actual-exp)<=2e-4:report['predictions_match']+=1
                else:report['mismatches'].append(f'{rel}: predictions score {actual}, published {exp}')
                row.update(saved=score,calculated=actual,expected=exp,num_predictions=len(preds))
            except Exception as exc:
                report['mismatches'].append(f'{rel}: {type(exc).__name__}: {exc}')
        inspect=(mode=='all' or mode=='sample' and model not in checked)
        if inspect:
            checked.add(model)
            report['checkpoint_checked']+=1
            ckpt=root/'checkpoints/stage5l3_top10_newmodels_5fold'/model/dataset/f'fold{fold}'/'best.pt'
            if not ckpt.is_file():report['missing'].append(str(ckpt))
            else:
                try:
                    row['checkpoint']=_check_checkpoint(ckpt,root,model,dataset,fold)
                    report['checkpoint_pass']+=1
                except Exception as exc:
                    report['mismatches'].append(f'{ckpt}: {type(exc).__name__}: {exc}')
        report['runs'].append(row)
    report['status']='PASS' if(report['metrics_match']==75 and report['predictions_match']==75 and
       report['checkpoint_pass']==report['checkpoint_checked'] and not report['missing'] and
       not report['mismatches']) else 'INCOMPLETE'
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',required=True)
    p.add_argument('--public-csv',default=str(BASE/'results/clean_250_fold_scores.csv'))
    p.add_argument('--check-checkpoints',choices=('none','sample','all'),default='sample')
    p.add_argument('--output',default=None)
    args=p.parse_args()
    result=run(args.project_root,args.public_csv,args.check_checkpoints)
    print('Metrics against Table 2:',result['metrics_match'],'/75')
    print('Predictions recalculated:',result['predictions_match'],'/75')
    print('Checkpoint compatibility:',result['checkpoint_pass'],'/',result['checkpoint_checked'])
    print('Missing:',len(result['missing']),'Mismatches:',len(result['mismatches']))
    for x in (result['missing']+result['mismatches']+result['warnings'])[:20]:print(' ',x)
    print('AUDIT:',result['status'])
    if args.output:
        dest=Path(args.output);dest.parent.mkdir(exist_ok=True,parents=True)
        dest.write_text(json.dumps(result,ensure_ascii=False,indent=2))
        print('SAVED:',dest)
    if result['status']!='PASS':sys.exit(2)

if __name__=='__main__':main()
