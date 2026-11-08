#!/usr/bin/env python3
"""Read-only audit of 75 original CNN1D/ResNet1D/TCN evaluation folders.

Does not run training or write inside the original project. Optional checkpoint
inspection uses torch.load(weights_only=True) for model state-dict verification.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
MODELS=('cnn1d','resnet1d','tcn')
DATASETS=('wisdm','uci_har','mhealth','pamap2','motionsense')


def audit(root, public_csv, checkpoint_mode='sample'):
    root=Path(root)
    df=pd.read_csv(public_csv)
    df['norm_model']=df['model'].str.lower()
    df['norm_dataset']=df['dataset'].str.lower()
    df=df.loc[df.norm_model.isin(MODELS)]
    if len(df)!=75:
        raise ValueError(f'Expected 75 supervised rows; found {len(df)}')
    expected={(r.norm_model,r.norm_dataset,int(r.fold)):float(r.macro_f1) for r in df.itertuples()}
    report={'expected_runs':75, 'metrics_match':0,'predictions_match':0,
            'checkpoint_checked':0,'checkpoint_pass':0,'missing':[], 'mismatches':[], 'warnings':[], 'runs':[]}
    tested_models=set()
    for dataset in DATASETS:
        for model in MODELS:
            for fold in range(5):
                rel=Path('results/stage5g3_baselines_hart_5fold')/dataset/model/f'fold{fold}'
                rundir=root/rel
                row={'dataset':dataset,'model':model,'fold':fold}
                for name in ('COMPLETE','metrics.json','test_predictions.csv'):
                    if not (rundir/name).exists():
                        report['missing'].append(f'{rel}/{name}')
                if not all((rundir/name).exists() for name in ('COMPLETE','metrics.json','test_predictions.csv')):
                    report['runs'].append(row)
                    continue
                try:
                    meta=json.loads((rundir/'metrics.json').read_text())
                    value=float(meta.get('test_macro_f1',meta.get('macro_f1',float('nan'))))
                    target=expected[(model,dataset,fold)]
                    if np.isfinite(value) and abs(value-target)<=2e-4:report['metrics_match']+=1
                    else:report['mismatches'].append(f'{rel}: metrics.json Macro-F1 {value}, expected {target}')
                    pred=pd.read_csv(rundir/'test_predictions.csv',usecols=['y_true','y_pred'])
                    calculated=float(f1_score(pred.y_true,pred.y_pred,average='macro',zero_division=0))
                    if abs(calculated-target)<=2e-4:report['predictions_match']+=1
                    else:report['mismatches'].append(f'{rel}: calculated Macro-F1 {calculated}, expected {target}')
                    row.update(saved=value,calculated=calculated,expected=target,rows=len(pred))
                except Exception as e:
                    report['mismatches'].append(f'{rel}: {type(e).__name__}: {e}')
                check=(checkpoint_mode=='all' or (checkpoint_mode=='sample' and model not in tested_models))
                if check:
                    tested_models.add(model)
                    report['checkpoint_checked']+=1
                    ckpt=root/'checkpoints/stage5g3_baselines_hart_5fold'/dataset/model/f'fold{fold}_best.pt'
                    if not ckpt.is_file():
                        report['warnings'].append(f'Checkpoint missing: {ckpt}')
                    else:
                        try:
                            import torch
                            from models.supervised_baselines import build_model
                            from tools.safe_numpy_checkpoint import load_numpy_metadata_checkpoint
                            obj=load_numpy_metadata_checkpoint(ckpt)
                            state=obj['state_dict']
                            if str(obj['dataset']).lower()!=dataset or str(obj['model']).lower()!=model or int(obj['fold'])!=fold:
                                raise ValueError('Checkpoint dataset/model/fold metadata mismatch')
                            for name in ('mean','std'):
                                arr=np.asarray(obj[name])
                                if arr.shape!=(3,) or not np.isfinite(arr).all():
                                    raise ValueError(f'Invalid checkpoint {name}: shape={arr.shape}')
                                if name=='std' and not (arr>0).all():
                                    raise ValueError('Checkpoint standard deviations must be positive')
                            net=build_model(model)
                            net.load_state_dict(state,strict=True)
                            net.eval()
                            with torch.inference_mode():logits=net(torch.randn(1,3,120))
                            if logits.shape!=(1,4):raise ValueError('Logits shape mismatch')
                            report['checkpoint_pass']+=1
                        except Exception as e:
                            report['mismatches'].append(f'{ckpt}: {type(e).__name__}: {e}')
                report['runs'].append(row)
    report['status']='PASS' if (report['metrics_match']==75 and report['predictions_match']==75 and not report['missing'] and not report['mismatches']) else 'INCOMPLETE'
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',required=True)
    p.add_argument('--public-csv',default=str(BASE/'results/clean_250_fold_scores.csv'))
    p.add_argument('--check-checkpoints',default='sample',choices=('none','sample','all'))
    p.add_argument('--output',default=None)
    args=p.parse_args()
    report=audit(args.project_root,args.public_csv,args.check_checkpoints)
    print('Metrics against Table 2:',report['metrics_match'],'/75')
    print('Predictions recalculated:',report['predictions_match'],'/75')
    print('Checkpoint compatibility:',report['checkpoint_pass'],'/',report['checkpoint_checked'])
    print('Missing:',len(report['missing']),'Mismatches:',len(report['mismatches']))
    for message in (report['missing']+report['mismatches']+report['warnings'])[:20]:print(' ',message)
    print('AUDIT:',report['status'])
    if args.output:
        dest=Path(args.output)
        dest.parent.mkdir(exist_ok=True,parents=True)
        dest.write_text(json.dumps(report,indent=2))
        print('SAVED:',dest)
    if report['status']!='PASS':sys.exit(2)

if __name__=='__main__':main()
