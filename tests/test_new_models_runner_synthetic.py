#!/usr/bin/env python3
"""Checks dry-runs for all three and one short synthetic training run (TimesNet)."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import numpy as np
import pandas as pd

REPO=Path(__file__).resolve().parents[1]
STUB='''import torch
from torch import nn
class Model(nn.Module):
    def __init__(self,args,**kwargs):
        super().__init__()
        self.layer=nn.Linear(3,4)
    def forward(self,x,mask,a,b):
        return self.layer(x.mean(dim=1))
'''
with tempfile.TemporaryDirectory() as tmp:
    p=Path(tmp)
    external=p/'external_models/Time-Series-Library/models'
    external.mkdir(parents=True)
    for model in ('TimesNet','PatchTST'):(external/f'{model}.py').write_text(STUB)
    tsl=p/'external_models/TSLANet/Classification'
    tsl.mkdir(parents=True)
    (tsl/'utils.py').write_text('def f(x): return x\n')
    (tsl/'TSLANet_classification.py').write_text('''from torch import nn
from utils import f
class TSLANet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer=nn.Linear(3,4)
    def forward(self,x):
        return f(self.layer(x.mean(dim=-1)))
''')
    data=p/'datasets/processed/harmonized_hart_v1/wisdm'
    data.mkdir(parents=True)
    splits=p/'splits/harmonized_hart_v1'
    splits.mkdir(parents=True)
    rng=np.random.default_rng(11)
    np.save(data/'X.npy',rng.normal(size=(48,120,3)).astype('float32'))
    pd.DataFrame({'subject':np.repeat(np.arange(1,9),6),'label_id':np.tile(np.arange(4),12)}).to_csv(data/'metadata.csv',index=False)
    (splits/'wisdm_5fold.json').write_text(json.dumps({'folds':[{'fold':0,'train_subjects':[1,2,3,4],'val_subjects':[5,6],'test_subjects':[7,8]}]}))
    root=[sys.executable,str(REPO/'experiments/new_models_clean.py'),'--project-root',str(p),'--dataset','wisdm','--fold','0']
    for model in ('timesnet','patchtst','tslanet'):
        args=root+['--model',model]
        r=subprocess.run(args,capture_output=True,text=True)
        assert r.returncode==0 and 'DRY RUN PASS' in r.stdout,(model,r.stderr,r.stdout)
        print('PASS',model,'synthetic model dry-run')
    cmd=root+['--model','timesnet','--train','--device','cpu','--epochs','1','--batch-size','16',
      '--workers','0','--max-batches','1','--output-root',str(p/'new_runs')]
    r=subprocess.run(cmd,capture_output=True,text=True)
    assert r.returncode==0,(r.stderr,r.stdout)
    assert (p/'new_runs/wisdm/timesnet/fold0/best.pt').is_file()
    assert (p/'new_runs/wisdm/timesnet/fold0/metrics.json').is_file()
    print('PASS 1-epoch synthetic training, checkpoint reload and predictions')
