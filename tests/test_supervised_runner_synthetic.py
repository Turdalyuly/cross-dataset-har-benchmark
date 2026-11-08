"""Full one-epoch runner test using synthetic signals. Does NOT use public HAR datasets."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import numpy as np
import pandas as pd

repo=Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix='har_supervised_test_') as tmp:
    p=Path(tmp)
    d=p/'datasets/processed/harmonized_hart_v1/wisdm'
    d.mkdir(parents=True)
    s=p/'splits/harmonized_hart_v1'
    s.mkdir(parents=True)
    rng=np.random.default_rng(12)
    x=rng.normal(size=(48,120,3)).astype('float32')
    np.save(d/'X.npy',x)
    pd.DataFrame({'subject':np.repeat(np.arange(1,9),6),'label_id':np.tile(np.arange(4),12)}).to_csv(d/'metadata.csv',index=False)
    (s/'wisdm_5fold.json').write_text(json.dumps({'folds':[{'fold':0,'train_subjects':[1,2,3,4],'val_subjects':[5,6],'test_subjects':[7,8]}]}))
    args=[sys.executable,str(repo/'experiments/supervised_clean.py'),
        '--project-root',str(p),'--dataset','wisdm','--model','cnn1d','--fold','0']
    result=subprocess.run(args,check=True,capture_output=True,text=True)
    assert 'DRY RUN PASS' in result.stdout, result.stdout
    result=subprocess.run(args+['--train','--device','cpu','--epochs','1','--batch-size','16','--workers','0','--max-batches','1','--output-root',str(p/'new_runs')],check=True,capture_output=True,text=True)
    out=p/'new_runs/wisdm/cnn1d/fold0'
    assert (out/'metrics.json').is_file() and (out/'best.pt').is_file()
    metrics=json.loads((out/'metrics.json').read_text())
    assert metrics['n_test']==12 and 0<=metrics['macro_f1']<=1
    print('PASS synthetic dry-run; one epoch train, checkpoint reload, test predictions')
