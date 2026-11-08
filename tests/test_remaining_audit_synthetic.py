"""Self-contained integration test for all 100 Stage08 paths, metrics and 4-class Macro-F1."""
import sys, tempfile, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
from tools.audit_original_remaining import audit,DATASETS,MODELS,SOURCES

def main():
 with tempfile.TemporaryDirectory() as td:
  root=Path(td)
  rows=[]
  for model in MODELS:
   rel,pred,ckpt_dir,names=SOURCES[model]
   for ds in DATASETS:
    for fold in range(5):
     path=root/rel/ds/f'fold{fold}'
     path.mkdir(parents=True)
     true=np.tile(np.arange(4),8)
     predicted=true.copy()
     predicted[:fold+1]=(predicted[:fold+1]+1)%4
     f=float(f1_score(true,predicted,labels=[0,1,2,3],average='macro',zero_division=0))
     pd.DataFrame({'y_true':true,'y_pred':predicted}).to_csv(path/pred,index=False)
     (path/'metrics.json').write_text(json.dumps(dict(status='COMPLETE',model=model,dataset=ds,fold=fold,test_macro_f1=f,test_windows=len(true))))
     (path/'COMPLETE').touch()
     rows.append(dict(model=model,dataset=ds,fold=fold,macro_f1=f))
     if ds==DATASETS[0] and fold==0:
      files=root/ckpt_dir/ds/'fold0';files.mkdir(parents=True)
      for n in names:(files/n).write_bytes(b'artifact-placeholder-test-only')
  # Complete the 250-reference CSV with 150 other model entries to test filtering.
  for model in ('CNN1D','ResNet1D','TCN','TimesNet','PatchTST','TSLANet'):
   for ds in DATASETS:
    for fold in range(5):rows.append(dict(model=model,dataset=ds,fold=fold,macro_f1=.5))
  csv=root/'clean.csv';pd.DataFrame(rows).to_csv(csv,index=False)
  r=audit(root,csv,'sample')
  assert r['status']=='PASS',r
  assert r['metrics_match']==r['predictions_match']==100
  assert r['checkpoints_checked']==r['checkpoints_pass']==8
  # Missing file -> incomplete; confirms no missing file is silently ignored
  p=root/'results/stage5l4_moment_5fold/folds/wisdm/fold0/predictions.csv'
  p.unlink()
  r2=audit(root,csv,'none')
  assert r2['status']=='INCOMPLETE' and r2['predictions_match']==99 and r2['missing']
 print('PASS Stage08: 100/100 synthetic metrics, 100/100 predictions, 8 sample artifacts, missing-file detection')
if __name__=='__main__':main()
