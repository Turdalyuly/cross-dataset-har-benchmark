#!/usr/bin/env python3
"""Offline test that 75 synthetic test predictions are independently re-scored."""
import json
import sys
import tempfile
from pathlib import Path
import pandas as pd

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
from tools.audit_original_foundation import run, MODELS, DATASETS

with tempfile.TemporaryDirectory() as td:
    root=Path(td)
    rows=[]
    for model in MODELS:
      for ds in DATASETS:
        for fold in range(5):
            folder=root/'results/stage5l3_top10_newmodels_5fold'/model/ds/f'fold{fold}'
            folder.mkdir(parents=True)
            (folder/'COMPLETE').write_text('synthetic')
            (folder/'metrics.json').write_text(json.dumps({'test_macro_f1':1.0,'status':'COMPLETE'}))
            pd.DataFrame({'y_true':[0,1,2,3],'y_pred':[0,1,2,3]}).to_csv(folder/'predictions.csv',index=False)
            rows.append({'model':model,'dataset':ds,'fold':fold,'macro_f1':1.0})
    csv=root/'data.csv'
    pd.DataFrame(rows).to_csv(csv,index=False)
    result=run(root,csv,'none')
    assert result['status']=='PASS', result
    assert result['metrics_match']==75 and result['predictions_match']==75
    print('PASS 75 synthetic aggregate values and 75 re-scored predictions')

# Original evaluation explicitly uses all four classes even if one is absent.
from sklearn.metrics import f1_score
y_true=[0,1,2]; y_pred=[0,1,2]
assert abs(f1_score(y_true,y_pred,labels=[0,1,2,3],average='macro',zero_division=0)-0.75)<1e-12
print('PASS fixed 4-class macro-F1 convention')
