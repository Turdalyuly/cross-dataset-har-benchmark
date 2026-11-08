#!/usr/bin/env python3
"""Offline Stage09 tests. Creates synthetic files; no real checkpoints or training."""
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import torch
SCRIPT=Path(__file__).resolve().parents[1]/'tools/audit_original_pretrained_artifacts.py'
spec=importlib.util.spec_from_file_location('stage09',SCRIPT)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
with TemporaryDirectory() as td:
 root=Path(td)
 for name,(sub,files,d) in m.ARTIFACTS.items():
  folder=root/'checkpoints'/sub/'wisdm'/'fold0';folder.mkdir(parents=True)
  if d is not None:
   rs=np.random.RandomState(11)
   X=rs.randn(12,d)
   y=np.array([0,1,2,3]*3)
   scaler=StandardScaler().fit(X)
   clf=LogisticRegression(max_iter=20).fit(scaler.transform(X),y)
   joblib.dump(scaler,folder/'scaler.joblib')
   joblib.dump(clf,folder/'logreg.joblib')
  else:
   for f in files:
    torch.save({'weight':torch.tensor([1.0,2.0])},folder/f)
 a=m.run(root,'wisdm',0,False,True)
 assert a['status']=='PASS',a['errors']
 for model in ('MantisV2','MOMENT-1'):
  assert 'trusted_joblib_structure' not in a['models'][model]
 b=m.run(root,'wisdm',0,True,False)
 assert b['status']=='PASS',b['errors']
 assert b['models']['MantisV2']['trusted_joblib_structure']['n_features']==1536
 assert b['models']['MOMENT-1']['trusted_joblib_structure']['n_features']==768
 target=root/'checkpoints'/'stage5g2_hart_5fold'/'wisdm'/'fold0'/'best.pt'
 torch.save({'weight':torch.tensor([float('nan')])},target)
 c=m.run(root,'wisdm',0,False,False)
 assert c['status']=='INCOMPLETE' and any('Non-finite' in err for err in c['errors']),c['errors']
print('STAGE09 SYNTHETIC TESTS: PASS (safe torch, joblib opt-in, sha256, invalid tensors)')
