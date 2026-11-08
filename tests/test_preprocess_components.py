#!/usr/bin/env python3
"""Synthetic unit tests of the exact recovered preprocessing functions, without running full pipeline."""
import ast, pathlib, tempfile, re, gc, numpy as np, pandas as pd
from scipy.signal import resample_poly
from fractions import Fraction
p=pathlib.Path(__file__).resolve().parents[1]/'preprocessing/build_harmonized.py'
source=p.read_text(); tree=ast.parse(source)
names={'fill_missing','convert_acc_to_g','resample_to_30hz','process_motionsense'}
selected=[x for x in tree.body if isinstance(x,(ast.FunctionDef,ast.ClassDef)) and (x.name in names or x.name=='DatasetBuilder')]
gl={'np':np,'pd':pd,'re':re,'gc':gc,'resample_poly':resample_poly,'Fraction':Fraction,'G':9.80665,'TARGET_FS':30.0,'WINDOW_LENGTH':120,'WINDOW_SECONDS':4.0,'HOP_LENGTH':30,'HOP_SECONDS':1.0,'CLASS_TO_ID':{'walking':0,'sitting':1,'standing':2,'stairs':3},'MOTIONSENSE_MAP':{'wlk':'walking','sit':'sitting','std':'standing','ups':'stairs','dws':'stairs'}}
exec(compile(ast.Module(body=selected,type_ignores=[]),str(p),'exec'),gl)
with tempfile.TemporaryDirectory() as temp:
    root=pathlib.Path(temp)
    a=root/'A_DeviceMotion_data/wlk_1';a.mkdir(parents=True)
    b=root/'B_Accelerometer_data/wlk_1';b.mkdir(parents=True)
    n=700
    data={'gravity.x':np.ones(n,dtype=np.float32), 'gravity.y':np.zeros(n),'gravity.z':np.zeros(n), 'userAcceleration.x':np.zeros(n),'userAcceleration.y':np.zeros(n),'userAcceleration.z':np.zeros(n)}
    pd.DataFrame(data).to_csv(a/'sub_1.csv',index=False)
    pd.DataFrame({'x':np.ones(n),'y':np.zeros(n),'z':np.zeros(n)}).to_csv(b/'sub_1.csv',index=False)
    gl['find_dataset_root']=lambda _:root
    ((x,meta),audit)=gl['process_motionsense']()
    assert x.ndim==3 and x.shape[1:]==(120,3),(x.shape)
    assert len(x)==len(meta) and len(x)>0
    assert np.mean(np.abs(x[:,:,0]-1))<0.01
    assert np.max(np.abs(x[:,:,1:]))<1e-4
    assert audit['selected_device_motion_files']==1,audit
    assert audit['all_csv_files']==2,audit
    assert meta['label_id'].unique().tolist()==[0]
    print('PASS: synthetic MotionSense selects A_DeviceMotion_data only; gravity+userAcceleration; output 120x3; metadata aligned')
x=np.full((250,3),9.80665,dtype=np.float32)
g,kind=gl['convert_acc_to_g'](x,'ms2')
assert kind=='m_s2_to_g' and np.allclose(g,1), (kind,g[0])
print('PASS: m/s^2 to g conversion')
