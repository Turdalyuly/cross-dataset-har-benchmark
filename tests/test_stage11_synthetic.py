#!/usr/bin/env python3
"""Synthetic end-to-end tests; no real HAR data, GPUs, or internet required."""
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch

TOOL_DIR=Path(__file__).resolve().parents[1]/'tools'
sys.path.insert(0,str(TOOL_DIR))
from stage11_common import compare_saved
from replay_crosshar_strict import check_one as check_crosshar,scoped_crosshar_import,ENCODER_CFG,CLASSIFIER_CFG


def test_compare():
    n=4
    p=np.array([[.9,.05,.03,.02],[.05,.9,.03,.02],[.02,.03,.9,.05],[.02,.03,.05,.9]],dtype=np.float32)
    df=pd.DataFrame({'global_idx':list(range(n)),'y_true':list(range(n)),'y_pred':list(range(n)),
                     **{f'prob_{x}':p[:,i] for i,x in enumerate(['walking','sitting','standing','stairs'])}})
    assert compare_saved(df,np.arange(n),p,expected_idx=np.arange(n))['status']=='PASS'
    wrong=np.array([0,1,2,0]);assert compare_saved(df,wrong,p,expected_idx=np.arange(n))['status']=='MISMATCH'
    try:compare_saved(df,np.arange(n),p,expected_idx=np.arange(n)+1)
    except ValueError:pass
    else:raise AssertionError('Identity mismatch was not detected')
    wrong_df=df.drop(columns=['prob_walking'])
    try:compare_saved(wrong_df,np.arange(n),p,expected_idx=np.arange(n))
    except ValueError:pass
    else:raise AssertionError('Probability missing was not detected')
    print('PASS comparison detects labels, probabilities and identity mismatch')


def make_fake_crosshar(root):
    d=root/'external_models'/'CrossHAR';d.mkdir(parents=True)
    for n in ['config','augmentations','utils']:(d/f'{n}.py').write_text('# synthetic stub\n')
    (d/'models.py').write_text('''import torch
from torch import nn
class MaskedModel4Pretrain(nn.Module):
 def __init__(self,cfg,output_embed=False):
  super().__init__(); self.layer=nn.Linear(cfg.feature_num,cfg.hidden)
 def forward(self,x):return self.layer(x)
class Transformer_ft(nn.Module):
 def __init__(self,cfg,output_size=4):
  super().__init__(); self.layer=nn.Linear(cfg.input_size,output_size)
 def forward(self,x):
  if x.ndim==3:x=x.mean(dim=1)
  return self.layer(x)
''')
    (d/'Contrastive.py').write_text('''from torch import nn
class Contrastive(nn.Module):
 def __init__(self):super().__init__();self.layer=nn.Linear(72,72)
''')
    return d


def test_crosshar(tmp):
    project=tmp/'project';ext=make_fake_crosshar(project)
    e,c,contr=scoped_crosshar_import(ext)
    dataset='wisdm';fold=0;n=8
    ck=project/'checkpoints'/'stage5l5b_crosshar_5fold'/dataset/'fold0';ck.mkdir(parents=True)
    torch.save(e(ENCODER_CFG,output_embed=False).state_dict(),ck/'encoder_best.pt')
    torch.save(contr().state_dict(),ck/'contrastive_best.pt')
    torch.manual_seed(5)
    clf=c(CLASSIFIER_CFG,output_size=4)
    torch.save(clf.state_dict(),ck/'classifier_best.pt')
    datasetroot=project/'datasets'/'processed'/'harmonized_hart_v1'/dataset;datasetroot.mkdir(parents=True)
    meta=pd.DataFrame({'subject':[1]*n+[2]*n+[3]*n,'label_id':([0,1,2,3]*2)*3,'window_id':[f'w{i}' for i in range(3*n)]})
    meta.to_csv(datasetroot/'metadata.csv',index=False)
    splits=project/'splits'/'harmonized_hart_v1';splits.mkdir(parents=True)
    (splits/f'{dataset}_5fold.json').write_text(json.dumps({'folds':[{'fold':fold,'train_subjects':[2],'val_subjects':[3],'test_subjects':[1]}]}))
    rep=project/'results'/'stage5l5b_crosshar_5fold'/dataset/'fold0';rep.mkdir(parents=True)
    z=np.random.default_rng(123).normal(size=(n,120,72)).astype(np.float32)
    np.savez_compressed(rep/'representations.npz',Z_test=z,test_idx=np.arange(n),Z_train=np.zeros((2,120,72)),Z_val=np.zeros((2,120,72)))
    with torch.inference_mode():prob=torch.softmax(clf(torch.from_numpy(z)),dim=1).numpy()
    df=meta.iloc[:n].copy();df['global_idx']=np.arange(n);df['y_true']=meta.iloc[:n]['label_id'].to_numpy();df['y_pred']=prob.argmax(axis=1)
    for i,s in enumerate(['walking','sitting','standing','stairs']):df[f'prob_{s}']=prob[:,i]
    df.to_csv(rep/'predictions.csv',index=False)
    check=check_crosshar(project,dataset,0,'replay','cpu',4,1e-5)
    assert check['status']=='PASS' and check['comparison']['label_mismatches']==0
    df.loc[0,'y_pred']=(int(df.loc[0,'y_pred'])+1)%4;df.to_csv(rep/'predictions.csv',index=False)
    check=check_crosshar(project,dataset,0,'replay','cpu',4,1e-5)
    assert check['status']=='MISMATCH'
    print('PASS synthetic CrossHAR 3/3 strict checkpoints and classifier predictions')


def test_hart_classes_with_synthetic_dependency():
    import importlib,types
    tspulse=types.ModuleType('tsfm_public.models.tspulse')
    class DummyBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__();self.time_encoding=torch.nn.Linear(3,3);self.fft_encoding=torch.nn.Linear(3,3)
        def forward(self,past_values,**kw):
            x=self.time_encoding(past_values)+self.fft_encoding(past_values)
            return SimpleNamespace(last_hidden_state=x.transpose(1,2),loc=None,scale=None)
    class DummyDecoder(torch.nn.Module):
        def __init__(self):super().__init__();self.proj=torch.nn.Linear(120*3,4)
        def forward(self,decoder_input,**kw):
            return SimpleNamespace(prediction_outputs=self.proj(decoder_input.reshape(len(decoder_input),-1)))
    class DummyTSPulse(torch.nn.Module):
        def __init__(self):
            super().__init__();self.backbone=DummyBackbone();self.decoder_with_head=DummyDecoder()
            self.config=SimpleNamespace(context_length=120,d_model=3,patch_length=1,patch_stride=1,
                                        num_input_channels=3,num_targets=4,disable_mask_in_classification_eval=True)
        @classmethod
        def from_pretrained(cls,path,**kw):return cls()
    tspulse.TSPulseForClassification=DummyTSPulse
    names=['tsfm_public','tsfm_public.models','tsfm_public.models.tspulse']
    old={k:sys.modules.get(k) for k in names}
    for k in names:sys.modules[k]=types.ModuleType(k)
    sys.modules['tsfm_public.models.tspulse']=tspulse
    try:
        sys.modules.pop('stage11_hart_original',None)
        mod=importlib.import_module('stage11_hart_original')
        orig=mod.HARTForHAR('/synthetic',4,torch.ones(4))
        checkpoint={'state_dict':orig.state_dict()}
        copy=mod.HARTForHAR('/synthetic',4,torch.ones(4))
        copy.load_state_dict(checkpoint['state_dict'],strict=True)
        orig.eval();copy.eval()
        x=torch.randn(2,120,3)
        with torch.inference_mode():a=orig(x).logits;b=copy(x).logits
        assert torch.equal(a,b)
        # Exercise the actual Stage 11 HART runner, not just the class.
        from replay_hart_strict import check_one as check_hart
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            pre=root/'external_models'/'HART'/'finetuning_HART'/'pretrained_checkpoint';pre.mkdir(parents=True)
            ck=root/'checkpoints'/'stage5g2_hart_5fold'/'wisdm'/'fold0';ck.mkdir(parents=True)
            torch.save({'state_dict':orig.state_dict(),'normalization':{'mean':[0.,0.,0.],'std':[1.,1.,1.]}},ck/'best.pt')
            dr=root/'datasets'/'processed'/'harmonized_hart_v1'/'wisdm';dr.mkdir(parents=True)
            rng=np.random.default_rng(1)
            X=rng.standard_normal((12,120,3)).astype(np.float32)
            np.save(dr/'X.npy',X)
            meta=pd.DataFrame({'window_id':[f'id{i}' for i in range(12)],'subject':[1]*4+[2]*4+[3]*4,'label_id':[0,1,2,3]*3})
            meta.to_csv(dr/'metadata.csv',index=False)
            sr=root/'splits'/'harmonized_hart_v1';sr.mkdir(parents=True)
            (sr/'wisdm_5fold.json').write_text(json.dumps({'folds':[{'fold':0,'train_subjects':[2],'val_subjects':[3],'test_subjects':[1]}]}))
            rr=root/'results'/'stage5g2_hart_5fold'/'wisdm'/'fold0';rr.mkdir(parents=True)
            (rr/'normalization.json').write_text(json.dumps({'mean':[0,0,0],'std':[1,1,1]}))
            with torch.inference_mode():
                pp=torch.softmax(orig(torch.from_numpy(X[:4])).logits,dim=1).numpy()
            preds=meta.iloc[:4].copy();preds['y_true']=preds.label_id;preds['y_pred']=pp.argmax(axis=1)
            for i,klass in enumerate(['walking','sitting','standing','stairs']):preds[f'prob_{klass}']=pp[:,i]
            preds.to_csv(rr/'test_predictions.csv',index=False)
            assert check_hart(root,'wisdm',0,'strict','cpu',2,1e-5,0)['strict_checkpoint_load']=='PASS'
            check=check_hart(root,'wisdm',0,'replay','cpu',2,1e-5,0)
            assert check['status']=='PASS' and check['comparison']['label_mismatches']==0
        print('PASS synthetic HART strict load and full inference runner')
    finally:
        sys.modules.pop('stage11_hart_original',None)
        for k in names:
            sys.modules.pop(k,None)
            if old[k] is not None:sys.modules[k]=old[k]


def main():
    test_compare()
    with tempfile.TemporaryDirectory() as td:test_crosshar(Path(td))
    test_hart_classes_with_synthetic_dependency()
    print('STAGE11 SYNTHETIC TESTS PASS (not real model validation)')
if __name__=='__main__':main()
