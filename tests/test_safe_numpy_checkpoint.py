#!/usr/bin/env python3
"""No real checkpoints required: test safe legacy NumPy loading and model compatibility."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
from models.supervised_baselines import build_model
from tools.safe_numpy_checkpoint import load_numpy_metadata_checkpoint

with tempfile.TemporaryDirectory() as tmp:
    for model in ('cnn1d','resnet1d','tcn'):
        src = build_model(model)
        path=Path(tmp)/f'{model}.pt'
        torch.save({
            'state_dict':src.state_dict(),
            'dataset':'wisdm',
            'model':model,
            'fold':0,
            'epoch':7,
            'val_macro_f1':0.6,
            'mean':np.array([1.0,2.0,3.0],dtype=np.float32),
            'std':np.array([0.2,0.3,0.4],dtype=np.float32),
        },path)
        obj=load_numpy_metadata_checkpoint(path)
        dst=build_model(model)
        dst.load_state_dict(obj['state_dict'],strict=True)
        dst.eval()
        with torch.inference_mode():
            assert dst(torch.zeros(1,3,120)).shape==(1,4)
        assert obj['mean'].dtype==np.float32
        print(f'PASS safe NumPy checkpoint: {model}')
# Unknown custom globals must be blocked, never auto-allowlisted.
class NotAllowedForCheckpoint:
    pass
with tempfile.TemporaryDirectory() as tmp:
    bad=Path(tmp)/'unexpected.pt'
    torch.save({'custom':NotAllowedForCheckpoint()},bad)
    try:
        load_numpy_metadata_checkpoint(bad)
    except ValueError as ex:
        assert 'Unexpected checkpoint classes' in str(ex)
        print('PASS unknown pickle global refused')
    else:
        raise AssertionError('Unexpected global was silently accepted')
print('SAFE CHECKPOINT TESTS PASS')
