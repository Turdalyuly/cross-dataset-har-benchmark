#!/usr/bin/env python3
"""Offline smoke test for external imports/wrappers, using tiny stubs only.

NOT an accuracy test or a test of the authors' actual external model versions.
"""
import sys
import tempfile
from pathlib import Path

import torch

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
from models.foundation_baselines import build_model

TEMPLATE='''import torch
from torch import nn
class Model(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.layer=nn.Linear(3,4)
    def forward(self, x, mask, a, b):
        return self.layer(x.mean(dim=1))
'''
with tempfile.TemporaryDirectory() as td:
    root=Path(td)
    tsl=root/'external_models/Time-Series-Library/models'
    tsl.mkdir(parents=True)
    for name in ('TimesNet','PatchTST'):
        (tsl/f'{name}.py').write_text(TEMPLATE)
    tln=root/'external_models/TSLANet/Classification'
    tln.mkdir(parents=True)
    (tln/'utils.py').write_text('def pass_through(x): return x\n')
    (tln/'TSLANet_classification.py').write_text('''from torch import nn
from utils import pass_through
class TSLANet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer=nn.Linear(3,4)
    def forward(self,x):
        return pass_through(self.layer(x.mean(dim=-1)))
''')
    for model in ('timesnet','patchtst','tslanet'):
        net=build_model(model,root)
        output=net(torch.randn(2,120,3))
        assert output.shape==(2,4),output.shape
        print('PASS isolated import + output:',model)
print('STUB FACTORY TESTS PASS (not the real external models)')
