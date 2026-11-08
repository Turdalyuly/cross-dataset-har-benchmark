#!/usr/bin/env python3
"""Faithful isolated model definitions from corrected historical Stage 5L.2.

Does not execute training on import; official external model source must be supplied
by the researcher under ``external_models/`` in their original project.

The two repositories both use generic modules called ``utils``. The TSLANet
loader mirrors the verified Stage5L2 temporary-module-isolation workaround.
"""
from __future__ import annotations
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn


class TimesNetWrapper(nn.Module):
    def __init__(self, model):
        super().__init__(); self.model = model
    def forward(self, x):
        mask = torch.ones(x.shape[:2], dtype=x.dtype, device=x.device)
        return self.model(x, mask, None, None)


class PatchTSTWrapper(nn.Module):
    def __init__(self, model):
        super().__init__(); self.model = model
    def forward(self, x):
        mask = torch.ones(x.shape[:2], dtype=x.dtype, device=x.device)
        return self.model(x, mask, None, None)


class TSLANetWrapper(nn.Module):
    def __init__(self, model):
        super().__init__(); self.model = model
    def forward(self, x):
        return self.model(x.permute(0, 2, 1))


def _load_file(unique_name: str, file: Path):
    spec = importlib.util.spec_from_file_location(unique_name, str(file))
    if spec is None or spec.loader is None:
        raise ImportError(f'Could not import model file: {file}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(unique_name, None)
        raise
    return module


def _load_ts_library(project_root: Path, model: str):
    root = project_root/'external_models/Time-Series-Library'
    file = root/'models'/f'{model}.py'
    if not file.is_file():
        raise FileNotFoundError(f'Original Time-Series-Library source missing: {file}')
    prev = list(sys.path)
    try:
        # Internal Time-Series-Library imports use `layers` and `utils` packages.
        sys.path.insert(0, str(root))
        module = _load_file(f'_har_original_tslib_{model}', file)
    finally:
        sys.path[:] = prev
    if not hasattr(module, 'Model'):
        raise AttributeError(f'{file}: missing class Model')
    return module.Model


def _load_tslanet(project_root: Path):
    root = project_root/'external_models/TSLANet/Classification'
    file = root/'TSLANet_classification.py'
    if not file.is_file():
        raise FileNotFoundError(f'Original TSLANet source missing: {file}')
    # Mirror original Stage 5L.2 to avoid utils/dataloader name collision.
    conflicts = ['utils', 'dataloader']
    saved = {name:sys.modules.pop(name) for name in conflicts if name in sys.modules}
    previous = list(sys.path)
    try:
        sys.path.insert(0, str(root))
        module = _load_file('_har_original_tslanet', file)
        loaded_utils = sys.modules.get('utils')
        utils_file = Path(getattr(loaded_utils,'__file__','') or '')
        if not utils_file.is_file() or root.resolve() not in utils_file.resolve().parents:
            raise RuntimeError(f'TSLANet imported the wrong utils module: {utils_file}')
        if not hasattr(module, 'TSLANet'):
            raise AttributeError(f'{file}: missing TSLANet class')
    finally:
        sys.path[:] = previous
        for name in conflicts:sys.modules.pop(name, None)
        sys.modules.update(saved)
    return module


def build_model(model_name: str, project_root: str|Path) -> nn.Module:
    """Return model accepting [batch,120,3], outputting [batch,4] logits."""
    project_root = Path(project_root)
    model_name = model_name.lower()
    if model_name == 'timesnet':
        Model = _load_ts_library(project_root, 'TimesNet')
        cfg = SimpleNamespace(task_name='classification', seq_len=120, label_len=0,
            pred_len=0,enc_in=3,c_out=3,d_model=64,d_ff=128,e_layers=2,
            top_k=3,num_kernels=6,embed='fixed',freq='h',dropout=0.1,num_class=4)
        return TimesNetWrapper(Model(cfg))
    if model_name == 'patchtst':
        Model = _load_ts_library(project_root, 'PatchTST')
        cfg = SimpleNamespace(task_name='classification',seq_len=120,pred_len=0,
            enc_in=3,d_model=64,d_ff=128,e_layers=2,n_heads=4,factor=1,
            dropout=0.1,activation='gelu',num_class=4)
        return PatchTSTWrapper(Model(cfg,patch_len=16,stride=8))
    if model_name == 'tslanet':
        module = _load_tslanet(project_root)
        module.args = SimpleNamespace(seq_len=120,patch_size=8,num_channels=3,
            emb_dim=128,depth=2,dropout_rate=0.15,num_classes=4,
            ICB=True,ASB=True,adaptive_filter=True,masking_ratio=0.4)
        return TSLANetWrapper(module.TSLANet())
    raise ValueError(f'Unknown architecture: {model_name}')
