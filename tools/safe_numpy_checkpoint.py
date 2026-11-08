#!/usr/bin/env python3
"""Restricted loader for original Stage 5G.3 checkpoints containing NumPy mean/std.

Keeps weights_only=True. Only permits narrowly specified NumPy serialization
constructors from the authors' original checkpoint format. It refuses all other
unknown globals and does not execute arbitrary checkpoint Python objects.

WARNING: Only load artifacts that you created or otherwise trust. Even with
weights_only=True, untrusted deserialization is not risk-free.
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

_NUMPY_GLOBALS = {
    'numpy.core.multiarray._reconstruct',
    'numpy._core.multiarray._reconstruct',
    'numpy.core.multiarray.scalar',
    'numpy._core.multiarray.scalar',
    'numpy.ndarray',
    'numpy.dtype',
}


def _restricted_numpy_globals(checkpoint: str | Path) -> list[Any]:
    if not hasattr(torch.serialization, 'get_unsafe_globals_in_checkpoint'):
        raise RuntimeError('PyTorch version lacks get_unsafe_globals_in_checkpoint')
    unsafe = set(torch.serialization.get_unsafe_globals_in_checkpoint(str(checkpoint)))
    unexpected = sorted(unsafe - _NUMPY_GLOBALS)
    if unexpected:
        raise ValueError('Unexpected checkpoint classes; refusing to load: ' + ', '.join(unexpected))

    # NumPy 1.x used numpy.core; NumPy 2.x uses numpy._core.
    core = importlib.import_module('numpy._core.multiarray' if np.__version__.split('.')[0] != '1' else 'numpy.core.multiarray')
    symbols = {
        'numpy.core.multiarray._reconstruct': core._reconstruct,
        'numpy._core.multiarray._reconstruct': core._reconstruct,
        'numpy.core.multiarray.scalar': core.scalar,
        'numpy._core.multiarray.scalar': core.scalar,
        'numpy.ndarray': np.ndarray,
        'numpy.dtype': np.dtype,
    }
    # tuple-based registrations explicitly match legacy and current pickle paths
    allowlist = [(symbols[name], name) for name in sorted(unsafe)]
    # NumPy dtype classes can be created dynamically during unpickling;
    # include only primitive numeric dtype classes required by normalization.
    allowlist.extend({
        type(np.dtype('float16')), type(np.dtype('float32')),
        type(np.dtype('float64')), type(np.dtype('int32')),
        type(np.dtype('int64')), type(np.dtype('uint8')),
        type(np.dtype('bool')),
    })
    return allowlist


def load_numpy_metadata_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load original checkpoint with restricted NumPy allowlisting.

    Never falls back to weights_only=False. Raises if unexpected globals occur.
    """
    path = Path(path)
    allowlist = _restricted_numpy_globals(path)
    with torch.serialization.safe_globals(allowlist):
        obj = torch.load(str(path), map_location='cpu', weights_only=True)
    if not isinstance(obj, dict) or not isinstance(obj.get('state_dict'), dict):
        raise ValueError('Unexpected checkpoint structure: expected dict with state_dict')
    for key in ('mean', 'std', 'dataset', 'model', 'fold'):
        if key not in obj:
            raise ValueError('Missing checkpoint field: '+key)
    return obj
