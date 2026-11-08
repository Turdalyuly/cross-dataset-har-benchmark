"""Architectural smoke checks for the original Stage 5G.3 model definitions."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from models.supervised_baselines import build_model

def main():
    for name in ('cnn1d','resnet1d','tcn'):
        model=build_model(name)
        model.eval()
        x=torch.randn(2,3,120)
        with torch.inference_mode(): out=model(x)
        assert out.shape==(2,4),(name,out.shape)
        assert torch.isfinite(out).all(),name
        state=model.state_dict()
        clone=build_model(name)
        clone.load_state_dict(state,strict=True)
        print('MODEL PASS',name, 'parameters',sum(p.numel() for p in model.parameters()), 'output', tuple(out.shape),flush=True)
    print('ALL MODEL SMOKE TESTS PASS')

if __name__=='__main__': main()
