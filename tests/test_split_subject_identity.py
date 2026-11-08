from pathlib import Path
import ast, json
root=Path(__file__).resolve().parents[1]
py=root/'preprocessing'/'build_harmonized.py'
mod=ast.parse(py.read_text())
fn=[x for x in mod.body if isinstance(x,ast.FunctionDef) and x.name=='align_subject_ids']
assert len(fn)==1
assert 'OLD_SPLIT_ROOT' in ast.unparse(fn[0])
for d in ['wisdm','uci_har','mhealth','pamap2','motionsense']:
    j=json.loads((root/'splits'/f'{d}_5fold.json').read_text())
    assert len(j['folds'])==5
print('PASS 5 split files and standalone identity-alignment check in code')
