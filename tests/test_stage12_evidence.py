"""Stage 12 report verifier unit tests. No real datasets, no GPU, no original checkpoints."""
from __future__ import annotations
import csv, json, sys, tempfile, subprocess, hashlib
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
from tools.validate_repro_evidence import validate, DATASETS, FOLDS, AuditError

def build(temp):
    ref=json.loads((BASE/'reference/stage5e2_original_manifest.json').read_text())['datasets']
    e={}
    e['preprocessing']={'mode':'hash','complete_for_selected_datasets':True,'datasets':{}}
    for name,src in ref.items():
        e['preprocessing']['datasets'][name]={
          'pass':True,'files_exist':True,'shape':src['shape'],'subjects':src['subjects'],
          'class_counts':src['class_counts'],'sha256':src['sha256'],
          'sha256_match':{'X.npy':True,'metadata.csv':True},
          'checks':{k:True for k in ['shape','subjects','metadata_rows','class_counts','sha256_X.npy','sha256_metadata.csv']}}
    for name,total,ck in [('supervised',75,3),('newmodels',75,3),('remaining',100,8)]:
        r={'status':'PASS','expected_runs':total,'metrics_match':total,'predictions_match':total,
           'missing':[],'mismatches':[]}
        if name=='remaining':r.update(checkpoints_pass=ck,checkpoints_checked=ck)
        else:r.update(checkpoint_pass=ck,checkpoint_checked=ck)
        e[name]=r
    models={}
    for model,n in [('MantisV2',2),('MOMENT-1',2),('HART',1),('CrossHAR-adapted',3)]:
        rec={'status':'PASS','files':{f'file{i}':{'status':'PASS','bytes':100,'sha256':'a'*64} for i in range(n)}}
        if model in ('MantisV2','MOMENT-1'):rec['trusted_joblib_structure']={'validation':'ok'}
        models[model]=rec
    e['artifacts']={'status':'PASS','models':models}
    e['artifacts_trusted']={'status':'PASS','models':models}
    rows=[]
    for model in ['MantisV2','MOMENT-1']:
      for name in DATASETS:
       for fold in FOLDS:
        rows.append({'model':model,'dataset':name,'fold':fold,'status':'PASS','predicted_label_mismatches':0,
                     'windows_test':100,'max_absolute_probability_error':1e-16})
    e['frozen_heads']={'status':'PASS','expected':50,'passed':50,'failed_checks':0,'errors':[], 'replayed_folds':rows}
    for stage,model in [('crosshar','CrossHAR-adapted'),('hart','HART')]:
      rows=[]
      for name in DATASETS:
       for fold in FOLDS:
        rows.append({'model':model,'dataset':name,'fold':fold,'mode':'replay','strict_checkpoint_load':'PASS',
                     'status':'PASS', 'comparison':{'status':'PASS','label_mismatches':0,'windows':2,'max_probability_difference':3e-8}})
        if model=='HART': p=temp/'results/stage5g2_hart_5fold'/name/f'fold{fold}'/'test_predictions.csv'
        else: p=temp/'results/stage5l5b_crosshar_5fold'/name/f'fold{fold}'/'predictions.csv'
        p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('w',newline='') as f:
           w=csv.writer(f);w.writerow(['y_true','y_pred']);w.writerow([1,1]);w.writerow([1,1])
      e[stage]={'status':'PASS','mode':'replay','pass_count':25,'total':25,'results':rows,'errors':[]}
    paths={}
    for name,doc in e.items():
       path=temp/f'{name}.json';path.write_text(json.dumps(doc));paths[name]=path
    return paths

def main():
  with tempfile.TemporaryDirectory() as td:
    tmp=Path(td)
    paths=build(tmp)
    result=validate(paths,tmp)
    assert result['result']=='PASS'
    assert result['evidence']['hart']['full_csv_length_checked']
    assert result['evidence']['preprocessing']['sha256_files_match']==10
    print('PASS complete synthetic evidence reports 10 hashes, 250 scores, 50 heads, 25+25 replays')
    bad=json.loads(paths['hart'].read_text())
    bad['results'][0]['comparison']['windows']=1
    paths['hart'].write_text(json.dumps(bad))
    try:validate(paths,tmp)
    except AuditError as e:assert 'PARTIAL replay' in str(e)
    else: raise AssertionError('Partial HART replay must be refused')
    print('PASS partial-window HART replay is refused')
    paths=build(tmp)
    bad=json.loads(paths['crosshar'].read_text())
    bad['results'][0]['comparison']['max_probability_difference']=0.015
    paths['crosshar'].write_text(json.dumps(bad))
    try:validate(paths,tmp)
    except AuditError as e:assert 'probability difference' in str(e)
    else:raise AssertionError('Excessive probability difference must be refused')
    print('PASS excessive replay probability difference is refused')
    paths=build(tmp)
    bad=json.loads(paths['frozen_heads'].read_text())
    bad['replayed_folds'][1]=bad['replayed_folds'][0].copy()
    paths['frozen_heads'].write_text(json.dumps(bad))
    try:validate(paths,tmp)
    except AuditError as e:assert 'repeated key' in str(e)
    else:raise AssertionError('Duplicated dataset/fold must be refused')
    print('PASS duplicate folds are refused')
  with tempfile.TemporaryDirectory() as td:
    tmp=Path(td)
    paths=build(tmp)
    output=tmp/'HAR_STAGE12_PUBLIC_EVIDENCE.json'
    cmd=[sys.executable,str(BASE/'tools/validate_repro_evidence.py')]
    for name,path in paths.items():
      cmd.extend(['--'+name.replace('_','-'),str(path)])
    cmd.extend(['--project-root',str(tmp),'--output',str(output)])
    run=subprocess.run(cmd, capture_output=True, text=True, timeout=45)
    assert run.returncode==0, (run.stdout,run.stderr)
    assert 'STAGE12 EVIDENCE: PASS' in run.stdout
    evidence=json.loads(output.read_text(encoding='utf-8'))
    assert evidence['result']=='PASS'
    assert set(evidence['source_report_sha256'])==set(paths), 'All nine report hashes must be present'
    for name,path in paths.items():
      assert evidence['source_report_sha256'][name]==hashlib.sha256(path.read_bytes()).hexdigest(), name
    assert str(tmp) not in output.read_text(encoding='utf-8'), 'Output must be path-free'
    assert evidence['evidence']['crosshar']['full_csv_length_checked'] is True
    assert evidence['evidence']['hart']['full_csv_length_checked'] is True
    print('PASS CLI regression: artifacts_trusted included, nine report SHA256, full CSV checks, path-free JSON')
    # The trusted audit input is mandatory and must never be silently omitted.
    paths['artifacts_trusted'].unlink()
    denied=subprocess.run(cmd, capture_output=True, text=True, timeout=45)
    assert denied.returncode!=0 and 'INCOMPLETE' in denied.stdout, (denied.stdout,denied.stderr)
    print('PASS CLI refuses missing trusted-artifacts report')
  print('STAGE12 EVIDENCE UNIT TESTS PASS')

if __name__=='__main__':main()
