#!/usr/bin/env python3
"""Standard-library-only integrity and derived-table generator for retained HAR outputs.
NOT a model-training reproduction. Use --export-csv to emit derived tables.
"""
import argparse, csv, json, math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
DATASETS = ('wisdm','uci_har','mhealth','pamap2','motionsense')
CLASSES = 4

def read_csv(name):
    with (RESULTS/name).open(newline='',encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def check(ok, message):
    if not ok: raise AssertionError(message)
    print('PASS',message)

def output_csv(name,rows,fields):
    dest=ROOT/'generated'/name; dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    print('WRITE',dest.relative_to(ROOT))

def main(export=False):
    clean=read_csv('clean_250_fold_scores.csv')
    key={(r['model'],r['dataset'],int(r['fold'])) for r in clean}
    models=sorted({r['model'] for r in clean})
    check(len(clean)==len(key)==250 and len(models)==10,'250 unique model/dataset/fold scores, 10 models')
    check(set(r['dataset'] for r in clean)==set(DATASETS),'five acquisition datasets')
    check(all(sum((m,d,f) in key for f in range(5))==5 for m in models for d in DATASETS),'five folds per model/dataset')
    means={}
    for m in models:
        for d in DATASETS:
            vals=[float(r['macro_f1']) for r in clean if r['model']==m and r['dataset']==d]
            check(len(vals)==5 and all(math.isfinite(v) and 0<=v<=1 for v in vals),f'five finite scores: {m}/{d}')
            means[m,d]=sum(vals)/5
    overall={m:sum(means[m,d] for d in DATASETS)/5 for m in models}
    prior=read_csv('original_clean_ranking.csv')
    check(len(prior)==10 and set(r['model'] for r in prior)==set(models),'published ranking contains 10 models')
    for row in prior:
        m=row['model'];v=float(row['mean_dataset_macro_f1'])
        check(abs(v-overall[m])<1e-8,f'original aggregate matches: {m}')
    ranked=sorted(models,key=lambda m:(-overall[m],m))
    check(ranked[0]=='MantisV2','MantisV2 first in original ranking')
    table2=[{'rank':i+1,'model':m,'macro_f1':f'{overall[m]:.8f}'} for i,m in enumerate(ranked)]
    exrows=[]
    for d in DATASETS:
        values={m:sum(means[m,dd] for dd in DATASETS if dd!=d)/4 for m in models}
        order=sorted(models,key=lambda m:(-values[m],m))
        check(order[0]=='MantisV2',f'MantisV2 first excluding {d}')
        exrows += [{'excluded_dataset':d,'rank':i+1,'model':m,'macro_f1':f'{values[m]:.8f}'} for i,m in enumerate(order)]
    splits=[]
    for d in DATASETS:
        j=json.loads((ROOT/'splits'/f'{d}_5fold.json').read_text())
        check(j['dataset']==d and len(j['folds'])==5,f'five predefined splits: {d}')
        all_test=[]; ref=None
        for fold in j['folds']:
            ss=[set(map(str,fold[k+'_subjects'])) for k in ('train','val','test')]
            check(all(ss[i].isdisjoint(ss[j]) for i,j in ((0,1),(0,2),(1,2))),f'no overlap: {d}/fold{fold["fold"]}')
            union=set.union(*ss)
            if ref is None:ref=union
            check(union==ref and len(ss[2])>0,f'full subject partition: {d}/fold{fold["fold"]}')
            all_test.extend(ss[2]); splits.append({'dataset':d,'fold':fold['fold'],'train':len(ss[0]),'val':len(ss[1]),'test':len(ss[2])})
        check(len(all_test)==len(set(all_test))==len(ref),f'each subject tested exactly once: {d}')
    def assert_no_target(row):
        return all(row.get(k,'False').lower() in ('false','0','') for k in ('target_used_for_training','target_used_for_validation','target_used_for_normalization','target_used_for_scaler'))
    for name,expect in [('lodo_mantis.csv',5),('lodo_hart.csv',5),('directional_mantis.csv',20),('directional_hart.csv',20)]:
        rows=read_csv(name)
        check(len(rows)==expect and all(assert_no_target(r) for r in rows),f'{name}: {expect} rows and target exclusion flags')
        if expect==20:
            pairs={(r['source_dataset'],r['target_dataset']) for r in rows}
            check(pairs=={(a,b) for a in DATASETS for b in DATASETS if a!=b},f'{name}: complete 20 directed pairs')
    supervised=read_csv('lodo_cnn1d_tslanet.csv')
    check(len(supervised)==10,'10 supervised-from-scratch LODO results')
    r1=read_csv('crosshar_R1_3_verified25_dataset_summary.csv')
    check(len(r1)==5,'five R1 CrossHAR dataset-level comparisons')
    if export:
        output_csv('Table2_clean_ranking.csv',table2,['rank','model','macro_f1'])
        output_csv('TableS1_50_dataset_exclusions.csv',exrows,['excluded_dataset','rank','model','macro_f1'])
        output_csv('TableS2_25_subject_counts.csv',splits,['dataset','fold','train','val','test'])
        for name,dst in [('directional_mantis.csv','TableS3_Mantis_directional.csv'),('directional_hart.csv','TableS4_HART_directional.csv')]:
            rows=read_csv(name); lookup={(r['source_dataset'],r['target_dataset']):float(r['target_macro_f1']) for r in rows}
            table=[dict(source=src,**{dstname:('' if src==dstname else f'{lookup[src,dstname]:.6f}') for dstname in DATASETS}) for src in DATASETS]
            output_csv(dst,table,['source',*DATASETS])
    print('FINAL: PASS — artifact-level integrity and derived-score checks; NO MODEL TRAINING WAS RUN')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--export-csv',action='store_true'); a=p.parse_args();main(a.export_csv)
