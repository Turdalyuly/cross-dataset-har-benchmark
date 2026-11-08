#!/usr/bin/env python3
"""Structure-first and optional SHA256 audit of isolated Stage 5E.2 replay.

FAST: check shape, subject counts, label counts, metadata rows without hashing arrays.
HASH: also compute SHA256 of X.npy and metadata.csv with per-file progress.
Does not modify source data. JSON is written to the replay root (or --audit-json).
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd

CHUNK = 8 * 1024 * 1024

def hash_with_progress(path: Path) -> str:
    total = path.stat().st_size
    done = 0
    hasher = hashlib.sha256()
    print(f"  SHA256 {path.name}: {total/1024**2:.1f} MiB", flush=True)
    last_report = 0
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b''):
            hasher.update(chunk)
            done += len(chunk)
            if done - last_report >= 32 * 1024 * 1024 or done == total:
                print(f"    {done/1024**2:.1f}/{total/1024**2:.1f} MiB ({100*done/max(total,1):.0f}%)", flush=True)
                last_report = done
    return hasher.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--replay-root',required=True,type=Path)
    parser.add_argument('--reference',type=Path,default=Path(__file__).resolve().parents[1] / 'reference' / 'stage5e2_original_manifest.json')
    parser.add_argument('--mode', choices=('fast','hash'), default='fast')
    parser.add_argument('--dataset', help='Optional: only check one dataset')
    parser.add_argument('--audit-json', type=Path)
    args=parser.parse_args()

    root=args.replay_root.expanduser().resolve()
    expected=json.loads(args.reference.read_text(encoding='utf-8'))['datasets']
    if args.dataset and args.dataset not in expected:
        parser.error(f"unknown dataset: {args.dataset}")
    names=[args.dataset] if args.dataset else list(expected)
    report={'mode':args.mode, 'replay_root':str(root), 'datasets':{}}
    all_ok=True
    for name in names:
        ref=expected[name]
        folder=root/'datasets'/'processed'/'harmonized_hart_v1'/name
        xp=folder/'X.npy'
        mp=folder/'metadata.csv'
        entry={'files_exist':xp.is_file() and mp.is_file()}
        print(f"\n[{name}]",flush=True)
        if not entry['files_exist']:
            entry['missing_files']=[str(p) for p in (xp,mp) if not p.is_file()]
            print(f"  MISSING: {entry['missing_files']}",flush=True)
            entry['pass']=False
            all_ok=False
            report['datasets'][name]=entry
            continue
        t0=time.monotonic()
        x=np.load(xp,mmap_mode='r',allow_pickle=False)
        meta=pd.read_csv(mp)
        counts={str(k):int(v) for k,v in Counter(meta['label']).items()}
        checks={
            'shape':list(x.shape)==ref['shape'],
            'subjects':int(meta['subject'].nunique())==ref['subjects'],
            'metadata_rows':len(meta)==len(x),
            'class_counts':counts==ref['class_counts'],
        }
        entry.update({'shape':list(x.shape), 'subjects':int(meta['subject'].nunique()),
                      'class_counts':counts, 'checks':checks})
        print(f"  Structure: {checks}  ({time.monotonic()-t0:.1f}s)",flush=True)
        if args.mode=='hash':
            file_hashes={}
            sha_matches={}
            for path in (xp,mp):
                file_hashes[path.name]=hash_with_progress(path)
                sha_matches[path.name]=file_hashes[path.name]==ref['sha256'][path.name]
                print(f"  {path.name} SHA256 match: {sha_matches[path.name]}",flush=True)
            entry['sha256']=file_hashes
            entry['sha256_match']=sha_matches
            checks.update({f'sha256_{k}':v for k,v in sha_matches.items()})
        entry['pass']=all(checks.values())
        all_ok=all_ok and entry['pass']
        report['datasets'][name]=entry
        print(f"  RESULT: {'PASS' if entry['pass'] else 'FAIL'}",flush=True)
    report['complete_for_selected_datasets']=all_ok
    output=args.audit_json or root/('REPLAY_VERIFICATION_FAST.json' if args.mode=='fast' else 'REPLAY_VERIFICATION_PROGRESS.json')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report, indent=2,ensure_ascii=False)+'\n', encoding='utf-8')
    print(f"\nAUDIT FILE: {output}",flush=True)
    if all_ok:
        print('ALL CHECKS PASS' if args.mode=='hash' else 'STRUCTURE PASS (SHA256 not checked)',flush=True)
        return 0
    return 1

if __name__=='__main__':
    sys.exit(main())
