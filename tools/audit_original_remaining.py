#!/usr/bin/env python3
"""Read-only Stage 08 audit: 100 clean evaluations for MantisV2, MOMENT-1, HART and CrossHAR-adapted.

The source path conventions were reconstructed from final Stage5G.1, Stage5G.2,
Stage5L.4 and Stage5L.5B scripts. Checks CSV predictions against recorded metrics
and the original 250-row clean benchmark. Does not train, import external models,
execute checkpoints, or modify source artifacts.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

BASE = Path(__file__).resolve().parents[1]
DATASETS = ("wisdm", "uci_har", "mhealth", "pamap2", "motionsense")
MODELS = ("MantisV2", "MOMENT-1", "HART", "CrossHAR-adapted")
# Explicit final source locations: no speculative recursive scanning or alternative runs.
SOURCES = {
    "MantisV2": ("results/stage5g1_mantisv2_hart_5fold", "test_predictions.csv", "checkpoints/stage5g1_mantisv2_hart_5fold", ("scaler.joblib", "logreg.joblib")),
    "MOMENT-1": ("results/stage5l4_moment_5fold/folds", "predictions.csv", "checkpoints/stage5l4_moment_5fold", ("scaler.joblib", "logreg.joblib")),
    "HART": ("results/stage5g2_hart_5fold", "test_predictions.csv", "checkpoints/stage5g2_hart_5fold", ("best.pt",)),
    "CrossHAR-adapted": ("results/stage5l5b_crosshar_5fold", "predictions.csv", "checkpoints/stage5l5b_crosshar_5fold", ("encoder_best.pt", "contrastive_best.pt", "classifier_best.pt")),
}

def audit(root: Path, reference_csv: Path, artifact_mode: str = "sample", tolerance: float = 2e-5) -> dict:
    root = Path(root)
    df = pd.read_csv(reference_csv)
    required = {"model", "dataset", "fold", "macro_f1"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns in benchmark CSV: {sorted(required-set(df.columns))}")
    if len(df) != 250:
        raise ValueError(f"Expected 250 clean rows, got {len(df)}")
    keys = [(str(r.model), str(r.dataset).lower(), int(r.fold)) for r in df.itertuples()]
    if len(set(keys)) != 250:
        raise ValueError("Duplicate clean benchmark model/dataset/fold keys")
    expected = dict(zip(keys, [float(v) for v in df.macro_f1]))
    for m in MODELS:
        for d in DATASETS:
            for f in range(5):
                if (m, d, f) not in expected:
                    raise ValueError(f"Missing reference key: {(m,d,f)}")
    report = dict(expected_runs=100, metrics_match=0, predictions_match=0,
                  checkpoints_checked=0, checkpoints_pass=0, missing=[], mismatches=[],
                  warnings=[], runs=[], limits=[
                      "Verifies saved predictions and metadata, not repeatability of model training.",
                      "Checkpoint checks are file-existence/integrity only: no unsafe deserialization or backbone compatibility claims.",
                      "Stage5L.5B is the original 100-epoch benchmark; this does not validate the R1 200-epoch extension.",
                  ])
    for model in MODELS:
        results_dir, prediction_name, ckpt_dir, ckpt_names = SOURCES[model]
        for dataset in DATASETS:
            for fold in range(5):
                rel = Path(results_dir)/dataset/f"fold{fold}"
                loc = root/rel
                row = dict(model=model,dataset=dataset,fold=fold,source=str(rel))
                required_files = ("COMPLETE", "metrics.json", prediction_name)
                present = all((loc/f).is_file() for f in required_files)
                if not present:
                    for f in required_files:
                        if not (loc/f).is_file(): report["missing"].append(str(rel/f))
                else:
                    try:
                        md=json.loads((loc/"metrics.json").read_text(encoding="utf-8"))
                        if md.get("status") != "COMPLETE":
                            report["mismatches"].append(f"{rel}: status is {md.get('status')!r}, not COMPLETE")
                        if str(md.get("dataset", "")).lower() != dataset or int(md.get("fold",-1)) != fold:
                            report["mismatches"].append(f"{rel}: model metric dataset/fold mismatch")
                        if md.get("model") and str(md["model"]).lower() != model.lower():
                            report["mismatches"].append(f"{rel}: model name mismatch {md['model']!r}")
                        measured = float(md["test_macro_f1"])
                        target = expected[(model,dataset,fold)]
                        if np.isfinite(measured) and abs(measured-target) <= tolerance:
                            report["metrics_match"] += 1
                        else:
                            report["mismatches"].append(f"{rel}: metric={measured:.8f} reference={target:.8f}")
                        preds = pd.read_csv(loc/prediction_name, usecols=["y_true", "y_pred"])
                        if len(preds) == 0 or preds.isna().any().any():
                            raise ValueError("Empty predictions or missing class labels")
                        yt=preds.y_true.astype(int).to_numpy()
                        yp=preds.y_pred.astype(int).to_numpy()
                        if not (np.isin(yt,[0,1,2,3]).all() and np.isin(yp,[0,1,2,3]).all()):
                            raise ValueError("Class IDs outside harmonized range 0..3")
                        recalc = float(f1_score(yt,yp,labels=[0,1,2,3],average="macro",zero_division=0))
                        if abs(recalc-target) <= tolerance and abs(recalc-measured) <= tolerance:
                            report["predictions_match"] += 1
                        else:
                            report["mismatches"].append(f"{rel}: predictions={recalc:.8f} metric={measured:.8f} reference={target:.8f}")
                        n_exp = md.get("n_test",md.get("test_windows",None))
                        if n_exp is not None and int(n_exp) != len(preds):
                            report["mismatches"].append(f"{rel}: prediction count {len(preds)} != test windows {n_exp}")
                        row.update(test_macro_f1=measured,recomputed_macro_f1=recalc,expected_macro_f1=target,
                                   n_predictions=len(preds))
                    except Exception as exc:
                        report["mismatches"].append(f"{rel}: {type(exc).__name__}: {exc}")
                if artifact_mode == "all" or (artifact_mode == "sample" and dataset == DATASETS[0] and fold == 0):
                    ckpt_rel=Path(ckpt_dir)/dataset/f"fold{fold}"
                    for artifact in ckpt_names:
                        report["checkpoints_checked"] += 1
                        file = root/ckpt_rel/artifact
                        if not file.is_file():
                            report["missing"].append(str(ckpt_rel/artifact))
                        elif file.stat().st_size == 0:
                            report["mismatches"].append(f"{ckpt_rel/artifact}: zero-byte artifact")
                        else:
                            report["checkpoints_pass"] += 1
                            row.setdefault("artifacts",[]).append({"file":str(ckpt_rel/artifact),"bytes":file.stat().st_size})
                report["runs"].append(row)
    report["status"]="PASS" if (report["metrics_match"]==100 and report["predictions_match"]==100
                    and report["checkpoints_checked"]==report["checkpoints_pass"]
                    and not report["missing"] and not report["mismatches"]) else "INCOMPLETE"
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root",required=True)
    p.add_argument("--public-csv",default=str(BASE/"results/clean_250_fold_scores.csv"))
    p.add_argument("--check-artifacts",choices=("none","sample","all"),default="sample")
    p.add_argument("--output")
    a=p.parse_args()
    r=audit(Path(a.project_root),Path(a.public_csv),a.check_artifacts)
    print(f"Metrics against Table 2: {r['metrics_match']} /100")
    print(f"Predictions recalculated: {r['predictions_match']} /100")
    print(f"Artifact files present: {r['checkpoints_pass']} / {r['checkpoints_checked']} (NOT model compatibility)")
    print(f"Missing: {len(r['missing'])} Mismatches: {len(r['mismatches'])}")
    for item in (r['missing']+r['mismatches'])[:22]:print(" ",item)
    print("AUDIT:",r["status"])
    if a.output:
        file=Path(a.output);file.parent.mkdir(parents=True,exist_ok=True)
        file.write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding="utf-8")
        print("SAVED:",file)
    return 0 if r["status"]=="PASS" else 2
if __name__=="__main__":sys.exit(main())
