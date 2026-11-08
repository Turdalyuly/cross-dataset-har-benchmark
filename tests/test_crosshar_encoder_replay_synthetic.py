#!/usr/bin/env python3
"""Synthetic validation tests only; no GPU/model training/inference."""
import copy
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools import replay_crosshar_encoder as mod


class CrossHAROnePassTests(unittest.TestCase):
    def _preflight(self):
        folds=[]
        for ds in mod.DATASETS:
            for f in range(5):
                folds.append({'dataset':ds,'fold':f,'status':'READY_FOR_ENCODER_REPLAY_PREFLIGHT_ONLY',
                  'issues':[],'train_norm_exact':True,
                  'index_order_matches':{'train':True,'val':True,'test':True}})
        return {'schema':'HAR-stage13K-crosshar-encoder-preflight-v1',
                'status':'READY_FOR_ENCODER_REPLAY_PREFLIGHT_ONLY','folds_checked':25,'folds_ready':25,
                'folds':folds,'fivefold_test_coverage':{
                    ds:{'pass':True,'dataset_rows':mod.ROWS[ds],
                        'test_rows_with_zero_coverage':0,'test_rows_with_multiple_coverage':0}
                    for ds in mod.DATASETS}}

    def test_preflight_valid(self):
        mod.verify_preflight(self._preflight())

    def test_preflight_rejects_partial(self):
        p=self._preflight();p['folds_ready']=24
        with self.assertRaises(ValueError):mod.verify_preflight(p)

    def test_preflight_rejects_duplicate_fold(self):
        p=self._preflight();p['folds'][1]=copy.deepcopy(p['folds'][0])
        with self.assertRaises(ValueError):mod.verify_preflight(p)

    def test_preflight_rejects_train_norm_mismatch(self):
        p=self._preflight();p['folds'][4]['train_norm_exact']=False
        with self.assertRaises(ValueError):mod.verify_preflight(p)

    def test_preflight_rejects_missing_test_coverage(self):
        p=self._preflight();p['fivefold_test_coverage']['wisdm']['test_rows_with_zero_coverage']=1
        with self.assertRaises(ValueError):mod.verify_preflight(p)

    def test_exact_numeric_match(self):
        x=np.array([[[0.,1.],[2.,3.]],[[4.,5.],[6.,7.]]],np.float32)
        s=mod.compare_arrays(x.copy(),x.copy(),1e-5,1e-4)
        self.assertEqual((s['exact_rows'],s['allclose_rows'],s['max_absolute_difference']),(2,2,0.))

    def test_tolerance(self):
        x=np.ones((2,2,2),np.float32)
        y=x.copy(); y[0,0,0]+=2e-5
        s=mod.compare_arrays(y,x,1e-5,1e-4)
        self.assertEqual(s['exact_rows'],1)
        self.assertEqual(s['allclose_rows'],2)

    def test_detects_numeric_mismatch(self):
        x=np.zeros((3,2,2),np.float32)
        y=x.copy();y[1,0,0]=0.2
        s=mod.compare_arrays(y,x,1e-5,1e-4)
        self.assertEqual(s['allclose_rows'],2)
        self.assertEqual(s['first_mismatch_index_in_batch'],1)

    def test_rejects_nan_and_wrong_shape(self):
        x=np.zeros((2,2,2),np.float32)
        y=x.copy();y[0,0,0]=np.nan
        with self.assertRaises(ValueError):mod.compare_arrays(x,y,1e-5,1e-4)
        with self.assertRaises(ValueError):mod.compare_arrays(x,x[:1],1e-5,1e-4)

    def test_paths(self):
        p=Path('/dummy')
        r,c=mod.artifact_paths(p,'wisdm',0)
        self.assertIn('stage5l5a_crosshar_fold0',str(r))
        r,c=mod.artifact_paths(p,'wisdm',4)
        self.assertIn('stage5l5b_crosshar_5fold',str(r))

    def test_source_sha256_rejected(self):
        with TemporaryDirectory() as td:
            root=Path(td)
            for rel in mod.EXPECTED_SOURCE_SHA256:
                p=root/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('wrong')
            with self.assertRaises(RuntimeError):mod.source_proof(root)

    def test_report_write(self):
        with TemporaryDirectory() as td:
            out=Path(td)/'report.json'
            mod.write_report({'status':'synthetic','original_project_files_modified':False},out)
            self.assertEqual(json.loads(out.read_text())['status'],'synthetic')

if __name__=='__main__':
    unittest.main(verbosity=2)
