"""Behavior checks for the experiment relocation, without training."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

EXPERIMENTS = Path(__file__).resolve().parents[1]
SCRIPTS = EXPERIMENTS / 'scripts'

class LayoutTests(unittest.TestCase):
    def paths(self, source=None):
        source = source or SCRIPTS / 'paths.py'
        self.assertTrue(source.is_file(), 'Shared relocation-aware paths module is missing')
        spec = importlib.util.spec_from_file_location('experiment_test_paths', source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_dependencies_exist_in_grouped_layout(self):
        p = self.paths()
        for path in [p.DATA / 'arithmetic_train.csv', p.DATA / 'arithmetic_eval.csv', p.DRIVER,
                     p.ORIGINAL_NOTEBOOK, p.MAIN_RUN / 'checkpoint.pt',
                     p.CONTINUATION_RUN / 'runs/lstm_e128_h512_seed2/resume.pt',
                     p.DIAGNOSTICS / 'synthetic_three_operand_holdout.csv']:
            self.assertTrue(path.is_file(), path)
        self.assertFalse((p.WORKSPACE / 'HW2_continuation_20261104_230700').exists())

    def test_paths_follow_relocated_hw2_directory(self):
        p = self.paths()
        with tempfile.TemporaryDirectory() as temp:
            hw = (Path(temp) / 'different location' / 'HW2').resolve()
            scripts = hw / 'experiments/scripts'; scripts.mkdir(parents=True)
            shutil.copy2(SCRIPTS / 'paths.py', scripts / 'paths.py')
            relocated = self.paths(scripts / 'paths.py')
            self.assertEqual(relocated.HW2, hw)
            self.assertEqual(relocated.DATA, hw / 'data')
            self.assertEqual(relocated.CONTINUATION_RUN, hw / 'experiments/runs/continuation_20261104_230700')
            (hw/'experiments/audits').mkdir()
            shutil.copy2(EXPERIMENTS/'audits/relocation_map.json',hw/'experiments/audits/relocation_map.json')
            self.assertEqual(relocated.resolve_recorded_path('/Users/shiyu./Code/AI/NTHU-NLP/HW2/experiments/mps_e128_h512_seed2'),relocated.MAIN_RUN)

    def test_historical_source_run_path_resolves(self):
        p = self.paths()
        old = '/Users/shiyu./Code/AI/NTHU-NLP/HW2/experiments/mps_e128_h512_seed2'
        self.assertEqual(p.resolve_recorded_path(old), p.MAIN_RUN)
        self.assertEqual(p.resolve_recorded_path(old + '/runs/lstm_e128_h512_seed2/history.json'),
                         p.MAIN_RUN / 'runs/lstm_e128_h512_seed2/history.json')
        self.assertEqual(p.resolve_recorded_path('experiments/runs/mps_e128_h512_seed2'),p.MAIN_RUN)
        self.assertEqual(p.resolve_recorded_path('/Users/shiyu./Code/AI/NTHU-NLP/HW2_continuation_20261104_230700/run_continuation.py'),SCRIPTS/'run_continuation.py')

    def test_continuation_preflight_from_unrelated_cwd_has_no_side_effects(self):
        p = self.paths()
        before = {f.relative_to(p.CONTINUATION_RUN).as_posix():(f.stat().st_size,f.stat().st_mtime_ns)
                  for f in p.CONTINUATION_RUN.rglob('*') if f.is_file()}
        with tempfile.TemporaryDirectory() as cwd:
            result = subprocess.run([sys.executable,str(SCRIPTS/'run_continuation.py'),'--check-paths'],
                                    cwd=cwd,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn('training_started',result.stdout)
        self.assertIn('false',result.stdout)
        after = {f.relative_to(p.CONTINUATION_RUN).as_posix():(f.stat().st_size,f.stat().st_mtime_ns)
                 for f in p.CONTINUATION_RUN.rglob('*') if f.is_file()}
        self.assertEqual(before,after)

    def test_preflight_survives_moving_and_renaming_whole_hw2(self):
        self.paths()
        with tempfile.TemporaryDirectory() as temp:
            hw = (Path(temp)/'moved assignment').resolve()
            scripts=hw/'experiments/scripts'; scripts.mkdir(parents=True)
            for name in ['paths.py','run_continuation.py']:
                shutil.copy2(SCRIPTS/name,scripts/name)
            run=hw/'experiments/runs/continuation_20261104_230700'
            required=['data/arithmetic_train.csv','data/arithmetic_eval.csv',
                      'experiments/diagnostics/synthetic_three_operand_holdout.csv',
                      'experiments/runs/continuation_20261104_230700/training_driver.ipynb',
                      'experiments/runs/continuation_20261104_230700/checkpoint.pt',
                      'experiments/runs/continuation_20261104_230700/runs/lstm_e128_h512_seed2/resume.pt']
            for name in required:
                f=hw/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_text('fixture')
            (hw/'submission.txt').write_text('protected')
            manifest={'submission.txt':hashlib.sha256(b'protected').hexdigest()}
            (run/'protected_submission_sha256.json').write_text(json.dumps(manifest))
            (run/'STOP').touch()
            result=subprocess.run([sys.executable,str(scripts/'run_continuation.py'),'--check-paths'],
                                  cwd='/',capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertEqual(json.loads(result.stdout)['run_dir'],str(run))
            self.assertFalse(json.loads(result.stdout)['training_started'])
            self.assertFalse((run/'process.json').exists())
            (hw/'submission.txt').write_text('changed')
            failed=subprocess.run([sys.executable,str(scripts/'run_continuation.py'),'--check-paths'],
                                  cwd='/',capture_output=True,text=True)
            self.assertNotEqual(failed.returncode,0)
            self.assertIn('Submitted artifact has changed',failed.stderr)

    def test_paused_continuation_requires_explicit_resume(self):
        p = self.paths()
        self.assertTrue((p.CONTINUATION_RUN/'STOP').is_file())
        result = subprocess.run([sys.executable,str(SCRIPTS/'run_continuation.py')],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('--resume',result.stdout+result.stderr)

    def test_submitted_files_match_pre_reorganization_hashes(self):
        p = self.paths()
        baseline=json.loads((p.EXPERIMENTS/'audits/reorganization_baseline.json').read_text())
        for name,digest in baseline['submission'].items():
            self.assertEqual(hashlib.sha256((p.HW2/name).read_bytes()).hexdigest(),digest,name)
        p.verify_submission(p.CONTINUATION_RUN/'protected_submission_sha256.json')

if __name__=='__main__':unittest.main()
