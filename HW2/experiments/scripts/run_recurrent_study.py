"""Sequential, resumable GRU/LSTM study without touching submission artifacts."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import torch
from paths import SCRIPTS
from recurrent_study import study_hw2, verify_study
from train_recurrent import atomic_json


def prepare(root):
    hw=study_hw2(root);root.mkdir(parents=True,exist_ok=True)
    frozen=root/'frozen_submission.json'
    if not frozen.exists():
        names=['NLP_HW2_NCU_114502540.ipynb','NLP_HW2_NCU_114502540.zip','checkpoint.pt','requirements.txt',
               'Assignment2/main.ipynb','data/arithmetic_train.csv','data/arithmetic_eval.csv']
        atomic_json(frozen,{name:hashlib.sha256((hw/name).read_bytes()).hexdigest() for name in names})
    verify_study(root)
    if not (root/'study.json').exists():
        old=hw/'experiments/runs/continuation_20261104_230700'
        state_path=old/'runs/lstm_e128_h512_seed2/resume.pt'
        state=torch.load(state_path,map_location='cpu',weights_only=True)
        assert state['next_epoch']==29 and state['next_batch']==830
        target=root/'lstm';(target/'state').mkdir(parents=True,exist_ok=True)
        assert not (target/'state/resume.pt').exists()
        state['config']['cell_type']='LSTM'
        torch.save(state,target/'state/resume.pt')
        atomic_json(target/'state/history.json',state['history'])
        shutil.copy2(hw/'checkpoint.pt',target/'checkpoint.pt')
        metadata=json.loads((old/'checkpoint_metadata.json').read_text())
        metadata['config']['cell_type']='LSTM'
        metadata['checkpoint_sha256']=hashlib.sha256((target/'checkpoint.pt').read_bytes()).hexdigest()
        atomic_json(target/'checkpoint_metadata.json',metadata)
        atomic_json(root/'study.json',dict(seed=2,matched='dimensions, data, optimizer, scheduling, decoder, convergence',
                    embedding=128,hidden=512,batch_size=128,deadline=None,target_em=None,
                    models=['GRU','LSTM'],lstm_start_epoch=29,lstm_start_batch=830,
                    lstm_source_resume_sha256=hashlib.sha256(state_path.read_bytes()).hexdigest(),
                    original_lstm_untouched=True))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study-dir',required=True,type=Path)
    parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    root=args.study_dir.resolve();prepare(root)
    if args.prepare_only:return
    lock=(root/'.pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def state(phase,**more):atomic_json(root/'pipeline.json',dict(pid=os.getpid(),phase=phase,**more))
    def request_stop(signum,frame):(root/'STOP').touch()
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    state('starting')
    for kind in ['GRU','LSTM']:
        model_dir=root/kind.lower()
        done=model_dir/'verification.json'
        if done.exists() and json.loads(done.read_text()).get('converged'):continue
        if (root/'STOP').exists():state('paused');return
        env=dict(os.environ,PYTORCH_ENABLE_MPS_FALLBACK='0',MPLBACKEND='Agg')
        model_dir.mkdir(exist_ok=True)
        with (model_dir/'training.log').open('a') as log:
            child=subprocess.Popen([sys.executable,'-u',str(SCRIPTS/'train_recurrent.py'),
                                    '--cell-type',kind,'--run-dir',str(model_dir)],env=env,
                                   stdout=log,stderr=subprocess.STDOUT)
            state('training',cell_type=kind,child_pid=child.pid)
            while child.poll() is None:
                if (root/'STOP').exists():(model_dir/'STOP').touch()
                time.sleep(5)
            if child.returncode:
                state('failed',cell_type=kind,returncode=child.returncode)
                raise RuntimeError(f'{kind} process failed; inspect its training.log. No automatic reinitialization.')
        process=json.loads((model_dir/'process.json').read_text())
        if process['status']!='converged':state('paused',cell_type=kind,reason=process.get('stop_reason'));return
        verify_study(root)
    state('analyzing')
    subprocess.run([sys.executable,str(SCRIPTS/'analyze_recurrent_study.py'),'--study-dir',str(root)],check=True)
    verify_study(root);state('complete',submission_unchanged=True)

if __name__=='__main__':main()
