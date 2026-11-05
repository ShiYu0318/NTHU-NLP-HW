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


def prepare(root, lstm_start=None):
    hw=study_hw2(root);root.mkdir(parents=True,exist_ok=True)
    frozen=root/'frozen_submission.json'
    if not frozen.exists():
        names=['NLP_HW2_NCU_114502540.ipynb','NLP_HW2_NCU_114502540.zip','checkpoint.pt','requirements.txt',
               'Assignment2/main.ipynb','data/arithmetic_train.csv','data/arithmetic_eval.csv']
        atomic_json(frozen,{name:hashlib.sha256((hw/name).read_bytes()).hexdigest() for name in names})
    verify_study(root)
    metadata=root/'study.json'
    if metadata.exists():
        mode=json.loads(metadata.read_text()).get('lstm_initialization','resume')
        if lstm_start is not None and lstm_start!=mode:
            raise ValueError('Cannot change initialization mode of an existing study.')
        return
    mode=lstm_start or 'resume'
    if mode not in {'fresh','resume'}:raise ValueError('Unknown LSTM initialization mode')
    target=root/'lstm';target.mkdir(exist_ok=True)
    start_epoch=1;start_batch=0;source_digest=None
    if mode=='resume':
        old=hw/'experiments/runs/continuation_20261104_230700'
        state_path=old/'runs/lstm_e128_h512_seed2/resume.pt'
        if not state_path.exists():
            raise FileNotFoundError('Local LSTM resume state is missing. Use --lstm-start fresh to reproduce both models from scratch.')
        state=torch.load(state_path,map_location='cpu',weights_only=True)
        assert state['config']['embed_dim']==128 and state['config']['hidden_dim']==512
        (target/'state').mkdir(exist_ok=True)
        assert not (target/'state/resume.pt').exists()
        state['config']['cell_type']='LSTM';start_epoch=state['next_epoch'];start_batch=state['next_batch']
        torch.save(state,target/'state/resume.pt');atomic_json(target/'state/history.json',state['history'])
        shutil.copy2(hw/'checkpoint.pt',target/'checkpoint.pt')
        selected=json.loads((old/'checkpoint_metadata.json').read_text());selected['config']['cell_type']='LSTM'
        selected['checkpoint_sha256']=hashlib.sha256((target/'checkpoint.pt').read_bytes()).hexdigest()
        atomic_json(target/'checkpoint_metadata.json',selected)
        source_digest=hashlib.sha256(state_path.read_bytes()).hexdigest()
    atomic_json(metadata,dict(seed=2,matched='dimensions, data, optimizer, scheduling, decoder, convergence',
                embedding=128,hidden=512,batch_size=128,deadline=None,target_em=None,models=['GRU','LSTM'],
                lstm_initialization=mode,lstm_start_epoch=start_epoch,lstm_start_batch=start_batch,
                lstm_source_resume_sha256=source_digest,original_lstm_untouched=True))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study-dir',required=True,type=Path)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--resume',action='store_true',help='Explicitly clear study/model STOP markers')
    parser.add_argument('--lstm-start',choices=['resume','fresh'],default=None,help='Fresh needs no local optimizer backup; an existing study keeps its saved mode')
    args=parser.parse_args()
    root=args.study_dir.resolve();prepare(root,lstm_start=args.lstm_start)
    if args.prepare_only:return
    lock=(root/'.pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.resume:
        for marker in [root/'STOP',root/'gru/STOP',root/'lstm/STOP']:marker.unlink(missing_ok=True)
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
