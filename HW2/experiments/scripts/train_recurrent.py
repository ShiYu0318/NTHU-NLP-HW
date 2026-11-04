"""Run an isolated GRU/LSTM study. Coding assistance: OpenAI Codex."""
import argparse
import fcntl
import hashlib
import json
import os
import platform
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from recurrent_study import make_driver, make_model, study_hw2, verify_study


def atomic_json(path, payload):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(payload,indent=2));temp.replace(path)


def evaluate(model, frame, output):
    started=time.monotonic();correct=0;rows=[];groups={}
    with torch.inference_mode():
        for row in frame.itertuples(index=False):
            answer=''.join(model.generator(row.src,max_len=len(row.src)+16))[len(row.src):]
            matched=int(answer==row.tgt);correct+=matched
            rows.append(dict(src=row.src,target=row.tgt,prediction=answer,correct=matched))
            for name in [f'answer_length_{len(row.tgt)}','multiplication' if '*' in row.src else 'addition_subtraction',
                         'parentheses' if '(' in row.src else 'no_parentheses']:
                group=groups.setdefault(name,dict(correct=0,total=0));group['correct']+=matched;group['total']+=1
    pd.DataFrame(rows).to_csv(output,index=False)
    return dict(correct=correct,total=len(frame),exact_match=correct/len(frame),seconds=time.monotonic()-started,groups=groups)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cell-type',required=True,choices=['GRU','LSTM'])
    parser.add_argument('--run-dir',required=True,type=Path)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    root=args.run_dir.resolve()
    hw=study_hw2(root.parent)
    if root.name!=args.cell_type.lower():parser.error('Run directory name must match the cell type.')
    data=hw/'data';diagnostics=hw/'experiments/diagnostics'
    if not torch.backends.mps.is_available():raise RuntimeError('This experiment requires a real MPS GPU.')
    root.mkdir(parents=True,exist_ok=True)
    lock=(root/'.training.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    frozen=root.parent/'frozen_submission.json'
    if not frozen.is_file():raise FileNotFoundError('Prepare the study frozen-submission manifest before training.')
    verify_study(root.parent)
    driver=make_driver(args.cell_type,smoke_epochs=2 if args.smoke else None)
    source=json.dumps(driver,ensure_ascii=False,indent=1)+'\n'
    snapshot=root/'source_notebook.ipynb'
    if snapshot.exists() and snapshot.read_text()!=source:raise ValueError('Immutable source snapshot differs; use a new run.')
    snapshot.write_text(source)
    torch.set_num_threads(4);torch.manual_seed(2)
    os.environ['HW2_DATA_DIR']=str(data);os.environ['HW2_OUTPUT_DIR']=str(root)
    def protect():verify_study(root.parent)
    ns=dict(cell_type=args.cell_type,answer_order='normal',invocation_id='recurrent-comparison',verify_protected_submission=protect)
    def execute(index):exec(compile(''.join(driver['cells'][index]['source']),f'study:cell_{index}','exec'),ns)
    started=datetime.now().astimezone().isoformat();clock=time.monotonic()
    atomic_json(root/'process.json',dict(pid=os.getpid(),status='initializing',started=started,cell_type=args.cell_type))
    for index in [1,2]:execute(index)
    if args.smoke:
        ns['df_train']=ns['df_train'].head(256).copy();ns['df_eval']=ns['df_eval'].head(16).copy()
    for index in [3,5,7,9]:execute(index)
    ns.update(batch_size=128,embed_dim=128,hidden_dim=512,target_em=None)
    for index in [11,12,13,15]:execute(index)
    ns['device']=torch.device('mps');ns['model']=ns['CharRNN'](ns['vocab_size'],128,512)
    execute(17)
    assert ns['seconds_remaining']()==float('inf')
    assert all(v.get('step',torch.tensor(0)).device.type=='cpu' for v in ns['optimizer'].state.values())
    experiment=dict(cell_type=args.cell_type,purpose='GPU smoke only' if args.smoke else 'Fixed-dimension convergence comparison',
                    config=ns['config'],parameter_count=sum(p.numel() for p in ns['model'].parameters()),
                    source_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),deadline=None,target_em=None,
                    epochs_limit=2 if args.smoke else None,train_examples=len(ns['df_train']),eval_examples=len(ns['df_eval']),
                    python=platform.python_version(),pytorch=str(torch.__version__),device='Apple MPS',
                    started=started,resumed_epoch=ns['start_epoch'],resumed_batch=ns['resume_batch'])
    atomic_json(root/'experiment.json',experiment)
    atomic_json(root/'process.json',dict(pid=os.getpid(),status='training',started=started,cell_type=args.cell_type))
    def interrupt(signum,frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupt)
    execute(19)
    protect()
    outcome=json.loads((root/'state/outcome.json').read_text())
    result=dict(cell_type=args.cell_type,started=started,invocation_seconds=time.monotonic()-clock,
                stop_reason=outcome['stop_reason'],converged=outcome['convergence']['converged'],
                finished=datetime.now().astimezone().isoformat(),submission_unchanged=True)
    if result['converged'] or args.smoke:
        model=make_model(driver,ns['config']).eval()
        model.load_state_dict(torch.load(root/'checkpoint.pt',map_location='cpu',weights_only=True))
        result['evaluation']=evaluate(model,ns['df_eval'],root/'evaluation_predictions.csv')
        assert result['evaluation']['exact_match']==ns['best_em']
        if not args.smoke:
            frame=pd.read_csv(diagnostics/'synthetic_three_operand_holdout.csv',dtype=str)
            result['diagnostic']=evaluate(model,frame,root/'diagnostic_predictions.csv')
        result['checkpoint_sha256']=hashlib.sha256((root/'checkpoint.pt').read_bytes()).hexdigest()
        torch.save(model.state_dict(),root/'checkpoint_cpu.pt')
        protect();atomic_json(root/'verification.json',result)
    atomic_json(root/'process.json',dict(pid=os.getpid(),status='converged' if result['converged'] else 'smoke_complete' if args.smoke else 'paused',**result))
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
