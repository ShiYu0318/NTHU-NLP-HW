"""Independent post-submission continuation; OpenAI Codex coding assistance."""
import argparse
import fcntl
import hashlib
import json
import os
import platform
import signal
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from paths import CONTINUATION_RUN, DATA, DIAGNOSTICS, verify_submission

parser = argparse.ArgumentParser(description='Resume the independent continuation, explicitly and without publishing homework.')
parser.add_argument('--check-paths', action='store_true', help='Check dependencies and submission hashes without loading torch or starting training')
parser.add_argument('--resume', action='store_true', help='Explicitly clear a saved STOP request and resume')
args = parser.parse_args()
ROOT = CONTINUATION_RUN

def verify_protected_submission():
    verify_submission(ROOT / 'protected_submission_sha256.json')

if args.check_paths:
    required = [DATA / 'arithmetic_train.csv', DATA / 'arithmetic_eval.csv',
                DIAGNOSTICS / 'synthetic_three_operand_holdout.csv',
                ROOT / 'training_driver.ipynb', ROOT / 'checkpoint.pt',
                ROOT / 'runs/lstm_e128_h512_seed2/resume.pt']
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    verify_protected_submission()
    print(json.dumps(dict(training_started=False, paused=(ROOT / 'STOP').exists(),
                          run_dir=str(ROOT), dependencies=[str(p) for p in required]), indent=2))
    raise SystemExit(0)
if (ROOT / 'STOP').exists() and not args.resume:
    parser.error('Training is paused. Pass --resume only when you intend to restart training.')

import torch
lock = (ROOT / '.training.lock').open('w')
try:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    raise SystemExit('Another training process already holds the experiment lock.')

verify_protected_submission()
if not torch.backends.mps.is_available():
    raise RuntimeError('Apple MPS GPU is required for this experiment.')
if args.resume:
    (ROOT / 'STOP').unlink(missing_ok=True)
torch.set_num_threads(4)
torch.manual_seed(2)
os.environ['HW2_DATA_DIR'] = str(DATA)
os.environ['HW2_OUTPUT_DIR'] = str(ROOT)

def interrupt(signum, frame):
    raise KeyboardInterrupt

signal.signal(signal.SIGTERM, interrupt)
started = datetime.now(ZoneInfo('Asia/Taipei')).isoformat()
(ROOT / 'process.json').write_text(json.dumps(dict(pid=os.getpid(), started=started, deadline=None,
    target_em=None, status='starting', python=sys.executable), indent=2))
print('Independent continuation started:', started, '| PID:', os.getpid(), flush=True)
notebook = json.loads((ROOT / 'training_driver.ipynb').read_text())
ns = dict(answer_order='normal', invocation_id='independent_post_submission',
          verify_protected_submission=verify_protected_submission)

def execute(index):
    exec(compile(''.join(notebook['cells'][index]['source']),f'training_driver.ipynb:cell_{index}','exec'),ns)

for index in [1,2,3,5,7,9]:
    execute(index)
ns.update(batch_size=128, embed_dim=128, hidden_dim=512, target_em=None)
for index in [11,12,13,15]:
    execute(index)
ns['device'] = torch.device('mps')
ns['model'] = ns['CharRNN'](ns['vocab_size'],ns['embed_dim'],ns['hidden_dim'])
execute(17)
assert ns['start_epoch'] >= 26 and ns['resume_batch'] >= 0
assert ns['loaded_checkpoint'] and ns['target_em'] is None
assert ns['seconds_remaining']() == float('inf') and ns['can_start_epoch'](-1)
print('Restored Adam steps:',sorted({int(v['step']) for v in ns['optimizer'].state.values()}),flush=True)
print('Starting epoch:',ns['start_epoch'],'| LR:',ns['optimizer'].param_groups[0]['lr'],flush=True)
(ROOT / 'process.json').write_text(json.dumps(dict(pid=os.getpid(), started=started, deadline=None,
    target_em=None,status='training',resumed_epoch=ns['start_epoch']),indent=2))
execute(19)
verify_protected_submission()

# Independent saved-weight verification after the training loop terminates.
import pandas as pd
model = ns['CharRNN'](ns['vocab_size'],ns['embed_dim'],ns['hidden_dim']).eval()
model.load_state_dict(torch.load(ROOT / 'checkpoint.pt',map_location='cpu',weights_only=True))
results = {}
for name, path in [('evaluation',DATA/'arithmetic_eval.csv'),
                   ('synthetic_diagnostic',DIAGNOSTICS/'synthetic_three_operand_holdout.csv')]:
    frame = pd.read_csv(path,dtype=str)
    matched = 0
    with torch.inference_mode():
        for row in frame.itertuples(index=False):
            prediction = ''.join(model.generator(row.src,max_len=len(row.src)+16))[len(row.src):]
            matched += int(prediction == row.tgt)
    results[name] = dict(correct=matched,total=len(frame),exact_match=matched/len(frame))
    print('Independent best-checkpoint verification:',name,results[name],flush=True)
assert results['evaluation']['exact_match'] == ns['best_em']
torch.save(model.state_dict(),ROOT/'checkpoint_cpu.pt')
verify_protected_submission()
results.update(submission_unchanged=True,best_em=ns['best_em'],stop_reason=ns['stop_reason'],
    finished=datetime.now(ZoneInfo('Asia/Taipei')).isoformat(),
    checkpoint_sha256=hashlib.sha256((ROOT/'checkpoint.pt').read_bytes()).hexdigest())
(ROOT/'final_verification.json').write_text(json.dumps(results,indent=2))
(ROOT/'process.json').write_text(json.dumps(dict(pid=os.getpid(),started=started,status='finished',
    finished=results['finished'],stop_reason=ns['stop_reason']),indent=2))
print('Independent continuation verified. Submitted homework unchanged.',flush=True)
