"""Local Apple GPU experiment using the submission notebook's actual model/cells.

Generative AI assistance: OpenAI Codex authored this experiment launcher.
This is an experiment helper, not part of the required submission ZIP.
The notebook's Colab CUDA guard remains unchanged; this launcher explicitly
selects a real Apple GPU for the separate, accurately labelled MPS experiment.
"""
import argparse
import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path
from paths import DATA, RUNS, TEMPLATES, WORKSPACE
from zoneinfo import ZoneInfo

import torch

workspace = WORKSPACE
notebook_path = TEMPLATES / 'training_driver.json'
parser = argparse.ArgumentParser()
parser.add_argument('--answer-order', choices=['normal', 'least_significant_digit_first'], default='normal')
parser.add_argument('--epochs', type=int, default=1000000)
parser.add_argument('--target-em', type=float, default=None, help='Optional score stop; disabled by default')
args = parser.parse_args()
if args.target_em is not None and not 0 < args.target_em <= 1:
    parser.error('--target-em must be in (0, 1]')
run_name = 'mps_e128_h512_seed2' + ('_lsf' if args.answer_order != 'normal' else '')
run_root = RUNS / run_name
run_root.mkdir(parents=True, exist_ok=True)
os.environ['HW2_DATA_DIR'] = str(DATA)
os.environ['HW2_OUTPUT_DIR'] = str(run_root)

if not torch.backends.mps.is_available():
    raise RuntimeError('This experiment requires the Apple GPU (MPS).')
torch.set_num_threads(4)
torch.manual_seed(2)
notebook = json.loads(notebook_path.read_text())
snapshot_name = 'source_notebook_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.ipynb'
(run_root / snapshot_name).write_text(json.dumps(notebook, ensure_ascii=False, indent=1))
active_tmp = run_root / 'active_invocation.tmp'
active_tmp.write_text(json.dumps(dict(pid=os.getpid(), invocation_id=snapshot_name)))
active_tmp.replace(run_root / 'active_invocation.json')
ns = {'answer_order': args.answer_order, 'invocation_id': snapshot_name}
def execute_cell(index):
    source = ''.join(notebook['cells'][index]['source'])
    exec(compile(source, f'{notebook_path.name}:cell_{index}', 'exec'), ns)

print('Experiment started:', datetime.now(ZoneInfo('Asia/Taipei')).isoformat(), flush=True)
print('Device: Apple M5 GPU via MPS; CUDA is not used in this experiment.', flush=True)
for index in [1, 2, 3, 5, 7, 9]:
    execute_cell(index)
ns.update(batch_size=128, epochs=args.epochs, embed_dim=128, hidden_dim=512, target_em=args.target_em)
for index in [11, 12, 13, 15]:
    execute_cell(index)
ns['device'] = torch.device('mps')
ns['model'] = ns['CharRNN'](ns['vocab_size'], ns['embed_dim'], ns['hidden_dim'])
(run_root / 'experiment.json').write_text(json.dumps(dict(
    device='Apple M5 GPU (MPS)', system=platform.platform(), python=platform.python_version(),
    pytorch=str(torch.__version__), data_storage=ns.get('data_storage', 'Python lists'), batch_size=128, epochs_limit=args.epochs, answer_order=args.answer_order,
    embed_dim=128, hidden_dim=512, target_full_eval_em=args.target_em,
    train_examples=len(ns['df_train']), eval_examples=len(ns['df_eval']),
    deadline='2026-11-04T22:30:00+08:00', seed=2,
    source_notebook=snapshot_name), indent=2))
execute_cell(17)
print('Starting full-data GPU training.', flush=True)
execute_cell(19)
print('Experiment finished:', datetime.now(ZoneInfo('Asia/Taipei')).isoformat(), flush=True)
