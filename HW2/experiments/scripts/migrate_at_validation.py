"""Checkpoint-safe migration at a read-only validation boundary.
Generative AI assistance: OpenAI Codex authored this process controller.
No signal is sent while an optimizer update could be in progress.
"""
import argparse
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from paths import AUDITS, LOGS, RUNS, SCRIPTS, TEMPLATES, WORKSPACE

from continue_until_converged import alive, remaining

ROOT = WORKSPACE
RUN = RUNS / 'mps_e128_h512_seed2'
LOG = RUNS / 'mps_e128_h512_seed2/training.log'
STATE = AUDITS / 'migration_state.json'


def record(phase, **values):
    value = dict(phase=phase, **values)
    temp = STATE.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(STATE)
    print(json.dumps(value), flush=True)


def validation_boundary(line):
    match = re.match(r'^Validation epoch (\d+):.*?\b(\d+)/10000\b', line)
    if match and 300 <= int(match[2]) <= 6000:
        return int(match[1])
    return None


def last_line():
    with LOG.open('rb') as stream:
        stream.seek(max(0, LOG.stat().st_size - 16384))
        lines = stream.read().decode('utf-8', errors='replace').splitlines()
    return lines[-1] if lines else ''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pid', required=True, type=int)
    args = parser.parse_args()
    proof = json.loads((AUDITS / 'compact_data_verification.json').read_text())
    notebook = json.loads((TEMPLATES / 'training_driver.json').read_text())
    digest = hashlib.sha256((''.join(notebook['cells'][7]['source']) + '\n' + ''.join(notebook['cells'][11]['source'])).encode()).hexdigest()
    assert proof['examples'] == 1000000 and proof['all_token_labels_identical']
    assert digest == proof['preprocessing_sha256']
    record('waiting_for_read_only_validation', observed_pid=args.pid)
    epoch = None
    while alive(args.pid):
        if (RUN / 'STOP_AUTOCONTINUE').exists():
            record('cancelled_without_signal')
            return
        if remaining() <= 120:
            raise RuntimeError('No safe migration time remains; preserve current training and best checkpoint.')
        epoch = validation_boundary(last_line())
        if epoch is not None:
            resume = RUN / 'runs/lstm_e128_h512_seed2/resume.pt'
            shutil.copy2(resume, RUN / 'pre_migration_resume.pt')
            os.kill(args.pid, signal.SIGINT)
            record('interrupted_read_only_validation', observed_pid=args.pid, epoch=epoch)
            break
        time.sleep(.1)
    if epoch is None:
        raise RuntimeError('Trainer became terminal before a safe migration boundary; inspect its actual outcome.')
    while alive(args.pid):
        if remaining() <= 0:
            os.kill(args.pid, signal.SIGTERM)
            raise RuntimeError('Deadline reached waiting for checkpoint flush; previous durable checkpoint preserved.')
        time.sleep(.2)
    import torch
    resume = RUN / 'runs/lstm_e128_h512_seed2/resume.pt'
    state = torch.load(resume, map_location='cpu', weights_only=True)
    expected_batches = math.ceil(1000000 / state['config']['batch_size'])
    assert state['next_epoch'] == epoch and state['next_batch'] == expected_batches, 'Not a completed-training boundary; do not resume ambiguous state.'
    assert state['config']['char_to_id'] == proof['vocabulary']
    steps = [float(value['step']) for value in state['optimizer']['state'].values()]
    assert min(steps) == max(steps) and min(steps) > 0
    learning_rates = [group['lr'] for group in state['optimizer']['param_groups']]
    del state
    offset = LOG.stat().st_size
    environment = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK='1', MPLBACKEND='Agg')
    with LOG.open('a') as log:
        trainer = subprocess.Popen([sys.executable, '-u', str(SCRIPTS / 'train_mps.py'), '--epochs', '1000000'],
                                   cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    with (LOGS / 'checkpoint_archive.log').open('a') as log:
        observer = subprocess.Popen([sys.executable, '-u', str(SCRIPTS / 'watch_checkpoints.py'),
                                     '--pid', str(trainer.pid), '--run-dir', str(RUN)],
                                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    with (LOGS / 'continuation_supervisor.log').open('a') as log:
        supervisor = subprocess.Popen([sys.executable, '-u', str(SCRIPTS / 'continue_until_converged.py'),
                                       '--existing-pid', str(trainer.pid), '--log-offset', str(offset)],
                                      cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    record('migration_launched', previous_pid=args.pid, trainer_pid=trainer.pid, supervisor_pid=supervisor.pid,
           observer_pid=observer.pid, resume_epoch=epoch, resume_batch=expected_batches,
           optimizer_steps_restored=steps[0], learning_rates=learning_rates,
           input_label_equivalence_sha256=proof['token_label_sha256'], log_offset=offset)


if __name__ == '__main__':
    main()
