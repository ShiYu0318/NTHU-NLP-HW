"""Continue the existing scored run with the user's new convergence stop rule.

Generative AI assistance: OpenAI Codex authored this local process supervisor.
It observes actual process state, resumes only after the legacy score stop, and
never restarts merely because a log or observation timed out.
"""
import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from paths import AUDITS, LOGS, PILOTS, RUNS, SCRIPTS, WORKSPACE
from zoneinfo import ZoneInfo

ROOT = WORKSPACE
RUN = RUNS / 'mps_e128_h512_seed2'
LOG = RUNS / 'mps_e128_h512_seed2/training.log'
DEADLINE = datetime(2026, 11, 4, 22, 30, tzinfo=ZoneInfo('Asia/Taipei'))
STATE = AUDITS / 'continuation_state.json'
FORCED_CUTOFF = set()


def remaining():
    return (DEADLINE - datetime.now(ZoneInfo('Asia/Taipei'))).total_seconds()


def alive(pid):
    result = subprocess.run(['ps', '-p', str(pid), '-o', 'stat=,args='], capture_output=True, text=True)
    fields = result.stdout.strip().split(maxsplit=1)
    if len(fields) != 2 or fields[0].startswith('Z'):
        return False
    try:
        tokens = shlex.split(fields[1])
    except ValueError:
        return False
    return (bool(tokens) and Path(tokens[0]).name.lower().startswith('python')
            and any(Path(token).name == 'train_mps.py' for token in tokens[1:]))


def resume_after_legacy_stop(reason, seconds):
    return reason in {'target full-eval EM reached', 'configured epochs completed'} and seconds > 390


def last_reason(since=0):
    with LOG.open('rb') as stream:
        stream.seek(max(since, LOG.stat().st_size - 65536))
        lines = stream.read().decode('utf-8', errors='replace').splitlines()
    values = [line.split('Stop reason:', 1)[1].strip() for line in lines if line.startswith('Stop reason:')]
    return values[-1] if values else None


def invocation_offset():
    content = LOG.read_bytes()
    marker = content.rfind(b'Experiment started:')
    return marker if marker >= 0 else len(content)


def terminal_reason(pid, since):
    if pid in FORCED_CUTOFF:
        return 'hard 22:30 cutoff'
    outcome_path = RUN / 'runs/lstm_e128_h512_seed2/outcome.json'
    if outcome_path.is_file():
        outcome = json.loads(outcome_path.read_text())
        if outcome.get('pid') == pid:
            return outcome['stop_reason']
    return last_reason(since)


def valid_final_reason(reason):
    return reason is not None and (reason.startswith('operational convergence') or
        reason.startswith('insufficient time') or reason.startswith('deadline') or
        reason == 'hard 22:30 cutoff')


def record(phase, **kwargs):
    payload = dict(phase=phase, supervisor_pid=os.getpid(), deadline=DEADLINE.isoformat(),
                   updated=datetime.now(ZoneInfo('Asia/Taipei')).isoformat(), score_stop_disabled_in_continuation=True, **kwargs)
    temporary = STATE.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(STATE)
    print(json.dumps(payload), flush=True)


def wait_for(pid):
    while alive(pid):
        if (RUN / 'STOP_AUTOCONTINUE').exists():
            record('automatic_continuation_disabled', observed_pid=pid)
            return False
        if remaining() <= 0:
            record('hard_deadline_cutoff', observed_pid=pid)
            FORCED_CUTOFF.add(pid)
            if alive(pid):
                os.kill(pid, signal.SIGTERM)
            for _ in range(20):
                if not alive(pid):
                    break
                time.sleep(0.1)
            if alive(pid):
                os.kill(pid, signal.SIGKILL)
            return True
        time.sleep(min(5, max(0.1, remaining())))
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--existing-pid', required=True, type=int)
    parser.add_argument('--log-offset', type=int)
    args = parser.parse_args()
    active_pid = args.existing_pid
    active_offset = invocation_offset() if args.log_offset is None else args.log_offset
    record('observing_existing_training', observed_pid=active_pid, log_offset=active_offset)
    if not wait_for(args.existing_pid):
        return
    reason = terminal_reason(active_pid, active_offset)
    record('existing_training_terminal', observed_pid=args.existing_pid, stop_reason=reason)
    if resume_after_legacy_stop(reason, remaining()):
        resume = RUN / 'runs/lstm_e128_h512_seed2/resume.pt'
        if not resume.is_file():
            raise RuntimeError('Complete optimizer/scheduler resume state is missing; refusing to reinitialize.')
        env = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK='1', MPLBACKEND='Agg')
        with LOG.open('a') as training_log:
            active_offset = LOG.stat().st_size
            child = subprocess.Popen([sys.executable, '-u', str(SCRIPTS / 'train_mps.py'),
                                      '--epochs', '1000000'], cwd=ROOT, env=env,
                                     stdout=training_log, stderr=subprocess.STDOUT)
            active_pid = child.pid
            record('convergence_training_running', child_pid=child.pid, log_offset=active_offset, resumed_from=str(resume))
            with (LOGS / 'checkpoint_archive.log').open('a') as archive_log:
                observer = subprocess.Popen([sys.executable, '-u', str(SCRIPTS / 'watch_checkpoints.py'),
                                             '--pid', str(child.pid), '--run-dir', str(RUN)],
                                            cwd=ROOT, stdout=archive_log, stderr=subprocess.STDOUT)
                if not wait_for(child.pid):
                    return
                returncode = child.wait()
                observer.wait(timeout=30)
        reason = terminal_reason(child.pid, active_offset)
        record('convergence_training_terminal', child_pid=child.pid, returncode=returncode, stop_reason=reason)
        if (returncode != 0 and child.pid not in FORCED_CUTOFF) or (reason and reason.startswith('training error:')):
            raise RuntimeError('Continuation failed; inspect the live log before recovery.')
    elif reason in {'target full-eval EM reached', 'configured epochs completed'}:
        reason = 'insufficient time to resume before 22:30'
    if not valid_final_reason(reason):
        record('unexpected_training_terminal', observed_pid=active_pid, stop_reason=reason)
        raise RuntimeError('Training stopped without a process-bound valid final reason; refusing automatic finalization.')
    terminal = dict(phase='terminal', pid=active_pid, stop_reason=reason, log_offset=active_offset,
                    finished=datetime.now(ZoneInfo('Asia/Taipei')).isoformat(), forced_cutoff=active_pid in FORCED_CUTOFF)
    terminal_tmp = RUN / 'primary_terminal.tmp'
    terminal_tmp.write_text(json.dumps(terminal, indent=2))
    terminal_tmp.replace(RUN / 'primary_terminal.json')
    if (RUN / 'STOP_AUTOCONTINUE').exists():
        record('automatic_continuation_disabled')
        return
    # Finish only the pending small report fit when the primary GPU job is
    # terminal and the observed control runtime leaves a safe time budget.
    pilot_root = PILOTS / 'report_experiments'
    two_result = pilot_root / 'lstm_train2_test2/result.json'
    three_result = pilot_root / 'lstm_train3_test2/result.json'
    if two_result.is_file() and not three_result.is_file():
        previous_two = json.loads(two_result.read_text())
        required_seconds = 90  # Small pending pilot now evaluates on CPU; its GPU loop stops with a 40s reserve.
        if remaining() > required_seconds:
            env = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK='0', MPLBACKEND='Agg')
            with (LOGS / 'report_experiments.log').open('a') as pilot_log:
                pilot = subprocess.Popen([sys.executable, '-u', str(SCRIPTS / 'run_report_experiments.py'),
                                          '--epochs', '10', '--only-shift'], cwd=ROOT, env=env,
                                         stdout=pilot_log, stderr=subprocess.STDOUT)
                record('completing_pending_report_fit', report_pid=pilot.pid, primary_stop_reason=reason)
                try:
                    pilot_code = pilot.wait(timeout=max(1, remaining()))
                except subprocess.TimeoutExpired:
                    pilot.terminate()
                    try:
                        pilot.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pilot.kill()
                        pilot.wait()
                    pilot_code = -15
                record('report_fit_terminal', returncode=pilot_code, primary_stop_reason=reason)
    if (RUN / 'STOP_AUTOCONTINUE').exists():
        record('automatic_continuation_disabled')
        return
    # Final packaging is independently verified; no training is performed here.
    record('verifying_final_checkpoint', stop_reason=reason)
    with (LOGS / 'final_prepare.log').open('w') as artifact_log:
        subprocess.run([sys.executable, str(SCRIPTS / 'prepare_submission.py'),
                        '--run-dir', str(RUN), '--publish', '--include-diagnostic', '--select-late-candidates'],
                       cwd=ROOT, stdout=artifact_log, stderr=subprocess.STDOUT, check=True)
    record('artifacts_verified', stop_reason=reason)


if __name__ == '__main__':
    main()
