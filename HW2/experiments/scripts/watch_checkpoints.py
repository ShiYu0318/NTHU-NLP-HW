"""Archive stable best checkpoints without changing the training process.

Generative AI assistance: OpenAI Codex authored this local file observer.
"""
import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--pid', type=int, required=True)
parser.add_argument('--run-dir', type=Path, required=True)
args = parser.parse_args()
run = args.run_dir.resolve()
archive = run / 'checkpoint_archive'
archive.mkdir(exist_ok=True)
previous = None
while True:
    status = subprocess.run(['ps', '-p', str(args.pid), '-o', 'stat='], capture_output=True, text=True).stdout.strip()
    alive = bool(status) and not status.startswith('Z')
    try:
        metadata_bytes = (run / 'checkpoint_metadata.json').read_bytes()
        stat = (run / 'checkpoint.pt').stat()
        identity = (metadata_bytes, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if identity == previous or not alive:
            metadata = json.loads(metadata_bytes)
            path = archive / f"epoch_{metadata['epoch']:03d}.pt"
            if not path.exists():
                payload = (run / 'checkpoint.pt').read_bytes()
                if (run / 'checkpoint_metadata.json').read_bytes() == metadata_bytes:
                    digest = hashlib.sha256(payload).hexdigest()
                    writer_digest = metadata.get('checkpoint_sha256')
                    if writer_digest is not None and writer_digest != digest:
                        previous = None
                        time.sleep(1)
                        continue
                    audit_path = run / 'final_verification.json'
                    audit = json.loads(audit_path.read_text()) if audit_path.is_file() else {}
                    metadata['binding_verified'] = (writer_digest == digest or
                        (audit.get('checkpoint_sha256') == digest and audit.get('full_evaluation')
                         and audit.get('correct') == round(float(metadata['exact_match']) * 10000)))
                    path.write_bytes(payload)
                    metadata['checkpoint_sha256'] = digest
                    path.with_suffix('.json').write_text(json.dumps(metadata, indent=2))
                    print('Archived:', path.name, metadata['exact_match'], flush=True)
        previous = identity
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    if not alive:
        print('Observed training process terminal; archive complete.', flush=True)
        break
    time.sleep(1)
