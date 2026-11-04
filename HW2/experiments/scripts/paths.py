"""Paths relative to this HW2 checkout, shared by experiment entry points."""
import hashlib
import json
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
EXPERIMENTS = SCRIPTS.parent
HW2 = EXPERIMENTS.parent
WORKSPACE = HW2.parent
DATA = HW2 / 'data'
TEMPLATES = EXPERIMENTS / 'templates'
RUNS = EXPERIMENTS / 'runs'
PILOTS = EXPERIMENTS / 'pilots'
DIAGNOSTICS = EXPERIMENTS / 'diagnostics'
LOGS = EXPERIMENTS / 'logs'
AUDITS = EXPERIMENTS / 'audits'
DRIVER = TEMPLATES / 'training_driver.json'
ORIGINAL_NOTEBOOK = TEMPLATES / 'assignment_original.ipynb'
MAIN_RUN = RUNS / 'mps_e128_h512_seed2'
CONTINUATION_RUN = RUNS / 'continuation_20261104_230700'


def resolve_recorded_path(value):
    """Resolve a current path or a historical location without editing evidence."""
    path = Path(value)
    if not path.is_absolute():
        return HW2 / path
    mapping = AUDITS / 'relocation_map.json'
    if mapping.is_file():
        prefixes = json.loads(mapping.read_text())['prefixes']
        for old, new in sorted(prefixes.items(), key=lambda item:len(item[0]), reverse=True):
            try:
                suffix = path.relative_to(old)
            except ValueError:
                continue
            return HW2 / new / suffix
    return path


def verify_submission(manifest):
    """Validate the current submission using HW2-relative filenames."""
    for name, digest in json.loads(Path(manifest).read_text()).items():
        path = resolve_recorded_path(name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Submitted artifact has changed: {path}')
