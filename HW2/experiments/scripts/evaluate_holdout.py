"""Evaluate a saved checkpoint on a disjoint synthetic diagnostic set.

Generative AI assistance: OpenAI Codex authored this inference-only helper.
This diagnostic is not the official private leaderboard and never trains a model.
"""
import argparse
import ast
import hashlib
import io
import json
import re
import time
from pathlib import Path
from paths import DIAGNOSTICS, TEMPLATES, WORKSPACE

import pandas as pd
import torch

ROOT = WORKSPACE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    metadata_path = run_dir / 'checkpoint_metadata.json'
    metadata_bytes = metadata_path.read_bytes()
    metadata = json.loads(metadata_bytes)
    checkpoint_bytes = (run_dir / 'checkpoint.pt').read_bytes()
    if metadata_bytes != metadata_path.read_bytes():
        raise RuntimeError('Checkpoint selection changed while loading; retry.')
    digest = hashlib.sha256(checkpoint_bytes).hexdigest()
    writer_binding = (metadata.get('checkpoint_sha256') == digest and metadata.get('binding_verified', True))
    audit_path = run_dir / 'final_verification.json'
    audit = json.loads(audit_path.read_text()) if audit_path.is_file() else {}
    independent_binding = (audit.get('checkpoint_sha256') == digest and audit.get('full_evaluation')
                           and audit.get('total') == 10000
                           and audit.get('correct') == round(float(metadata['exact_match']) * 10000))
    if not writer_binding and not independent_binding:
        raise RuntimeError('Unbound legacy checkpoint: run prepare_submission.py full verification first.')
    config = metadata['config']
    order = config.get('answer_order', 'normal')
    driver = json.loads((TEMPLATES / 'training_driver.json').read_text())
    ns = dict(torch=torch, answer_order=order, char_to_id=config['char_to_id'],
              id_to_char={i: c for c, i in config['char_to_id'].items()})
    for node in ast.parse(''.join(driver['cells'][7]['source'])).body:
        if isinstance(node, ast.FunctionDef) and node.name == 'reorder_answer_digits':
            exec(compile(ast.Module(body=[node], type_ignores=[]), 'answer_order', 'exec'), ns)
    exec(''.join(driver['cells'][15]['source']), ns)
    torch.set_num_threads(2)
    model = ns['CharRNN'](config['vocab_size'], config['embed_dim'], config['hidden_dim'])
    model.load_state_dict(torch.load(io.BytesIO(checkpoint_bytes), map_location='cpu', weights_only=True))
    model.eval()
    data_path = DIAGNOSTICS / 'synthetic_three_operand_holdout.csv'
    frame = pd.read_csv(data_path, dtype=str)
    assert len(frame) == 10000
    coverage_path = DIAGNOSTICS / 'equivalent_expression_coverage.csv'
    known_inputs = None
    if coverage_path.is_file():
        coverage = pd.read_csv(coverage_path)
        known_inputs = set(coverage[(coverage.dataset == 'synthetic_diagnostic') & (coverage.seen_equivalent_input.astype(str).str.lower() == 'true')].src)
    matched = 0
    groups = {}
    errors = []
    started = time.monotonic()
    with torch.inference_mode():
        for index, row in enumerate(frame.itertuples(index=False), 1):
            prediction = ''.join(model.generator(row.src, max_len=len(row.src) + 16))[len(row.src):]
            correct = int(prediction == row.tgt)
            matched += correct
            has_zero = '0' in re.findall(r'\d+', row.src)
            categories = [f'answer_length_{len(row.tgt)}', 'multiplication' if '*' in row.src else 'addition_subtraction',
                          'zero_operand' if has_zero else 'no_zero_operand']
            if known_inputs is not None:
                categories.append('seen_equivalent_input' if row.src in known_inputs else 'unseen_equivalent_input')
            for name in categories:
                counts = groups.setdefault(name, dict(correct=0, total=0))
                counts['correct'] += correct
                counts['total'] += 1
            if not correct and len(errors) < 50:
                errors.append(dict(src=row.src, target=row.tgt, prediction=prediction))
            if index % 1000 == 0:
                print(f'Diagnostic evaluation: {index}/10000', flush=True)
    result = dict(purpose='Synthetic disjoint diagnostic, not official private test',
                  checkpoint_sha256=hashlib.sha256(checkpoint_bytes).hexdigest(),
                  evaluation_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
                  epoch=metadata['epoch'], official_eval_em=metadata['exact_match'],
                  diagnostic_em=matched / len(frame), correct=matched, total=len(frame),
                  groups=groups, errors=errors, seconds=time.monotonic() - started)
    path = run_dir / f"diagnostic_epoch_{metadata['epoch']}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ['groups', 'errors']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
