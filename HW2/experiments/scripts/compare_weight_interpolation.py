"""CPU-only late-weight interpolation evaluation; never modifies primary training.
Generative AI assistance: OpenAI Codex authored this candidate comparison.
"""
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from paths import DATA, HW2, RUNS, WORKSPACE

import pandas as pd
import torch

ROOT = WORKSPACE
RUN = RUNS / 'mps_e128_h512_seed2'
output = RUN / ('weight_comparison_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
output.mkdir()
metadata = json.loads((RUN / 'checkpoint_metadata.json').read_text())
config = metadata['config']
os.link(RUN / 'checkpoint.pt', output / 'base_checkpoint.pt')
os.link(RUN / 'runs/lstm_e128_h512_seed2/resume.pt', output / 'recent_resume.pt')
base_hash = hashlib.sha256((output / 'base_checkpoint.pt').read_bytes()).hexdigest()
base = torch.load(output / 'base_checkpoint.pt', map_location='cpu', weights_only=True, mmap=True)
resume = torch.load(output / 'recent_resume.pt', map_location='cpu', weights_only=True, mmap=True)
recent = resume['model']
assert set(base) == set(recent)
assert all(base[key].shape == recent[key].shape and base[key].is_floating_point() for key in base)
assert resume['config']['char_to_id'] == config['char_to_id']
provenance = dict(base_epoch=metadata['epoch'], base_recorded_em=metadata['exact_match'],
                  base_sha256=base_hash, recent_epoch=resume['next_epoch'], recent_next_batch=resume['next_batch'])
notebook = json.loads((HW2 / 'NLP_HW2_NCU_114502540.ipynb').read_text())
ns = dict(torch=torch, char_to_id=config['char_to_id'], id_to_char={i: c for c, i in config['char_to_id'].items()})
exec(''.join(notebook['cells'][15]['source']), ns)
frame = pd.read_csv(DATA / 'arithmetic_eval.csv', dtype=str)
assert len(frame) == 10000
torch.set_num_threads(2)
results = []
for alpha in [.25, .5]:
    candidate = {key: torch.lerp(base[key], recent[key], alpha) for key in base}
    model = ns['CharRNN'](config['vocab_size'], config['embed_dim'], config['hidden_dim'])
    model.load_state_dict(candidate)
    model.eval()
    correct = 0
    predictions = []
    with torch.inference_mode():
        for index, row in enumerate(frame.itertuples(index=False), 1):
            answer = ''.join(model.generator(row.src, max_len=len(row.src) + 16))[len(row.src):]
            matched = answer == row.tgt
            correct += int(matched)
            predictions.append(dict(src=row.src, target=row.tgt, prediction=answer, correct=matched))
            if index % 2000 == 0:
                print('alpha=', alpha, 'verified=', index, flush=True)
    path = output / f'alpha_{alpha}'
    path.mkdir()
    torch.save(candidate, path / 'checkpoint.pt')
    record = dict(alpha=alpha, exact_match=correct / 10000, correct=correct, total=10000,
                  device='CPU', provenance=provenance, config=config,
                  checkpoint_sha256=hashlib.sha256((path / 'checkpoint.pt').read_bytes()).hexdigest())
    (path / 'evaluation.json').write_text(json.dumps(record, indent=2))
    pd.DataFrame(predictions).to_csv(path / 'predictions.csv', index=False)
    results.append(record)
    print(json.dumps({key: record[key] for key in ['alpha', 'exact_match', 'correct', 'total']}), flush=True)
(output / 'comparison.json').write_text(json.dumps(results, indent=2))
print('Candidate comparison complete; primary checkpoint and training unchanged:', output, flush=True)
