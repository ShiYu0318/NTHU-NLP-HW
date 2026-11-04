"""Compare every compact training token/label with the prior training encoder.
Generative AI assistance: OpenAI Codex authored this exact-equivalence audit.
"""
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
from paths import AUDITS, DATA, RUNS, TEMPLATES, WORKSPACE

ROOT = WORKSPACE
os.environ['HW2_DATA_DIR'] = str(DATA)
os.environ['HW2_OUTPUT_DIR'] = '/tmp/hw2_compact_audit'
new = json.loads((TEMPLATES / 'training_driver.json').read_text())
old = json.loads((RUNS / 'mps_e128_h512_seed2' / 'source_notebook.ipynb').read_text())
ns = {'answer_order': 'normal'}
for index in [1, 2, 3, 5, 7, 11]:
    exec(''.join(new['cells'][index]['source']), ns)
function = next(node for node in ast.parse(''.join(old['cells'][7]['source'])).body
                if isinstance(node, ast.FunctionDef) and node.name == 'encode_training_row')
oracle = dict(char_to_id=ns['char_to_id'], pad_id=ns['pad_id'], eos_id=ns['eos_id'])
exec(compile(ast.Module(body=[function], type_ignores=[]), 'legacy_encoder', 'exec'), oracle)
expected_hash = hashlib.sha256()
compact_hash = hashlib.sha256()
count = 0
with (DATA / 'arithmetic_train.csv').open() as stream:
    for index, row in enumerate(csv.DictReader(stream)):
        x, y = oracle['encode_training_row'](row['src'] + row['tgt'])
        length = int(ns['encoded_lengths'][index])
        assert len(x) == len(y) == length
        expected_hash.update(bytes(x) + b'\xff' + bytes(y) + b'\xfe')
        compact_hash.update(ns['encoded_x'][index, :length].tobytes() + b'\xff' + ns['encoded_y'][index, :length].tobytes() + b'\xfe')
        count += 1
assert count == 1000000 and expected_hash.digest() == compact_hash.digest()
result = dict(examples=count, all_token_labels_identical=True,
              token_label_sha256=compact_hash.hexdigest(),
              preprocessing_sha256=hashlib.sha256((''.join(new['cells'][7]['source']) + '\n' + ''.join(new['cells'][11]['source'])).encode()).hexdigest(),
              storage_bytes=ns['encoded_x'].nbytes + ns['encoded_y'].nbytes + ns['encoded_lengths'].nbytes,
              vocabulary=ns['char_to_id'])
(AUDITS / 'compact_data_verification.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)
