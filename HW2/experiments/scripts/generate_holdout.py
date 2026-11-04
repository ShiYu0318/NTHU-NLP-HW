"""Generate evaluation-only arithmetic diagnostics, never training augmentation.

Generative AI assistance: OpenAI Codex authored this diagnostic-data generator.
The generated file is not the course's private test set and is not submitted.
"""
import ast
import csv
import hashlib
import json
import operator
import random
import re
from collections import Counter
from pathlib import Path
from paths import DATA, DIAGNOSTICS, WORKSPACE

ROOT = WORKSPACE


def arithmetic_label(expression):
    operators = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul}
    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in operators:
            return operators[type(node.op)](visit(node.left), visit(node.right))
        raise ValueError('Only integer literals, +, -, *, and parentheses are supported.')
    return visit(ast.parse(expression.removesuffix('='), mode='eval').body)


def main():
    rng = random.Random(20261104)
    excluded = set()
    templates = Counter()
    for name in ['arithmetic_train.csv', 'arithmetic_eval.csv']:
        with (DATA / name).open() as stream:
            for row in csv.DictReader(stream):
                excluded.add(row['src'])
                if name == 'arithmetic_train.csv':
                    template = re.sub(r'\d+', '#', row['src'])
                    if template.count('#') == 3:
                        templates[template] += 1
    assert all(re.fullmatch(r'[#()+*\-=]+', template) for template in templates)
    choices, weights = zip(*sorted(templates.items()))
    generated = {}
    attempts = 0
    while len(generated) < 10000:
        attempts += 1
        template = rng.choices(choices, weights=weights, k=1)[0]
        numbers = iter(rng.randrange(50) for _ in range(3))
        expression = re.sub('#', lambda _: str(next(numbers)), template)
        if expression in excluded or expression in generated:
            continue
        generated[expression] = str(arithmetic_label(expression))
    output = DIAGNOSTICS / 'synthetic_three_operand_holdout.csv'
    with output.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['src', 'tgt'])
        writer.writerows(generated.items())
    manifest = dict(rows=len(generated), seed=20261104, operand_range=[0, 49],
                    operands_per_expression=3, overlaps_with_train_or_eval=0,
                    purpose='Evaluation-only synthetic diagnostic; not official private test; never used for training',
                    attempts=attempts, templates=dict(templates),
                    sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    output.with_suffix('.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if k != 'templates'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
