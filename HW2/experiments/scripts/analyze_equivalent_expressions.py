"""Dataset-only analysis of commutative/associative overlap.

Generative AI assistance: OpenAI Codex authored this diagnostic. It never
computes model predictions and is not included in the submission ZIP.
Numeric constants are retained as input symbols; expressions are not evaluated.
"""
import ast
import csv
import json
from pathlib import Path
from paths import DATA, DIAGNOSTICS, WORKSPACE

ROOT = WORKSPACE


def canonical_expression(source):
    def normalize(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return 'n' + str(node.value)
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            terms = []
            def add_terms(value, sign):
                if isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add):
                    add_terms(value.left, sign)
                    add_terms(value.right, sign)
                elif isinstance(value, ast.BinOp) and isinstance(value.op, ast.Sub):
                    add_terms(value.left, sign)
                    add_terms(value.right, -sign)
                else:
                    terms.append(('+' if sign > 0 else '-') + normalize(value))
            add_terms(node, 1)
            return 'A(' + ','.join(sorted(terms)) + ')'
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            factors = []
            def add_factors(value):
                if isinstance(value, ast.BinOp) and isinstance(value.op, ast.Mult):
                    add_factors(value.left)
                    add_factors(value.right)
                else:
                    factors.append(normalize(value))
            add_factors(node)
            return 'M(' + ','.join(sorted(factors)) + ')'
        raise ValueError('Unsupported expression syntax')
    return normalize(ast.parse(source.removesuffix('='), mode='eval').body)


def main():
    known = set()
    with (DATA / 'arithmetic_train.csv').open() as stream:
        for row in csv.DictReader(stream):
            known.add(canonical_expression(row['src']))
    result = dict(method='Input-preserving commutativity/associativity only; no numeric evaluation or distributivity',
                  training_canonical_classes=len(known))
    coverage = []
    for name, path in [('provided_eval', DATA / 'arithmetic_eval.csv'),
                       ('synthetic_diagnostic', DIAGNOSTICS / 'synthetic_three_operand_holdout.csv')]:
        seen = total = 0
        with path.open() as stream:
            for row in csv.DictReader(stream):
                canonical = canonical_expression(row['src'])
                present = canonical in known
                seen += int(present)
                coverage.append(dict(dataset=name, src=row['src'], seen_equivalent_input=present))
                total += 1
        result[name] = dict(known_equivalent_input=seen, total=total, fraction=seen / total)
    with (DIAGNOSTICS / 'equivalent_expression_coverage.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['dataset', 'src', 'seen_equivalent_input'])
        writer.writeheader()
        writer.writerows(coverage)
    (DIAGNOSTICS / 'equivalent_expression_analysis.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
