"""Controlled GPU report pilots, separate from leaderboard training.

Generative AI assistance: OpenAI Codex authored these reproducible experiments.
All comparisons use the same small capacity, sample budget, seed, and evaluator.
They are not the full-size leaderboard model and never replace its checkpoint.
"""
import argparse
from copy import deepcopy
import csv
import json
import random
import re
import time
from datetime import datetime
from pathlib import Path
from paths import DATA, HW2, PILOTS, RUNS, WORKSPACE
from zoneinfo import ZoneInfo

import pandas as pd
import torch

from generate_holdout import arithmetic_label

ROOT = WORKSPACE
DEADLINE = datetime(2026, 11, 4, 22, 30, tzinfo=ZoneInfo('Asia/Taipei'))


def remaining():
    return (DEADLINE - datetime.now(ZoneInfo('Asia/Taipei'))).total_seconds()


def corrupt_answers(frame, seed=2):
    rng = random.Random(seed)
    result = frame.copy(deep=True)
    changed = rng.sample(range(len(frame)), len(frame) // 5)
    for index in changed:
        target = result.iloc[index]['tgt']
        alternatives = [str(x) for x in range(10) if str(x) != target[-1]]
        if target.startswith('-') and len(target) == 2:
            alternatives = [x for x in alternatives if x != '0']
        result.iat[index, result.columns.get_loc('tgt')] = target[:-1] + rng.choice(alternatives)
    return result, changed


def synthetic_frame(count, lower, upper, seed, excluded=None):
    rng = random.Random(seed)
    templates = ['#+#+#=', '#+#-#=', '#-#+#=', '#-#-#=', '#+#*#=', '#-#*#=',
                 '#*#+#=', '#*#-#=', '#*#*#=', '(#+#)*#=', '(#-#)*#=',
                 '#*(#+#)=', '#*(#-#)=', '#-(#+#)=', '#-(#-#)=']
    blocked = set() if excluded is None else set(excluded)
    records = {}
    while len(records) < count:
        template = rng.choice(templates)
        numbers = iter(rng.randint(lower, upper) for _ in range(3))
        prompt = re.sub('#', lambda _: str(next(numbers)), template)
        if prompt not in blocked and prompt not in records:
            records[prompt] = str(arithmetic_label(prompt))
    return pd.DataFrame(records.items(), columns=['src', 'tgt'])


def fit(name, cell_type, train, evaluation, config, output):
    if remaining() <= 75:
        raise TimeoutError('Insufficient time before the 22:30 cutoff.')
    torch.manual_seed(config['seed'])
    mapping = config['char_to_id']
    ns = dict(torch=torch, char_to_id=mapping, id_to_char={i: c for c, i in mapping.items()},
              answer_order='normal', reorder_answer_digits=lambda answer, order='normal': answer)
    source = ''.join(config['notebook']['cells'][15]['source'])
    source = source.replace('torch.nn.LSTM(', f'torch.nn.{cell_type}(')
    exec(source, ns)
    model = ns['CharRNN'](len(mapping), config['embed_dim'], config['hidden_dim']).to(config['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=mapping['<pad>'])
    examples = []
    for row in train.itertuples(index=False):
        x = [mapping[c] for c in row.src + row.tgt]
        y = [mapping['<pad>']] * (len(row.src) - 1) + [mapping[c] for c in row.tgt] + [mapping['<eos>']]
        examples.append((x, y))
    def collate(batch):
        xs = [torch.tensor(x, dtype=torch.long) for x, _ in batch]
        ys = [torch.tensor(y, dtype=torch.long) for _, y in batch]
        return (torch.nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=mapping['<pad>']),
                torch.nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=mapping['<pad>']),
                torch.tensor([len(x) for x in xs], dtype=torch.long))
    generator = torch.Generator()
    loader = torch.utils.data.DataLoader(examples, batch_size=128, shuffle=True,
                                        collate_fn=collate, generator=generator)
    history = []
    best = -1.0
    best_predictions = None
    best_state = None
    started = time.monotonic()
    for epoch in range(1, config['epochs'] + 1):
        generator.manual_seed(config['seed'] + epoch)
        model.train()
        for xs, ys, lengths in loader:
            if remaining() <= 40:
                raise TimeoutError('22:30 safety reserve reached during report experiment.')
            optimizer.zero_grad(set_to_none=True)
            logits = model(xs.to(config['device']), lengths)
            loss = criterion(logits.reshape(-1, len(mapping)), ys.to(config['device']).reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 1)
            optimizer.step()
        model.eval()
        evaluation_model = deepcopy(model).to('cpu').eval()
        matched = 0
        predictions = []
        for row in evaluation.itertuples(index=False):
            if remaining() <= 30:
                raise TimeoutError('22:30 safety reserve reached during report evaluation.')
            prediction = ''.join(evaluation_model.generator(row.src, max_len=len(row.src) + 16))[len(row.src):]
            matched += int(prediction == row.tgt)
            predictions.append(dict(src=row.src, target=row.tgt, prediction=prediction))
        del evaluation_model
        score = matched / len(evaluation)
        history.append(dict(epoch=epoch, exact_match=score))
        if score > best:
            best = score
            best_predictions = predictions
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(f'{name}: epoch={epoch}, full pilot EM={score:.4f}', flush=True)
    result = dict(name=name, cell_type=cell_type, train_rows=len(train), eval_rows=len(evaluation),
                  epochs_completed=len(history), embed_dim=config['embed_dim'], hidden_dim=config['hidden_dim'],
                  batch_size=128, seed=config['seed'], device=str(config['device']),
                  parameter_count=sum(p.numel() for p in model.parameters()), evaluation_device='cpu',
                  best_exact_match=best, history=history, seconds=time.monotonic() - started)
    path = output / name
    path.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, path / 'pilot_checkpoint.pt')
    (path / 'result.json').write_text(json.dumps(result, indent=2))
    pd.DataFrame(best_predictions).to_csv(path / 'predictions.csv', index=False)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true', help='Tiny GPU/API checks only; not report results')
    parser.add_argument('--only-shift', action='store_true', help='Complete only pending digit-shift fits, reusing compatible completed runs')
    parser.add_argument('--train-rows', type=int, default=50000)
    parser.add_argument('--eval-rows', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=5)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    if device.type == 'cpu':
        raise RuntimeError('Report training also requires a real GPU.')
    torch.set_num_threads(2)
    output = PILOTS / ('report_smoke' if args.smoke else 'report_experiments')
    output.mkdir(exist_ok=True)
    notebook = json.loads((HW2 / 'NLP_HW2_NCU_114502540.ipynb').read_text())
    metadata = json.loads((RUNS / 'mps_e128_h512_seed2' / 'checkpoint_metadata.json').read_text())
    train = pd.read_csv(DATA / 'arithmetic_train.csv', dtype=str)
    evaluation = pd.read_csv(DATA / 'arithmetic_eval.csv', dtype=str)
    train = train.sample(n=128 if args.smoke else args.train_rows, random_state=2).reset_index(drop=True)
    evaluation = evaluation.sample(n=4 if args.smoke else args.eval_rows, random_state=42).reset_index(drop=True)
    config = dict(seed=2, epochs=1 if args.smoke else args.epochs,
                  embed_dim=16 if args.smoke else 64, hidden_dim=24 if args.smoke else 64,
                  char_to_id=metadata['config']['char_to_id'], device=device, notebook=notebook)
    previous_path = output / 'summary.json'
    previous = json.loads(previous_path.read_text()) if args.only_shift and previous_path.is_file() else {}
    results = dict(purpose='GPU API smoke only' if args.smoke else 'Controlled small-capacity report pilots; not leaderboard training',
                   started=previous.get('started', datetime.now(ZoneInfo('Asia/Taipei')).isoformat()),
                   complete=False, runs=previous.get('runs', []))
    if 'noise_rows' in previous:
        results['noise_rows'] = previous['noise_rows']
    def run_fit(name, cell_type, training, testing):
        result_path = output / name / 'result.json'
        if args.only_shift and result_path.is_file():
            old = json.loads(result_path.read_text())
            required = dict(cell_type=cell_type, train_rows=len(training), eval_rows=len(testing),
                            epochs_completed=config['epochs'], embed_dim=config['embed_dim'],
                            hidden_dim=config['hidden_dim'], batch_size=128, seed=config['seed'], device=str(device))
            if all(old.get(key) == value for key, value in required.items()):
                print('Reusing completed controlled fit:', name, flush=True)
                return old
        return fit(name, cell_type, training, testing, config, output)
    def retain(result):
        results['runs'] = [old for old in results['runs'] if old['name'] != result['name']] + [result]
        (output / 'summary.json').write_text(json.dumps(results, indent=2))
    try:
        if not args.only_shift:
            for cell_type in ['RNN', 'LSTM', 'GRU']:
                retain(run_fit(cell_type.lower() + '_clean', cell_type, train, evaluation))
            if not args.smoke:
                noisy, changed = corrupt_answers(train)
                assert len(changed) == len(train) // 5
                assert sum(train.tgt != noisy.tgt) == len(changed)
                results['noise_rows'] = len(changed)
                retain(run_fit('lstm_noise20', 'LSTM', noisy, evaluation))
        if not args.smoke:
            two_train = synthetic_frame(args.train_rows, 10, 99, 3)
            two_eval = synthetic_frame(args.eval_rows, 10, 99, 4, set(two_train.src))
            three_train = synthetic_frame(args.train_rows, 100, 999, 3)
            retain(run_fit('lstm_train2_test2', 'LSTM', two_train, two_eval))
            retain(run_fit('lstm_train3_test2', 'LSTM', three_train, two_eval))
        results['complete'] = True
    except TimeoutError as exc:
        results['stop_reason'] = str(exc)
    finally:
        results['finished'] = datetime.now(ZoneInfo('Asia/Taipei')).isoformat()
        (output / 'summary.json').write_text(json.dumps(results, indent=2))
        print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
