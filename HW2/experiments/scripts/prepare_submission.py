"""Verify a measured checkpoint, then optionally build the three-file submission.

Generative AI assistance: OpenAI Codex authored this verification/packaging tool.
Experiment helpers and data are never included in the submission ZIP.
"""
import argparse
import ast
import hashlib
import io
import json
import platform
import re
import shutil
import subprocess
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path
from paths import AUDITS, DATA, DIAGNOSTICS, HW2, ORIGINAL_NOTEBOOK, PILOTS, SCRIPTS, TEMPLATES, WORKSPACE, resolve_recorded_path

import nbformat
import pandas as pd
import torch

ROOT = WORKSPACE
NOTEBOOK = HW2 / 'NLP_HW2_NCU_114502540.ipynb'
ORIGINAL = ORIGINAL_NOTEBOOK


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace_answer(report, heading, next_heading, answer):
    start = report.index('Answer:', report.index(heading)) + len('Answer:')
    end = report.index(next_heading, start)
    return report[:start] + ' ' + answer + '\n\n' + report[end:]


def build_archive(path, files):
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for file in files:
            archive.write(file, file.name)
    with zipfile.ZipFile(path) as archive:
        assert archive.namelist() == [file.name for file in files]
        assert archive.testzip() is None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--limit', type=int, help='Smoke verification only; cannot publish a subset score')
    parser.add_argument('--publish', action='store_true')
    parser.add_argument('--include-diagnostic', action='store_true')
    parser.add_argument('--select-late-candidates', action='store_true')
    args = parser.parse_args()
    if (args.publish or args.include_diagnostic) and args.limit is not None:
        parser.error('Submission publication requires all 10,000 evaluation examples.')
    run_dir = args.run_dir.resolve()
    checkpoint = run_dir / 'checkpoint.pt'
    metadata = json.loads((run_dir / 'checkpoint_metadata.json').read_text())
    config = metadata['config']
    order = config.get('answer_order', 'normal')
    torch.set_num_threads(2)
    notebook = json.loads(NOTEBOOK.read_text())
    driver = json.loads((TEMPLATES / 'training_driver.json').read_text())
    # Reverse-digit models need the corresponding preprocessing and decoder in
    # the same original cells; the standard model keeps the minimal template.
    if order != 'normal':
        notebook['cells'][7]['source'] = driver['cells'][7]['source']
        notebook['cells'][15]['source'] = driver['cells'][15]['source']
        source = ''.join(notebook['cells'][7]['source'])
        source = source.replace("answer_order = globals().get('answer_order', 'normal')",
                                f'answer_order = {order!r}')
        notebook['cells'][7]['source'] = source.splitlines(keepends=True)
    ns = dict(torch=torch, char_to_id=config['char_to_id'],
              id_to_char={i: c for c, i in config['char_to_id'].items()}, answer_order=order)
    for node in ast.parse(''.join(driver['cells'][7]['source'])).body:
        if isinstance(node, ast.FunctionDef) and node.name == 'reorder_answer_digits':
            exec(compile(ast.Module(body=[node], type_ignores=[]), 'answer_order', 'exec'), ns)
    exec(''.join(notebook['cells'][15]['source']), ns)
    model = ns['CharRNN'](config['vocab_size'], config['embed_dim'], config['hidden_dim'])
    checkpoint_bytes = checkpoint.read_bytes()
    state = torch.load(io.BytesIO(checkpoint_bytes), map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    model.eval()
    checkpoint_hash = hashlib.sha256(checkpoint_bytes).hexdigest()
    if metadata.get('checkpoint_sha256', checkpoint_hash) != checkpoint_hash:
        raise RuntimeError('Checkpoint hash does not match metadata; retry a stable checkpoint.')
    evaluation_file = DATA / 'arithmetic_eval.csv'
    evaluation = pd.read_csv(evaluation_file, dtype=str)
    assert len(evaluation) == 10000
    evaluated = evaluation if args.limit is None else evaluation.head(args.limit)
    # The supplied train range has at most six answer characters; 16 new tokens
    # independently checks that the training evaluator's shorter limit hid no errors.
    correct = 0
    predictions = []
    started = time.monotonic()
    with torch.inference_mode():
        for row in evaluated.itertuples(index=False):
            result = ''.join(model.generator(row.src, max_len=len(row.src) + 16))
            answer = result[len(row.src):]
            matched = answer == row.tgt
            correct += int(matched)
            predictions.append(dict(src=row.src, target=row.tgt, prediction=answer, correct=matched))
            if len(predictions) % 1000 == 0:
                print(f'Independently verified {len(predictions)}/{len(evaluated)}', flush=True)
    assert sha256(checkpoint) == checkpoint_hash, 'Checkpoint changed during verification; retry the current best.'
    result = dict(checkpoint_sha256=checkpoint_hash, exact_match=correct / len(evaluated),
                  correct=correct, total=len(evaluated), device='CPU independent inference',
                  recorded_training_em=metadata['exact_match'], answer_order=order,
                  eval_sha256=sha256(evaluation_file), seconds=time.monotonic() - started,
                  full_evaluation=args.limit is None,
                  metadata_weights_match_hash=(metadata.get('checkpoint_sha256') == checkpoint_hash and metadata.get('binding_verified', True)),
                  metadata_em_matches=(correct == round(float(metadata['exact_match']) * len(evaluated))) if args.limit is None else None)
    print(json.dumps(result, indent=2), flush=True)
    audit_name = 'final_verification.json' if args.limit is None else 'smoke_verification.json'
    (run_dir / audit_name).write_text(json.dumps(result, indent=2))
    if args.limit is not None:
        return
    pd.DataFrame(predictions).to_csv(run_dir / 'verified_predictions.csv', index=False)
    if not result['metadata_em_matches']:
        if not result['metadata_weights_match_hash'] or abs(result['exact_match'] - metadata['exact_match']) > 0.001:
            raise RuntimeError('Independent EM differs from unbound metadata or materially across devices; investigate before packaging.')
        print('Hash-bound weights have a small cross-device EM difference; both measured scores will be disclosed.', flush=True)
    if args.publish and args.select_late_candidates:
        eligible = []
        for candidate_file in run_dir.glob('weight_comparison_*/alpha_*/evaluation.json'):
            candidate = json.loads(candidate_file.read_text())
            candidate_path = candidate_file.parent
            if (candidate.get('total') == 10000 and candidate.get('device') == 'CPU'
                    and candidate.get('exact_match', -1) > result['exact_match']
                    and (candidate_path / 'checkpoint_metadata.json').is_file()
                    and sha256(candidate_path / 'checkpoint.pt') == candidate['checkpoint_sha256']):
                eligible.append((candidate['exact_match'], candidate_path))
        if eligible:
            winner = max(eligible, key=lambda entry: entry[0])[1]
            print('Higher full-eval CPU candidate selected for fresh final verification:', winner, flush=True)
            command = [sys.executable, str(Path(__file__).resolve()), '--run-dir', str(winner), '--publish']
            if args.include_diagnostic:
                command.append('--include-diagnostic')
            subprocess.run(command, check=True)
            return
    if args.include_diagnostic:
        subprocess.run([sys.executable, str(SCRIPTS / 'evaluate_holdout.py'),
                        '--run-dir', str(run_dir)], check=True)
    if not args.publish:
        return

    # Keep all 21 original cells and every original report question.
    notebook['cells'][9]['source'] = (
        f"batch_size = {config['batch_size']}\nepochs = {max(1, metadata['epoch'])}\n"
        f"embed_dim = {config['embed_dim']}\nhidden_dim = {config['hidden_dim']}\n"
        'lr = 0.001\ngrad_clip = 1\n').splitlines(keepends=True)
    report = ''.join(notebook['cells'][20]['source'])
    lines = report.splitlines()
    for i, line in enumerate(lines):
        if line.startswith('Running environment:'):
            lines[i] = 'Running environment: 本機 Apple M5 GPU（MPS）訓練；最終 checkpoint 另以 CPU 完整獨立驗證。先前 baseline 使用 Colab Tesla T4。'
        elif line.startswith('Python version:'):
            lines[i] = f"Python version: {metadata['python']}；PyTorch {metadata['pytorch']}。requirements.txt 記錄選用實驗的實際環境版本。"
    report = '\n'.join(lines)
    model_description = (
        f"字元級模型為 Embedding({config['embed_dim']}) → LSTM({config['hidden_dim']}) → "
        f"LSTM({config['hidden_dim']}) → Linear({config['hidden_dim']},{config['hidden_dim']}) → ReLU → "
        f"Linear({config['hidden_dim']},18)，不含 attention。字典僅由訓練集建立，含 <pad>/<eos>。"
        "使用 teacher forcing：等號前的標籤與 batch padding 皆為 <pad> 並由交叉熵忽略；"
        "等號位置預測答案第一字元，答案最後一字元的位置預測 EOS。生成時僅給算式，不提供真實答案。\n\n"
        '模板表格保留原樣；實際參數以 Hyper Parameters 的 code cell 與本節為準。'
        f"選用設定：batch_size={config['batch_size']}、embed_dim={config['embed_dim']}、hidden_dim={config['hidden_dim']}、"
        "Adam 初始 lr=0.001、clip_grad_value_=1；ReduceLROnPlateau 監控完整生成 EM，factor=0.5、patience=1、"
        "threshold=0.001、min_lr=1e-5。較大的 hidden state 用於表示運算元與括號資訊，較小的 embedding 用於節省運算；"
        "這些是驗證集上的實驗選擇，不能視為保證 private test 表現。\n\n"
        f"答案順序：{'正常十進位順序' if order == 'normal' else '先生成個位數，負號保留開頭；輸出前還原正常十進位順序'}。"
        f"最佳 checkpoint 的訓練輪標記為 {metadata['epoch']}，紀錄的完整生成 EM={metadata['exact_match']:.2%}；"
        f"載入已儲存權重後，以此 Notebook 的 generator 在 CPU 獨立生成全部 10,000 筆，EM={result['exact_match']:.2%} "
        f"（{correct}/10000）。private-test EM 與排名尚未取得。"
    )
    if metadata.get('kind') == 'weight_interpolation':
        recipe = metadata['interpolation']
        model_description += (f"\n\n所選權重不是單一 epoch 的輸出：以 alpha={recipe['alpha']}，將第 {recipe['base_epoch']} 輪最佳權重與"
                              f"第 {recipe['recent_epoch']} 輪、下一個 batch 指標 {recipe['recent_next_batch']} 的晚期狀態作逐參數線性插值。"
                              "公式為 theta=(1-alpha)*theta_base+alpha*theta_recent。兩個來源都以 GPU teacher forcing 訓練，"
                              "插值在 CPU 進行；推論仍是單一兩層 LSTM，沒有 attention、外部算式求解或多模型集成。"
                              "上方 epoch 僅為來源參考標記；所列生成 EM 是插值後重新完整評估的實測值。")
        helper = ('\n# Generative AI assistance: checkpoint interpolation used in the selected experiment.\n'
                  'def interpolate_state_dicts(base_state, recent_state, alpha=0.5):\n'
                  '    assert 0 <= alpha <= 1 and base_state.keys() == recent_state.keys()\n'
                  '    return {key: torch.lerp(base_state[key].detach().cpu(), recent_state[key].detach().cpu(), alpha)\n'
                  '            for key in base_state}\n')
        optimizer_source = ''.join(notebook['cells'][17]['source'])
        if 'def interpolate_state_dicts(' not in optimizer_source:
            notebook['cells'][17]['source'] = (optimizer_source + helper).splitlines(keepends=True)
    progress = metadata.get('training_progress') or {}
    if progress.get('completed_epoch') is False:
        model_description += (f"\n\n此 checkpoint 來自最後一個部分訓練輪：第 {metadata['epoch']} 輪完成 "
                              f"{progress['completed_batches']}/{progress['total_batches']} batches，未完成整輪訓練，但 EM 仍是完整 10,000 筆。")
    report = replace_answer(report, '##### 1.', '##### 2.', model_description)
    run_key = f"lstm_e{config['embed_dim']}_h{config['hidden_dim']}_seed{config['seed']}"
    if order != 'normal':
        run_key += '_lsf'
    training_root = resolve_recorded_path(metadata.get('source_run_dir', str(run_dir))).resolve()
    assert training_root.is_relative_to(ROOT.resolve())
    matched_run = training_root / 'runs' / run_key
    history_path = matched_run / 'history.json'
    history = json.loads(history_path.read_text()) if history_path.is_file() else []
    table = '| Epoch | 完整生成 EM | 學習率 | 每輪秒數 |\n|---|---:|---:|---:|\n'
    for row in history:
        if 'learning_rate' in row:
            table += f"| {row['epoch']}{'（部分訓練）' if row.get('completed_epoch') is False else ''} | {row['exact_match']:.2%} | {row['learning_rate']:.6g} | {row['seconds']:.1f} |\n"
    error_file = matched_run / f"errors_epoch_{metadata['epoch']}.json"
    groups = {}
    for prediction_row in predictions:
        prompt, target = prediction_row['src'], prediction_row['target']
        for category in [f'answer_length_{len(target)}', 'multiplication' if '*' in prompt else 'addition_subtraction',
                         'parentheses' if '(' in prompt else 'no_parentheses',
                         'negative_answer' if target.startswith('-') else 'nonnegative_answer']:
            counts = groups.setdefault(category, dict(correct=0, total=0))
            counts['correct'] += int(prediction_row['correct'])
            counts['total'] += 1
    group_table = '| 分組 | EM | 正確／總數 |\n|---|---:|---:|\n'
    for name, counts in groups.items():
        group_table += f"| {name} | {counts['correct']/counts['total']:.2%} | {counts['correct']}/{counts['total']} |\n"
    actual_errors = [row for row in predictions if not row['correct']][:8]
    example_table = '| 算式 | 正解 | 實際生成 |\n|---|---:|---:|\n'
    for row in actual_errors:
        example_table += f"| `{row['src']}` | `{row['target']}` | `{row['prediction']}` |\n"
    extra = (
        "原始訓練集 1,000,000 筆、eval 10,000 筆，皆無缺值且 src 跨集交集為 0。"
        "eval 僅用於模型選擇，未加入訓練；private test 並未取得。T4 256/256 baseline 的兩輪 EM 為 49.73%、55.20%。"
        "下列為 MPS 實驗實測紀錄；不同模型、batch、裝置的比較並非控制所有變因的消融實驗。\n\n"
        + table + '\n若某輪由已完成訓練的狀態恢復評估，該列秒數僅為此次恢復評估時間，不代表整輪訓練耗時。\n最佳 epoch 的分類統計：\n\n' + group_table
        + '\n獨立驗證的實際錯誤範例：\n\n' + example_table
        + '\n訓練以實驗工具執行，主 Notebook 保留老師的 cell 與迴圈骨架。'
        "繳交的 checkpoint.pt 為 plain state_dict，可用相同 CharRNN 與字典載入；"
        "CPU 驗證使用較寬的 16-token 生成上限，仍逐筆比較完整答案字串。"
    )
    diagnostic_path = run_dir / f"diagnostic_epoch_{metadata['epoch']}.json"
    if diagnostic_path.is_file():
        diagnostic = json.loads(diagnostic_path.read_text())
        if diagnostic['checkpoint_sha256'] == checkpoint_hash:
            extra += (f"\n\n額外合成診斷集：10,000 筆三運算元算式，運算元 0–49，與 train/eval 的 src 皆不重複；"
                      f"僅用於推論診斷，未加入訓練。所選 checkpoint 的完整 EM={diagnostic['diagnostic_em']:.2%}"
                      f"（{diagnostic['correct']}/10000）。這不是官方 private test，不能據此宣稱 private 成績或排名。")
    profile_path = DIAGNOSTICS / 'diagnostic_distribution.json'
    if profile_path.is_file():
        profile = json.loads(profile_path.read_text())
        original_profile = profile['provided_eval']
        synthetic_profile = profile['synthetic_diagnostic']
        extra += (f"\n\n診斷集分布限制：含字面運算元 0 的算式佔合成集 {synthetic_profile['zero_operand_fraction']:.2%}，"
                  f"原 eval 則為 {original_profile['zero_operand_fraction']:.2%}。因此它是不同分布的額外壓力測試，"
                  "不能直接把兩組 EM 差距視為 private test 的預估下降。")
    equivalent_path = DIAGNOSTICS / 'equivalent_expression_analysis.json'
    if equivalent_path.is_file():
        equivalent = json.loads(equivalent_path.read_text())
        extra += (f"\n\n等價輸入分析：只依交換／結合律重排、保留數字為輸入符號，不做數值計算或分配律展開。"
                  f"原 eval 有 {equivalent['provided_eval']['fraction']:.2%} 的算式，其重排等價形式已出現在 train；"
                  f"合成診斷集則為 {equivalent['synthetic_diagnostic']['fraction']:.2%}。"
                  "所以 src 完全不重複仍不代表所有計算組合均為全新；原 eval 的高 EM 不能直接證明掌握通用算術演算法。"
                  "這是資料分析，未用來產生模型答案，也不能單憑此統計斷定全部成績差距的原因。")
    coverage_path = DIAGNOSTICS / 'equivalent_expression_coverage.csv'
    if coverage_path.is_file():
        coverage = pd.read_csv(coverage_path)
        known = set(coverage[(coverage.dataset == 'provided_eval') & (coverage.seen_equivalent_input.astype(str).str.lower() == 'true')].src)
        partition = {'seen_equivalent_input': dict(correct=0, total=0), 'unseen_equivalent_input': dict(correct=0, total=0)}
        for prediction_row in predictions:
            category = 'seen_equivalent_input' if prediction_row['src'] in known else 'unseen_equivalent_input'
            partition[category]['correct'] += int(prediction_row['correct'])
            partition[category]['total'] += 1
        extra += '\n\n| 集合／等價輸入是否在 train 出現 | EM | 正確／總數 |\n|---|---:|---:|\n'
        for name, counts in partition.items():
            extra += f"| 原 eval／{name} | {counts['correct']/counts['total']:.2%} | {counts['correct']}/{counts['total']} |\n"
        if diagnostic_path.is_file():
            diagnostic = json.loads(diagnostic_path.read_text())
            if diagnostic['checkpoint_sha256'] == checkpoint_hash:
                for name in ['seen_equivalent_input', 'unseen_equivalent_input']:
                    counts = diagnostic.get('groups', {}).get(name)
                    if counts:
                        extra += f"| 合成診斷／{name} | {counts['correct']/counts['total']:.2%} | {counts['correct']}/{counts['total']} |\n"
        extra += '\n這些分組是觀察性分析，並非因果實驗；用於區分相同計算組合的重排與較新計算組合的生成能力，不用於替模型計算答案。'
    outcome_path = matched_run / 'outcome.json'
    primary_terminal_path = training_root / 'primary_terminal.json'
    active_path = training_root / 'active_invocation.json'
    active_pid = json.loads(active_path.read_text()).get('pid') if active_path.is_file() else None
    terminal = json.loads(primary_terminal_path.read_text()) if primary_terminal_path.is_file() else {}
    saved_outcome = json.loads(outcome_path.read_text()) if outcome_path.is_file() else {}
    outcome = saved_outcome if saved_outcome.get('pid') == active_pid and active_pid is not None else {}
    if terminal.get('phase') == 'terminal' and terminal.get('pid') == active_pid:
        outcome = dict(outcome, stop_reason=terminal['stop_reason'],
                       completed_evaluation_epochs=max((row['epoch'] for row in history), default=0))
    if outcome:
        extra += (f"\n\n主模型實驗停止原因：{outcome['stop_reason']}；已完成完整評估至第 "
                  f"{outcome['completed_evaluation_epochs']} epoch。依使用者更新要求，續訓取消固定分數停止門檻，"
                  "以學習率降低後的 EM 停滯判定操作上收斂，並保留 11/4 22:30 時限。"
                  "此收斂判斷不代表證明任何其他設定都無法提高成績。")
    report = replace_answer(report, '##### … Anything', '##### Generative AI Usage Statement', extra)

    pilot_path = PILOTS / 'report_experiments/summary.json'
    if pilot_path.is_file():
        pilot = json.loads(pilot_path.read_text())
        pilots = {row['name']: row for row in pilot.get('runs', []) if row.get('epochs_completed', 0) > 0}
        if all(name in pilots for name in ['rnn_clean', 'lstm_clean', 'gru_clean']):
            clean = [pilots[name] for name in ['rnn_clean', 'lstm_clean', 'gru_clean']]
            baseline = clean[1]
            controlled_fields = ['train_rows', 'eval_rows', 'epochs_completed', 'embed_dim', 'hidden_dim', 'batch_size', 'seed', 'device']
            assert all(all(row[key] == baseline[key] for key in controlled_fields) for row in clean)
            setup = (f"另做固定維度的小規模 GPU 對照：相同 {baseline['train_rows']} 筆訓練、"
                     f"{baseline['eval_rows']} 筆 eval、seed={baseline['seed']}、embedding={baseline['embed_dim']}、"
                     f"hidden={baseline['hidden_dim']}、batch={baseline['batch_size']}，皆訓練 {baseline['epochs_completed']} epochs。"
                     "Adam lr=0.001、梯度逐元素裁切 1、不使用 scheduler；各模型取相同 epoch 預算內最高完整生成 EM。"
                     "這是固定維度而非固定參數量比較，也不能與主模型的百萬筆／完整萬筆 EM 直接比較。")
            table = '| 模型 | 最佳生成 EM | 參數量 |\n|---|---:|---:|\n'
            for row in clean:
                table += f"| {row['cell_type']} | {row['best_exact_match']:.2%} | {row['parameter_count']} |\n"
            observation = ('本次固定預算對照中 LSTM 高於 RNN。' if baseline['best_exact_match'] > clean[0]['best_exact_match']
                           else '本次固定預算對照中 LSTM 未優於 RNN；gating 不保證在有限資料與固定 epoch 下必然較佳。')
            answer = ("理論上，LSTM 的 cell state 與輸入／遺忘／輸出閘，以及 GRU 的更新／重設閘，有助保存跨字元資訊並減緩梯度消失；"
                      "vanilla RNN 較難學習長距離依賴。GRU 的 gate 較少，通常有較少參數，但品質仍需實測。\n\n"
                      + setup + '\n\n' + table + '\n' + observation
                      + '這些小容量模型仍可能未收斂；只有單一 seed，不能據此推論所有容量／訓練預算下的排序，也不能認定已學會精確算術演算法。')
            report = replace_answer(report, '##### 2.', '##### 3.', answer)
        if all(name in pilots for name in ['lstm_train2_test2', 'lstm_train3_test2']):
            two, three = pilots['lstm_train2_test2'], pilots['lstm_train3_test2']
            answer = (f"另建合成小規模對照，兩位數訓練運算元 10–99、三位數訓練運算元 100–999，各 {two['train_rows']} 筆；"
                      f"都在同一組 {two['eval_rows']} 筆兩位數算式上生成完整答案。維度 {two['embed_dim']}/{two['hidden_dim']}、"
                      f"batch={two['batch_size']}、seed={two['seed']}、Adam lr=0.001、epochs={two['epochs_completed']}，兩組相同。"
                      "兩位數 eval 與對照訓練集的 src 不重複；三位數訓練依運算元位數自然不重複。\n\n"
                      '| 訓練位數 → 評估位數 | 最佳生成 EM |\n|---|---:|\n'
                      f"| 兩位數 → 兩位數 | {two['best_exact_match']:.2%} |\n"
                      f"| 三位數 → 兩位數 | {three['best_exact_match']:.2%} |\n\n"
                      f"評估裝置：兩位數對照為 {two.get('evaluation_device', 'MPS')}，三位數模型為 {three.get('evaluation_device', 'MPS')}；生成均為相同 greedy 完整答案比較。"
                      "三位數到兩位數會改變輸入長度、答案範圍、數字位置與進位模式；共用字元字典不代表具備位數泛化能力。"
                      "這是小容量／有限預算實驗，收斂程度與任務難度也可能影響差距；若同位數對照本身仍很低，就不能把差距完全歸因於分布轉移。"
                      "提供的原始百萬筆資料仍原樣用於主模型，未混入這些合成訓練資料。")
            report = replace_answer(report, '##### 3.', '##### 4.', answer)
        if all(name in pilots for name in ['lstm_clean', 'lstm_noise20']):
            clean, noisy = pilots['lstm_clean'], pilots['lstm_noise20']
            answer = (f"在相同 {clean['train_rows']} 筆訓練資料中，固定 seed 隨機選出恰好 20%（{pilot['noise_rows']} 筆）修改答案最後一位數，"
                      "保留算式、答案字元長度與負號，且確保新答案不同於正解；這是特定的末位數標籤污染，不是任意形式的錯誤。"
                      f"評估集維持相同乾淨的 {clean['eval_rows']} 筆，兩模型同初始化 seed、維度、batch、Adam、裁切與 "
                      f"{clean['epochs_completed']} epochs。\n\n"
                      '| 訓練標籤 | 最佳生成 EM |\n|---|---:|\n'
                      f"| 乾淨 | {clean['best_exact_match']:.2%} |\n"
                      f"| 20% 末位數污染 | {noisy['best_exact_match']:.2%} |\n\n"
                      "錯誤監督會對相似輸入提供衝突梯度，可能使模型記住錯誤數字並降低乾淨資料的 EM。"
                      "本實驗只有單一 seed 與有限容量，若未觀察到下降，也不能推論錯誤標籤可靠地提升泛化。\n\n")
            clean_predictions = pd.read_csv(pilot_path.parent / 'lstm_clean/predictions.csv', dtype=str, keep_default_na=False)
            noisy_predictions = pd.read_csv(pilot_path.parent / 'lstm_noise20/predictions.csv', dtype=str, keep_default_na=False)
            merged = clean_predictions.merge(noisy_predictions, on=['src', 'target'], suffixes=('_clean', '_noise'), validate='one_to_one')
            assert len(merged) == clean['eval_rows']
            different = merged[merged.prediction_clean != merged.prediction_noise]
            examples = different.head(6) if len(different) else merged.head(6)
            answer += '| 算式 | 正解 | 乾淨模型實際生成 | 污染模型實際生成 |\n|---|---|---|---|\n'
            for row in examples.itertuples(index=False):
                answer += f"| `{row.src}` | `{row.target}` | `{row.prediction_clean}` | `{row.prediction_noise}` |\n"
            report = replace_answer(report, '##### 4.', '##### 5.', answer)

    ai_start = report.index('Answer:', report.index('##### Generative AI Usage Statement'))
    report = report[:ai_start] + ('Answer: 使用 OpenAI Codex 協助閱讀規定、填寫字典／前處理／Dataset／LSTM 生成／訓練與完整 EM 評估、'
                                 '處理 GPU 資料路徑、續訓與 checkpoint、比較晚期權重插值、建立額外合成診斷及小規模架構／標籤污染／位數對照實驗，'
                                 '並協助實際啟動監控 Apple M5 MPS 訓練、分析錯誤、驗證權重與整理報告。對應程式亦有 AI 使用註解。'
                                 '未使用同學程式，未直接複製網路範例；API 文件參考列於下方。最終內容與實驗紀錄需由本人核對。')
    report += ('\n\nAPI 參考：[PyTorch LSTM](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html)、'
               '[ReduceLROnPlateau](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html)。'
               '程式由本人與 Codex 協作實作，未直接複製網路範例程式。\n')
    report += '\ncheckpoint.pt 的所有 tensor 以 CPU storage 儲存，便於在 CPU、CUDA 或 MPS 環境載入；權重數值與已驗證的 GPU checkpoint 完全相同。\n'
    notebook['cells'][20]['source'] = report.splitlines(keepends=True)
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            cell['execution_count'] = None
            cell['outputs'] = []
    notebook['cells'][19]['outputs'] = [dict(output_type='stream', name='stdout', text=[
        'Measured training was executed by the experiment launcher on Apple M5 MPS.\n',
        f"Selected checkpoint reference epoch: {metadata['epoch']}; recorded full-eval EM: {metadata['exact_match']:.6f}\n",
        f"Independent saved-checkpoint verification using this notebook's generator on CPU: {correct}/10000 = {result['exact_match']:.6f}\n",
        'Submission cells were not rerun end-to-end to manufacture these outputs.\n',
        'Private leaderboard score/rank: unavailable.\n'])]
    notebook['metadata']['language_info']['version'] = metadata['python']
    original = json.loads(ORIGINAL.read_text())
    assert len(notebook['cells']) == len(original['cells']) == 21
    assert re.findall(r'^#{1,6} .*$', report, re.M) == re.findall(r'^#{1,6} .*$', ''.join(original['cells'][20]['source']), re.M)
    assert Counter(c['cell_type'] for c in notebook['cells']) == {'code': 13, 'markdown': 8}
    assert [c['cell_type'] for c in notebook['cells']] == [c['cell_type'] for c in original['cells']]
    for i, (a, b) in enumerate(zip(original['cells'], notebook['cells'])):
        if a['cell_type'] == 'markdown' and i != 20:
            assert a['source'] == b['source']
        if b['cell_type'] == 'code':
            compile(''.join(b['source']), f'cell_{i}', 'exec')
    nbformat.validate(nbformat.from_dict(notebook))
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n')
    # Normalize GPU storages to CPU without changing any parameter values.
    portable_path = HW2 / 'checkpoint.tmp'
    torch.save(state, portable_path)
    portable_path.replace(HW2 / 'checkpoint.pt')
    shutil.copy2(run_dir / 'requirements.txt', HW2 / 'requirements.txt')
    archive_path = HW2 / 'NLP_HW2_NCU_114502540.zip'
    build_archive(archive_path, [NOTEBOOK, HW2 / 'checkpoint.pt', HW2 / 'requirements.txt'])
    with zipfile.ZipFile(archive_path) as archive:
        archived = torch.load(io.BytesIO(archive.read('checkpoint.pt')), weights_only=True)
        assert all(value.device.type == 'cpu' for value in archived.values())
        assert all(torch.equal(state[key], archived[key]) for key in state)
    artifact_audit = dict(result, submission_checkpoint_sha256=sha256(HW2 / 'checkpoint.pt'),
                          archive_sha256=sha256(archive_path), tensors_on_cpu=True,
                          notebook_cells=21, code_cells=13, markdown_cells=8)
    (AUDITS / 'submission_verification.json').write_text(json.dumps(artifact_audit, indent=2))
    print('Verified three-file submission:', archive_path, flush=True)


if __name__ == '__main__':
    main()
