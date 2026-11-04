"""Analyze fixed-model predictions after both recurrent models converge."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import binomtest
from recurrent_study import verify_study
from train_recurrent import atomic_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study-dir',required=True,type=Path);args=parser.parse_args()
    root=args.study_dir.resolve();verify_study(root)
    results={};frames={}
    for kind in ['gru','lstm']:
        p=root/kind
        result=json.loads((p/'verification.json').read_text());assert result['converged']
        config=json.loads((p/'experiment.json').read_text())
        selected=json.loads((p/'checkpoint_metadata.json').read_text())
        history=json.loads((p/'state/history.json').read_text())
        results[kind]=dict(parameter_count=config['parameter_count'],best_epoch=selected['epoch'],
                           last_epoch=history[-1]['epoch'],evaluation=result['evaluation'],diagnostic=result['diagnostic'])
        frames[kind]=pd.read_csv(p/'evaluation_predictions.csv',dtype={'src':str,'target':str})
        records=[x for x in history if x.get('completed_epoch',True)]
        plt.plot([x['epoch'] for x in records],[100*x['exact_match'] for x in records],label=kind.upper())
    assert frames['gru'][['src','target']].equals(frames['lstm'][['src','target']])
    a=frames['gru']['correct'].astype(bool);b=frames['lstm']['correct'].astype(bool)
    only_gru=int((a&~b).sum());only_lstm=int((~a&b).sum());discordant=only_gru+only_lstm
    delta=100*(results['gru']['evaluation']['exact_match']-results['lstm']['evaluation']['exact_match'])
    paired=dict(gru_only_correct=only_gru,lstm_only_correct=only_lstm,
                exact_two_sided_p=float(binomtest(only_gru,discordant,.5).pvalue) if discordant else 1.)
    reduction=1-results['gru']['parameter_count']/results['lstm']['parameter_count']
    findings=abs(delta)>=.1 or paired['exact_two_sided_p']<.05 or (reduction>=.2 and abs(delta)<=.1)
    comparison=dict(models=results,evaluation_delta_percentage_points=delta,gru_parameter_reduction=reduction,
                    paired_fixed_model_test=paired,useful_finding=bool(findings),single_seed=True,
                    limitation='Matched dimensions, not parameter count. The paired test concerns these fixed models, not an architecture effect across training seeds.')
    atomic_json(root/'comparison.json',comparison)
    plt.xlabel('Completed epoch');plt.ylabel('Full-answer EM (%)');plt.legend();plt.grid(alpha=.3)
    plt.tight_layout();plt.savefig(root/'learning_curves.png');plt.close()
    table='| 模型 | 參數量 | 最佳 epoch | Eval EM | 診斷 EM |\n|---|---:|---:|---:|---:|\n'
    for kind,r in results.items():table+=f"| {kind.upper()} | {r['parameter_count']:,} | {r['best_epoch']} | {r['evaluation']['exact_match']:.2%} | {r['diagnostic']['exact_match']:.2%} |\n"
    report=f'''# GRU 與 LSTM 收斂實驗

## 摘要

比較兩個單一 seed 的固定維度模型；兩者均完成相同收斂判準，繳交檔案維持原樣。

## 實作方法

兩層 recurrent network、Embedding 128、hidden size 512、batch size 128、seed 2。沿用相同原始資料、teacher forcing、Adam、gradient clipping、scheduler 與 greedy generation。GRU 從頭訓練，LSTM 從完整狀態的隔離副本接續；以原始評估集選模，再獨立驗證最佳權重與合成診斷集。

## 實驗結果

{table}
GRU 相對 LSTM 的 Eval EM 差異為 {delta:+.2f} 個百分點，參數量減少 {reduction:.2%}。固定模型的 paired exact test：GRU 單獨答對 {only_gru} 題、LSTM 單獨答對 {only_lstm} 題，雙尾 p={paired['exact_two_sided_p']:.4g}。

此為單一 seed、固定維度而非固定參數量的比較，不能據此建立跨 seed 的架構優劣結論。LSTM 的早期紀錄包含中斷與續訓，不以歷史秒數直接推論純 GPU 速度。詳見各模型的逐輪紀錄與完整預測檔。
'''
    (root/'report.md').write_text(report)
    print(json.dumps(comparison,indent=2),flush=True)

if __name__=='__main__':main()
