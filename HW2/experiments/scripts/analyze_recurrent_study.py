"""Analyze verified fixed-model predictions after both models converge."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import torch
from paths import SCRIPTS
from recurrent_study import make_model, study_hw2, verify_study
from train_recurrent import atomic_json


def paired_predictions(gru, lstm):
    if not gru[['src','target']].equals(lstm[['src','target']]):
        raise ValueError('Predictions must refer to the same ordered questions and targets.')
    if len(gru)==0:raise ValueError('Empty evaluation set')
    if not all(f['correct'].isin([0,1,True,False]).all() for f in [gru,lstm]):
        raise ValueError('Correctness values must be binary')
    a=gru['correct'].astype(bool);b=lstm['correct'].astype(bool)
    return dict(gru_only_correct=int((a&~b).sum()),lstm_only_correct=int((~a&b).sum()),
                both_correct=int((a&b).sum()),both_wrong=int((~a&~b).sum()),
                delta_percentage_points=100*(float(a.mean())-float(b.mean())))


def benchmark(models, config, hw):
    torch.set_num_threads(2)
    frame=pd.read_csv(hw/'data/arithmetic_train.csv',dtype=str,nrows=128)
    mapping=config['char_to_id'];xs=[];ys=[]
    for row in frame.itertuples(index=False):
        x=[mapping[c] for c in row.src+row.tgt]
        y=[0]*(len(row.src)-1)+[mapping[c] for c in row.tgt]+[mapping['<eos>']]
        xs.append(torch.tensor(x));ys.append(torch.tensor(y))
    lengths=torch.tensor([len(x) for x in xs]);x=torch.nn.utils.rnn.pad_sequence(xs,batch_first=True)
    y=torch.nn.utils.rnn.pad_sequence(ys,batch_first=True)
    prompts=list(pd.read_csv(hw/'data/arithmetic_eval.csv',dtype=str,nrows=128).src)
    result={}
    for kind,model in models.items():
        cpu=[]
        for _ in range(3):
            start=time.perf_counter()
            for prompt in prompts:model.generator(prompt,max_len=len(prompt)+16)
            cpu.append(1000*(time.perf_counter()-start)/len(prompts))
        result[kind]=dict(cpu_generation_median_ms_per_expression=statistics.median(cpu))
        if torch.backends.mps.is_available():
            candidate=copy.deepcopy(model).to('mps').train();opt=torch.optim.Adam(candidate.parameters(),lr=.001)
            bx=x.to('mps');by=y.to('mps');times=[]
            for i in range(13):
                torch.mps.synchronize();start=time.perf_counter();opt.zero_grad(set_to_none=True)
                logits=candidate(bx,lengths)
                loss=torch.nn.functional.cross_entropy(logits.reshape(-1,len(mapping)),by.reshape(-1),ignore_index=0)
                loss.backward();torch.nn.utils.clip_grad_value_(candidate.parameters(),1);opt.step()
                torch.mps.synchronize()
                if i>=3:times.append(1000*(time.perf_counter()-start))
            result[kind]['mps_update_median_ms']=statistics.median(times)
            del candidate,opt,bx,by
    return dict(models=result,scope='CPU cached generation: same 128 prompts, 3 trials. MPS: fixed pre-transferred batch, 3 warmups and 10 forward/backward/clip/Adam updates; excludes data loading and validation.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study-dir',required=True,type=Path);args=parser.parse_args()
    root=args.study_dir.resolve();hw=study_hw2(root);verify_study(root)
    results={};frames={};diagnostics={};models={};configs={};histories={}
    truth=pd.read_csv(hw/'data/arithmetic_eval.csv',dtype=str)[['src','tgt']].rename(columns={'tgt':'target'})
    for kind in ['gru','lstm']:
        p=root/kind;result=json.loads((p/'verification.json').read_text());assert result['converged']
        config=json.loads((p/'experiment.json').read_text());selected=json.loads((p/'checkpoint_metadata.json').read_text())
        assert hashlib.sha256((p/'checkpoint.pt').read_bytes()).hexdigest()==result['checkpoint_sha256']==selected['checkpoint_sha256']
        history=json.loads((p/'state/history.json').read_text());histories[kind]=history;configs[kind]=selected['config']
        results[kind]=dict(parameter_count=config['parameter_count'],best_epoch=selected['epoch'],
                           last_epoch=history[-1]['epoch'],evaluation=result['evaluation'],diagnostic=result['diagnostic'])
        for filename,storage,key in [('evaluation_predictions.csv',frames,'evaluation'),('diagnostic_predictions.csv',diagnostics,'diagnostic')]:
            f=pd.read_csv(p/filename,dtype={'src':str,'target':str,'prediction':str}).fillna({'prediction':''})
            assert (f['prediction'].eq(f['target']).astype(int)==f['correct']).all()
            assert len(f)==result[key]['total']==10000 and int(f['correct'].sum())==result[key]['correct']
            storage[kind]=f
        assert frames[kind][['src','target']].equals(truth)
        driver=json.loads((p/'source_notebook.ipynb').read_text());model=make_model(driver,selected['config']).eval()
        model.load_state_dict(torch.load(p/'checkpoint.pt',map_location='cpu',weights_only=True));models[kind]=model
        records=[x for x in history if x.get('completed_epoch',True)]
        plt.plot([x['epoch'] for x in records],[100*x['exact_match'] for x in records],label=kind.upper())
    assert {k:v for k,v in configs['gru'].items() if k!='cell_type'}=={k:v for k,v in configs['lstm'].items() if k!='cell_type'}
    paired=paired_predictions(frames['gru'],frames['lstm']);paired_diagnostic=paired_predictions(diagnostics['gru'],diagnostics['lstm'])
    delta=paired['delta_percentage_points'];reduction=1-results['gru']['parameter_count']/results['lstm']['parameter_count']
    findings=abs(delta)>=.1 or abs(paired_diagnostic['delta_percentage_points'])>=.5 or (reduction>=.2 and abs(delta)<=.1)
    measurements=benchmark(models,configs['gru'],hw)
    comparison=dict(models=results,evaluation_delta_percentage_points=delta,gru_parameter_reduction=reduction,
                    paired_evaluation=paired,paired_diagnostic=paired_diagnostic,benchmark=measurements,
                    useful_finding=bool(findings),single_seed=True,
                    limitation='Matched dimensions, not parameter count. The supplied eval was used for selection; differences are descriptive, not confirmatory architecture significance tests.')
    atomic_json(root/'comparison.json',comparison)
    plt.xlabel('Completed epoch');plt.ylabel('Full-answer EM (%)');plt.legend();plt.grid(alpha=.3)
    plt.tight_layout();plt.savefig(root/'learning_curves.png');plt.close()
    table='| 模型 | 參數量 | 最佳 epoch | 完成 epoch | Eval EM | 診斷 EM |\n|---|---:|---:|---:|---:|---:|\n'
    for kind,r in results.items():table+=f"| {kind.upper()} | {r['parameter_count']:,} | {r['best_epoch']} | {r['last_epoch']} | {r['evaluation']['exact_match']:.2%} | {r['diagnostic']['exact_match']:.2%} |\n"
    performance='| 模型 | MPS update median（ms） | CPU generation median（ms／題） |\n|---|---:|---:|\n'
    for kind,r in measurements['models'].items():performance+=f"| {kind.upper()} | {r.get('mps_update_median_ms',float('nan')):.2f} | {r['cpu_generation_median_ms_per_expression']:.3f} |\n"
    mode=json.loads((root/'study.json').read_text()).get('lstm_initialization','resume')
    initialization='GRU 與 LSTM 皆從頭訓練' if mode=='fresh' else 'GRU 從頭訓練，LSTM 從完整狀態的隔離副本接續'
    report=f'''# GRU 與 LSTM 收斂實驗

## 摘要

比較兩個單一 seed 的固定維度模型；兩者均完成相同收斂判準，已繳交檔案維持原樣。

## 實作方法

兩層 recurrent network、Embedding 128、hidden size 512、batch size 128、seed 2。沿用相同原始資料、teacher forcing、Adam、gradient clipping、scheduler 與 greedy generation。{initialization}；以完整 Eval EM 選模，再獨立載入最佳權重驗證 Eval 與合成診斷集。沒有時間、分數或 epoch 上限。

收斂判準為 12 輪未提升超過 0.01 個百分點，且包含至少 6 輪以 lr≤1e-5 完成訓練。原始 LSTM 實驗狀態與繳交權重均未覆寫。大型實驗權重留在本機，PR 收錄可重現程式、逐輪紀錄與完整預測。

## 實驗結果

{table}
GRU 相對 LSTM 的 Eval EM 差異為 {delta:+.2f} 個百分點，參數量減少 {reduction:.2%}。Eval 中 GRU 單獨答對 {paired['gru_only_correct']} 題、LSTM 單獨答對 {paired['lstm_only_correct']} 題，兩者皆答錯 {paired['both_wrong']} 題。診斷集差異為 {paired_diagnostic['delta_percentage_points']:+.2f} 個百分點。

{performance}
MPS 數字來自相同、已傳入 GPU 的 batch：3 次 warmup 後測量 10 次 forward／backward／gradient clipping／Adam update，不含 DataLoader 與 validation。CPU generation 使用相同 128 題，重複 3 次。這些是本機測量，不代表其他硬體的固定加速比。

此為單一 seed、固定維度而非固定參數量的比較；Eval 參與選模，分數差異以描述性觀察呈現，不能直接建立跨 seed 的架構優劣或顯著性結論。LSTM 早期紀錄包含中斷與續訓，不直接用歷史秒數推論純 GPU 速度。完整學習曲線、預測與分類統計隨附。
'''
    (root/'report.md').write_text(report);verify_study(root)
    if json.loads((root/'study.json').read_text()).get('publish_draft_if_useful') and findings:
        subprocess.run([sys.executable,str(SCRIPTS/'publish_recurrent_study.py'),'--study-dir',str(root),'--publish'],check=True)
    print(json.dumps(comparison,indent=2),flush=True)

if __name__=='__main__':main()
