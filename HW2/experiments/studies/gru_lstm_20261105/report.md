# GRU 與 LSTM 收斂實驗

## 摘要

比較兩個單一 seed 的固定維度模型；兩者均完成相同收斂判準，已繳交檔案維持原樣。

## 實作方法

兩層 recurrent network、Embedding 128、hidden size 512、batch size 128、seed 2。沿用相同原始資料、teacher forcing、Adam、gradient clipping、scheduler 與 greedy generation。GRU 從頭訓練，LSTM 從完整狀態的隔離副本接續；以完整 Eval EM 選模，再獨立載入最佳權重驗證 Eval 與合成診斷集。沒有時間、分數或 epoch 上限。

收斂判準為 12 輪未提升超過 0.01 個百分點，且包含至少 6 輪以 lr≤1e-5 完成訓練。原始 LSTM 實驗狀態與繳交權重均未覆寫。大型實驗權重留在本機，PR 收錄可重現程式、逐輪紀錄與完整預測。

## 實驗結果

| 模型 | 參數量 | 最佳 epoch | 完成 epoch | Eval EM | 診斷 EM |
|---|---:|---:|---:|---:|---:|
| GRU | 2,836,242 | 34 | 46 | 96.18% | 88.67% |
| LSTM | 3,690,258 | 32 | 44 | 99.25% | 94.61% |

GRU 相對 LSTM 的 Eval EM 差異為 -3.07 個百分點，參數量減少 23.14%。Eval 中 GRU 單獨答對 20 題、LSTM 單獨答對 327 題，兩者皆答錯 55 題。診斷集差異為 -5.94 個百分點。

| 模型 | MPS update median（ms） | CPU generation median（ms／題） |
|---|---:|---:|
| GRU | 25.36 | 0.913 |
| LSTM | 27.91 | 1.442 |

MPS 數字來自相同、已傳入 GPU 的 batch：3 次 warmup 後測量 10 次 forward／backward／gradient clipping／Adam update，不含 DataLoader 與 validation。CPU generation 使用相同 128 題，重複 3 次。這些是本機測量，不代表其他硬體的固定加速比。

此為單一 seed、固定維度而非固定參數量的比較；Eval 參與選模，分數差異以描述性觀察呈現，不能直接建立跨 seed 的架構優劣或顯著性結論。LSTM 早期紀錄包含中斷與續訓，不直接用歷史秒數推論純 GPU 速度。完整學習曲線、預測與分類統計隨附。
