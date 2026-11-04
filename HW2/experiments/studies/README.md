# Post-submission recurrent studies

## 摘要

比較 GRU 與 LSTM 的收斂結果、參數量、固定 batch 更新速度及生成速度；所有實驗保護已繳交 Notebook、checkpoint、ZIP 與原始資料，不會重新打包作業。

## 實作方法

完整設定為兩層、Embedding 128、hidden 512、batch 128、seed 2；使用相同 teacher forcing、Adam、gradient clipping、ReduceLROnPlateau 與完整答案生成 EM。固定維度不等於固定參數量。

目前的本機實驗先訓練 GRU，再從舊 LSTM 的完整狀態副本接續至相同收斂判準。其他支援 MPS 的 Mac 若沒有本機 optimizer 備份，可以讓兩個模型皆從頭訓練：

```bash
python HW2/experiments/scripts/run_recurrent_study.py \
  --study-dir HW2/experiments/runs/gru_lstm_fresh \
  --lstm-start fresh
```

若使用獨立 worktree，可把 study-dir 指向主要資料夾內的 HW2/experiments/runs/ 新目錄。所有輸入與輸出位置由該實驗目錄推導，程式與實驗資料可分開存放。

預設不設時間、分數或 epoch 上限。收斂需要連續 12 輪沒有超過 0.01 個百分點的改善，並包含至少 6 輪以 lr≤1e-5 完成訓練。每 1000 batches 及每輪保存恢復狀態。

在 study-dir 建立 STOP 可安全暫停；需要恢復時使用相同 study-dir 與 --resume。無須重新指定既有 study 的 LSTM 初始化模式。

## 實驗結果

完成後，各 study 會保存兩個模型的逐輪紀錄、最佳權重的獨立驗證、完整預測、配對錯誤比較、共同條件速度測量及學習曲線。診斷集僅於選定模型後評估，不參與選模。

單一 seed 的結果屬於固定模型的描述性比較，不能直接代表架構的一般優劣。歷史 LSTM 的中斷／續訓秒數不作為純 GPU 速度比較。

發布工具預設只產生 publication plan。只有 --publish、兩模型均收斂並驗證完成、結果通過 finding 條件時，才會保存挑選的結果、推送獨立 feature branch 並建立中文 Draft PR；程式沒有自動 merge 功能。本機實驗已依使用者指示啟用有新發現時建立 Draft PR。最佳實驗權重與 optimizer 備份維持留在本機。
