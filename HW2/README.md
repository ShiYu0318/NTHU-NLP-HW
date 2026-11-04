# HW2：Arithmetic as a language

## 介紹

以字元序列生成學習算術運算。`Assignment2/` 保存課程模板與資料，根目錄保存英文 Notebook、最佳權重、環境版本及繳交 ZIP；`data/` 為實驗使用的原始 CSV。

## 方法

- Embedding(128) → LSTM(512) → LSTM(512) → Linear → ReLU → Linear；不使用 attention。
- 使用 teacher forcing，忽略算式前綴與 padding 的損失；生成時只提供算式。
- Adam、batch size 128、初始學習率 0.001、逐元素梯度裁切 1，依完整生成 EM 調整學習率。
- 每輪以全部 10,000 筆評估資料的完整答案完全匹配率選模。

## 結果

| 檢查 | 結果 |
|---|---:|
| 最佳 checkpoint | 第 26 輪 |
| 原始評估集 | 98.85%（9885/10000） |
| 額外合成診斷集 | 93.92%（9392/10000） |

數字來自既有 GPU 訓練與已儲存權重的 CPU 獨立驗證；本次 Git 整理未重新訓練。三位數訓練轉兩位數評估的比較未完成，報告保留理論分析。訓練狀態仍為暫停，並未判定收斂。

## 驗證與重現

在 repository 根目錄執行：

```bash
python -m unittest discover -s HW2/experiments/tests -v
```

Git 收錄小型資料、實驗程式、評估結果、Notebook 與最終 checkpoint；不收錄本機中間權重、optimizer/resume 狀態、備份與重複打包副本。因此，純 Git checkout 的本機完整續訓 preflight 測試會明確跳過；路徑搬移、暫停保護及最終繳交檔案完整性仍會檢查。

需要恢復訓練時，須先從自己的本機備份還原 `experiments/runs/continuation_20261104_230700/` 內的最佳模型與 `runs/lstm_e128_h512_seed2/resume.pt`，再檢查：

```bash
python HW2/experiments/scripts/run_continuation.py --check-paths
```

確認要續訓後，才明確使用 `--resume`；單純執行預設命令不會解除 STOP：

```bash
python -u HW2/experiments/scripts/run_continuation.py --resume
```

`experiments/scripts/paths.py` 集中管理路徑。歷史 JSON 中的舊位置由 relocation map 解析；原始紀錄與實驗快照保持可追溯。第一階段訓練與報告產生腳本保留當時設定，後續續訓使用 `run_continuation.py`。
