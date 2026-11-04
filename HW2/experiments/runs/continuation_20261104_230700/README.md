# 第 25 輪之後的獨立續訓

本目錄由原專案根目錄的 `HW2_continuation_20261104_230700` 搬入。

- 最佳模型：第 26 輪，完整生成 EM 98.85%；`checkpoint.pt` / `checkpoint_cpu.pt`。
- 完整評估紀錄至第 28 輪；第 29 輪中途暫停。
- `runs/lstm_e128_h512_seed2/resume.pt` 保存最新訓練狀態，包含 optimizer、scheduler 與 batch 位置。
- `training.log` 為歷史訓練輸出；`final_verification.json` 為最後完整獨立驗證。
- `best_archive/epoch_26/` 保存第 26 輪權重與獨立驗證。
- `resubmission/` 是 23:59 前重新提交版本的副本。
- `training_driver.ipynb` 為本實驗的固定程式快照；`submitted_notebook_reference.ipynb` 是當初啟動續訓時的繳交版參考。
- `provenance/` 保留搬移前 README 與重新提交前的原始保護清單。

啟動與驗證腳本已集中至 `../../scripts/`，統一讀取 `HW2/data` 及 `experiments/diagnostics`，不再依賴本目錄原先位於 repository 根目錄的層級。

目前仍保留 STOP，沒有執行訓練。檢查及恢復指令見 [HW2 說明](../../../README.md)。
