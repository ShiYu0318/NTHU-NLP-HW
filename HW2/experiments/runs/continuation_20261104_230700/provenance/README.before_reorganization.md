# Independent post-submission continuation

This directory is separate from HW2. All training artifacts and logs stay here; the submitted files are read only. The unchanged original CSVs are read from ../HW2/data.

Resume from the complete epoch-25 state (195325 Adam steps, learning rate 0.00025), beginning at epoch 26. Preserve the epoch-24 best checkpoint with supplied-evaluation EM 0.9874. Continue the same two-layer LSTM, data, teacher forcing, loss and greedy evaluation; no evaluation labels are added to training.

No wall-clock deadline, target score or epoch cap. Stop operationally after 12 completed epochs without an improvement greater than 0.0001 EM, including 6 consecutive completed epochs trained at learning rate <= 1e-5. Keep every strictly higher full-evaluation score. This does not prove a global optimum.

`training.log` contains progress; `runs/lstm_e128_h512_seed2/history.json` contains evaluation history; `checkpoint.pt` is the best model; `resume.pt` under the run directory includes optimizer and scheduler. `outcome.json` under the run directory records termination. A `STOP` file requests a safe stop. Restart the same runner to resume. The runner holds a process lock to prevent duplicate training and validates submitted-file hashes before and after training and at each completed evaluation. It does not publish or repackage the submitted homework.
