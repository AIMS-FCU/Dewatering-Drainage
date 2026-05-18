# PINN 一期模型訓練手冊

對應程式：`PINN正式版本.py`

## 1. 程式用途

此程式用於一期資料的 PINN 訓練、盲測驗證與 7 天水位自回歸預測。訓練完成後會輸出模型權重、物理參數、背景水位、流量預測、診斷圖與推論包，供後續最佳化程式使用。

## 2. 必要輸入檔

| 檔案 | 用途 |
|---|---|
| `Master_Training_Data_Continuous3.csv` | 一期訓練與預測資料 |
| `Distance_Matrix.csv` | 觀測井與抽水井距離矩陣 |

## 3. 模型架構

主要類別：

- `FlowPredictor`：預測抽水井流量與入滲/補注量 `Qin`。
- `WaterLevelPredictor`：預測觀測井與抽水井水位。
- `PINN_Feedback_Model`：整合流量模型、水位模型與物理 loss。

核心架構：

```text
歷史 window 資料
→ Dense input projection
→ Positional Encoding
→ 多層 Transformer Encoder
→ GlobalAveragePooling
→ 水位 head / 流量 head / Qin head
→ PINN 物理約束
```

## 4. 重要超參數

| 參數 | 目前設定 | 意義 | 調整建議 |
|---|---:|---|---|
| `window_size` | 336 | 輸入歷史長度，半小時資料下為 7 天 | 預測需要更長記憶才調大 |
| `epochs` | 200 | 訓練總輪數 | loss 未收斂可增加 |
| `batch_size` | 512 | 每批訓練樣本數 | OOM 時改 256 或 128 |
| `nn_lr_init` | 0.0005 | 神經網路學習率 | 不穩時降低 |
| `phys_lr` | 0.0001 | 物理參數學習率 | 物理參數震盪時降低 |
| `lambda_flow` | 150.0 | 流量 loss 權重 | 流量預測差時調整 |
| `lambda_phys_final` | 2.0 | 物理 loss 最終權重 | 水位太貼資料但物理不合理時增加 |
| `warmup_epochs` | 50 | 前期不啟用完整物理 loss | 物理 loss 太早干擾時增加 |
| `PREDICT_START/END` | 各 task 設定 | 7 天預測區間 | 改任務時必改 |
| `TEST_START/END` | 各 task 設定 | 盲測評估區間 | 不可與訓練邏輯混淆 |

## 5. 目前內建任務

程式內建三組一期任務：

| 任務 | 輸出資料夾 | 預測區間 |
|---|---|---|
| Run1 | `PINN_MAPE_Complete_Report3_Run1_56` | `2021-05-01` 到 `2021-05-08` |
| Run2 | `PINN_MAPE_Complete_Report3_Run2_56` | `2020-10-01` 到 `2020-10-08` |
| Run3 | `PINN_MAPE_Complete_Report3_Run3_56` | `2020-12-01` 到 `2020-12-08` |

## 6. 執行方式

```powershell
python PINN正式版本.py
```

若 GPU 記憶體不足，優先把 `batch_size` 從 `512` 改成 `256` 或 `128`。

## 7. 主要輸出檔

| 檔案 | 用途 |
|---|---|
| `pinn_model.weights.h5` | 模型權重 |
| `inference_pack.pkl` | 推論所需 scaler、欄位順序、config |
| `learned_T.npy` | 學得平均透水/傳輸參數 |
| `learned_C.npy` | 井損係數 |
| `calibrated_inflow_sy.npy` | 平均 Qin 與 Sy |
| `background_h_7d.npy` | 7 天背景水位預測 |
| `future_q_7d.npy` | 7 天流量預測 |
| `qin_7d_dynamic.npy` | 7 天 Qin 預測 |
| `accurate_pred_h_7d.npy` | 使用實際序列產生的基準水位預測 |
| `Prediction_vs_Actual_Obs.png` | 觀測井預測對比 |
| `Prediction_vs_Actual_PW.png` | 抽水井水位對比 |
| `Prediction_vs_Actual_Flow.png` | 流量對比 |
| `Full_Diagnostic_Report_Test.csv` | 盲測區間診斷 |
| `Full_Diagnostic_Report.csv` | 訓練區間診斷 |

## 8. 結果判讀

優先檢查：

1. `Prediction_vs_Actual_Obs.png`：觀測井水位趨勢是否合理。
2. `Full_Diagnostic_Report_Test.csv`：各井 MAE 是否集中在某幾口井。
3. `background_h_7d_trend.png`：7 天背景水位是否有不合理跳動。
4. `future_q_7d.npy`：流量是否出現長時間不合理零值或爆量。

若 `Prediction_vs_Actual_Obs.png` 的曲線形狀對但整體上下偏移，代表模型學到趨勢但水位基準偏移。此情況在二期程式會透過 `START_LEVEL_CORRECTION` 另外處理。

