# PINN 二期 Transfer Learning 微調手冊

本手冊對應程式：

```text
github/PINN二期內容/PINN正式_Phase2_Training_Transfer_二期微調.py
```

此程式會讀取二期資料，並嘗試載入一期 PINN 權重作為初始化，再針對二期井位、二期觀測井與二期時間區間進行微調。

## 執行前準備

請確認同一資料夾內至少有：

```text
Phase2_Training_Data2.csv
Distance_Matrix_Phase2.csv
PINN正式_Phase2_Training_Transfer_二期微調.py
```

另需確認一期權重存在。程式預設：

```python
"phase1_weights_path": "PINN_MAPE_Complete_Report3_Run1_56/pinn_model.weights.h5"
```

若一期結果資料夾不在二期資料夾內，請改成正確路徑，例如：

```python
"phase1_weights_path": "../PINN一期內容/PINN_MAPE_Complete_Report3_Run1_56/pinn_model.weights.h5"
```

## 執行方式

```powershell
cd github\PINN二期內容
python PINN正式_Phase2_Training_Transfer_二期微調.py
```

二期程式預設使用 mixed precision，GPU 記憶體通常比一期省，但若仍 OOM，將 `batch_size` 從 `256` 改成 `128` 或 `64`。

## 二期資料與井位

二期觀測井預設：

```text
PW02, PW03, PW04
```

二期抽水井預設：

```text
PW01, PW010, PW011, PW05, PW06, PW07, PW08, PW09
```

請注意 `PW010`、`PW011` 是三碼井號。資料欄位、距離矩陣、最佳化設定必須都使用同一種命名。

## 重要設定

設定集中在程式前段 `config`。

| 參數 | 預設值 | 說明 |
|---|---:|---|
| `window_size` | 336 | 7 天歷史 context，30 分鐘資料下為 336 筆 |
| `epochs` | 300 | 微調總輪數 |
| `batch_size` | 256 | 每批樣本數 |
| `nn_lr_init` | 0.0001 | 微調學習率，低於一期避免破壞既有權重 |
| `warmup_epochs` | 150 | 物理 loss 暖身 |
| `HEAD_WARMUP_EPOCHS` | 30 | 前 30 epoch 先訓練輸出頭，降低 transfer shock |
| `EARLY_STOPPING_PATIENCE` | 40 | loss 長期未改善時停止 |
| `START_LEVEL_CORRECTION` | True | 用預測起點實測水位校正 7 天背景水位 |
| `training_data_path` | `Phase2_Training_Data2.csv` | 二期訓練資料 |
| `allow_partial_transfer` | True | 允許部分權重載入，適合一期與二期欄位數不同 |
| `phase1_weights_path` | 一期權重路徑 | 執行前最常需要修改 |

## 預設任務

程式中的 `tasks` 預設啟用兩個 Full 微調任務：

| 任務 | 輸出資料夾 | 預測區間 | 盲測區間 |
|---|---|---|---|
| Early | `微調/Report_Full_0301_Early` | 2026-03-01 到 2026-03-08 | 2026-02-01 到 2026-02-15 |
| Late | `微調/Report_Full_0324_Late` | 2026-03-24 到 2026-03-31 | 2026-03-01 到 2026-03-15 |

程式內也保留 1Month、3Month 的 task 範例，但目前是註解狀態。若要啟用，移除對應區塊的註解即可。

## Transfer Learning 流程

二期微調大致流程：

1. 讀取 `Phase2_Training_Data2.csv`。
2. 讀取 `Distance_Matrix_Phase2.csv`。
3. 建立二期欄位對應與 scaler。
4. 建立與二期欄位數相符的 PINN 模型。
5. 嘗試載入 `phase1_weights_path`。
6. 若一期與二期模型部分層不相容，依 `allow_partial_transfer=True` 跳過不相容權重。
7. 先執行 head warmup，再進入完整微調。
8. 產生盲測報告、7 天自回歸背景水位與診斷圖。

## 起始水位校正

`START_LEVEL_CORRECTION=True` 時，程式會用 `PREDICT_START` 當下的實測水位，校正 7 天自回歸預測的起始偏移。

輸出會多出：

```text
background_h_7d_raw.npy
background_h_7d.npy
start_level_bias_correction.csv
```

含義：

- `background_h_7d_raw.npy`：校正前的背景水位。
- `background_h_7d.npy`：校正後的背景水位，最佳化主要使用這份。
- `start_level_bias_correction.csv`：每口觀測井套用的偏移量。

若 `Applied_Bias` 過大，代表模型在預測起點已有明顯偏移，建議回頭檢查資料期間、scaler、二期欄位與一期權重是否合適。

## 主要輸出

每個 task 會輸出到 `save_folder`：

| 檔案 | 說明 | 後續用途 |
|---|---|---|
| `pinn_model.weights.h5` | 二期微調後模型權重 | 後續二期再微調或推論 |
| `inference_pack.pkl` | scaler、欄位順序、config | 重現與推論 |
| `learned_T.npy` | 二期學得的 T | 最佳化 |
| `learned_C.npy` | 二期井損係數 | 最佳化 |
| `calibrated_inflow_sy.npy` | 二期 Qin 與 Sy | 最佳化 |
| `background_h_7d.npy` | 7 天無抽水背景水位 | 二期最佳化關鍵輸入 |
| `future_q_7d.npy` | 7 天流量預測 | 診斷 |
| `qin_7d_dynamic.npy` | 7 天動態補注量 | 最佳化與診斷 |
| `accurate_pred_h_7d.npy` | 使用實際抽水條件的水位預測 | 最佳化反向推導 |
| `Full_Diagnostic_Report_Test.csv` | 盲測評估 | 模型品質判斷 |
| `Full_Diagnostic_Report.csv` | 預測區間診斷 | 報告 |
| `Prediction_vs_Actual_Obs.png` | 觀測井水位預測圖 | 首要檢查 |
| `Prediction_vs_Actual_PW.png` | 抽水井水位預測圖 | 輔助檢查 |
| `Prediction_vs_Actual_Flow.png` | 流量預測圖 | 輔助檢查 |
| `background_h_7d_trend.png` | 背景水位趨勢 | 最佳化前檢查 |

## 結果判讀

建議先看：

1. `Prediction_vs_Actual_Obs.png`：二期觀測井 `PW02/PW03/PW04` 趨勢是否合理。
2. `Full_Diagnostic_Report_Test.csv`：盲測 MAE 是否可接受。
3. `start_level_bias_correction.csv`：校正偏移是否過大。
4. `background_h_7d_trend.png`：背景水位是否平滑且符合現場直覺。

若 Raw MAE 很大但 Corrected MAE 明顯改善，代表模型趨勢可用，但起點偏移需要校正。若兩者都很差，通常是資料欄位、訓練區間或 transfer 權重不合適。

## 最常修改的位置

| 需求 | 修改位置 |
|---|---|
| 換一期權重 | `phase1_weights_path` |
| 換二期資料檔 | `training_data_path` |
| 只跑一個任務 | 註解掉 `tasks` 內不需要的 dict |
| 改預測日期 | task 內 `PREDICT_START`、`PREDICT_END` |
| 改盲測日期 | task 內 `TEST_START`、`TEST_END` |
| 啟用 1Month 或 3Month 微調 | 取消註解對應 task |
| GPU 記憶體不足 | 降 `batch_size` |
| 校正偏移限制過大 | 設定 `START_LEVEL_CORRECTION_MAX_ABS`，例如 `2.0` |

## 新資料進來時

若二期新資料只是增加時間長度，井位與井名沒有變，通常流程是：

```powershell
python data_preprocess_phase2.py
python PINN正式_Phase2_Training_Transfer_二期微調.py
```

若新資料包含新增井、刪除井、井名改變或新的實測座標，請先更新：

```text
generate_mock_distance_matrix.py
```

主要修改：

- `pumping_wells`：二期抽水井。
- `obs_wells`：二期觀測井。
- `coords`：每口井座標。

執行後會重新產生：

```text
Distance_Matrix_Phase2.csv
```

更完整的新資料處理、欄位命名與距離矩陣更新步驟，請看：

```text
00_二期新資料與距離矩陣更新說明.md
```
