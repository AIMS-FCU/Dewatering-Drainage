# PINN 二期遷移學習與微調手冊

對應程式：`PINN正式_Phase2_Training_Transfer_二期微調.py`

## 1. 程式用途

此程式用於二期資料訓練，並可載入一期 PINN 權重進行 transfer learning。主要目的：

- 使用二期資料微調一期模型。
- 比較不同訓練資料範圍對 7 天預測的影響。
- 輸出二期最佳化需要的 PINN 參數與背景水位。

## 2. 必要輸入檔

| 檔案 | 用途 |
|---|---|
| `Phase2_Training_Data2.csv` | 二期訓練資料 |
| `Distance_Matrix_Phase2.csv` | 二期距離矩陣 |
| `PINN_MAPE_Complete_Report3_Run1_56/pinn_model.weights.h5` | 一期權重，供遷移學習使用 |

若找不到 `Phase2_Training_Data2.csv`，程式會嘗試改用 `Phase2_Training_Data.csv`。

## 3. 二期資料欄位邏輯

觀測井候選：

```text
PW02, PW03, PW04
```

抽水井候選：

```text
PW01, PW010, PW011, PW05, PW06, PW07, PW08, PW09
```

流量欄位會依抽水井名稱自動尋找 `Qw**` 或 `QW**`。

## 4. 模型架構

沿用一期雙模型架構：

```text
FlowPredictor
WaterLevelPredictor
PINN_Feedback_Model
```

二期新增/強化重點：

- 可載入一期權重。
- 可局部載入不完全匹配的權重。
- 載入權重後會重置物理參數 `T/R/C/Sy`，避免一期地質參數直接套到二期。
- 有 `HEAD_WARMUP_EPOCHS`，先訓練輸出頭，再解凍全模型。
- 有 `START_LEVEL_CORRECTION`，可做起始水位錨定。

## 5. 重要超參數

| 參數 | 目前設定 | 意義 | 調整建議 |
|---|---:|---|---|
| `window_size` | 336 | 輸入歷史 7 天半小時資料 | 通常不動 |
| `epochs` | 300 | 總訓練輪數 | 微調不收斂可增加 |
| `batch_size` | 256 | 每批樣本數 | OOM 改 128 或 64 |
| `nn_lr_init` | 0.0001 | 遷移學習微調學習率 | 二期不穩時降低 |
| `HEAD_WARMUP_EPOCHS` | 30 | 先只訓練輸出頭 | 資料少建議 20-50 |
| `EARLY_STOPPING_PATIENCE` | 40 | loss 無改善後停止 | 想更久訓練可增加 |
| `warmup_epochs` | 150 | 物理 loss 暖身 | physics 太早干擾時增加 |
| `lambda_flow` | 150.0 | 流量 loss 權重 | 流量預測差時調整 |
| `lambda_phys_final` | 2.0 | 物理 loss 權重 | 水位趨勢不物理時調整 |
| `START_LEVEL_CORRECTION` | True | 起始水位錨定 | 報告需同時看 raw/corrected |
| `START_LEVEL_CORRECTION_MAX_ABS` | None | 校正最大幅度 | 想限制最大修正可設 2.0 |

## 6. `HEAD_WARMUP_EPOCHS` 是什麼

`HEAD_WARMUP_EPOCHS=30` 代表前 30 epoch 只訓練輸出層與 bias，凍結 Transformer backbone。目的：

```text
先讓模型輸出層適應二期水位/流量尺度，
避免一開始就把一期學到的時序特徵全部改亂。
```

之後程式會解凍全模型，進入完整微調。

## 7. `START_LEVEL_CORRECTION` 邏輯

此功能只使用 `PREDICT_START` 當下已知水位，不使用未來 7 天答案。

```text
校正量 = 起始時刻實測水位 - 模型第一步預測水位
校正後水位 = 原始 7 天預測水位 + 校正量
```

可解釋為：

```text
將預測起點錨定到現場已知初始水位，後續 7 天走勢仍由模型自回歸決定。
```

報告時建議同時提供：

- `background_h_7d_raw.npy`：未校正預測。
- `background_h_7d.npy`：校正後預測。
- `start_level_bias_correction.csv`：每口井校正量。

若 `Applied_Bias` 超過 2 m，代表模型絕對水位基準偏移明顯，需特別註明。

## 8. 微調任務設計

程式內目前主要有 Full 任務，1Month / 3Month 任務可取消註解使用。

| 類型 | 訓練資料概念 | 用途 |
|---|---|---|
| Full | 使用指定 cutoff 前可用資料 | 看完整歷史資料效果 |
| 1Month | 只用 2026-01-01 到 2026-02-01 | 看少量資料微調效果 |
| 3Month | 用 2026-01-01 到預測日前夕 | 看近期資料是否改善預測 |

注意：推論時仍會使用 `PREDICT_START` 前 7 天作為 context window，這是實務上可取得的現場歷史狀態。

## 9. 執行方式

```powershell
python PINN正式_Phase2_Training_Transfer_二期微調.py
```

## 10. 主要輸出檔

| 檔案 | 用途 |
|---|---|
| `pinn_model.weights.h5` | 二期微調後權重 |
| `inference_pack.pkl` | 二期推論資訊 |
| `learned_T.npy` | 二期學得 T |
| `learned_C.npy` | 二期井損係數 |
| `calibrated_inflow_sy.npy` | Qin 與 Sy |
| `background_h_7d_raw.npy` | 起始校正前水位預測 |
| `background_h_7d.npy` | 起始校正後水位預測 |
| `start_level_bias_correction.csv` | 起始水位校正量 |
| `accurate_pred_h_7d.npy` | 含真實序列基準水位 |
| `Prediction_vs_Actual_Obs.png` | 觀測井水位對比 |
| `Prediction_vs_Actual_PW.png` | 抽水井水位對比 |
| `Prediction_vs_Actual_Flow.png` | 流量對比 |

## 11. 判讀規則

- Raw MAE 好：模型本身絕對水位基準準。
- Raw MAE 差、Corrected MAE 好：趨勢有學到，但基準偏移。
- Corrected MAE 仍差：模型未學到該 7 天動態，需檢查抽水型態、資料區間、流量欄位。
- 三個月不一定一定比一個月好；若三個月資料包含不同抽水型態，可能讓模型基準漂移。

