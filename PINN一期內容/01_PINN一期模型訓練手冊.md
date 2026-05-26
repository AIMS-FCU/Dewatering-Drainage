# PINN 一期模型訓練手冊

本手冊對應程式：

```text
github/PINN一期內容/PINN正式版本.py
```

此程式會使用一期訓練資料建立 PINN 模型，輸出水位預測、抽水流量預測、物理參數、推論包與診斷圖表。這些產物會被二期 transfer learning 與一期最佳化使用。

## 執行前準備

請確認同一資料夾內至少有：

```text
Master_Training_Data_Continuous3.csv
Distance_Matrix.csv
PINN正式版本.py
```

其中：

- `Master_Training_Data_Continuous3.csv` 由 `data_preporcess.py` 產生。
- `Distance_Matrix.csv` 是觀測井與抽水井之間的距離矩陣，井名必須與訓練資料欄位一致。

建議先檢查資料：

```powershell
python -c "import pandas as pd; df=pd.read_csv('Master_Training_Data_Continuous3.csv', index_col=0); print(df.shape); print(df.head())"
```

## 執行方式

```powershell
cd github\PINN一期內容
python PINN正式版本.py
```

若有 GPU，程式會優先使用 GPU。若記憶體不足，先把 `config["batch_size"]` 從 `512` 降到 `256`，仍不足再降到 `128` 或 `64`。

## 模型架構概念

程式主要包含三個模型區塊：

| 類別 | 功能 |
|---|---|
| `FlowPredictor` | 預測抽水井流量與地下水補注量 `Qin` |
| `WaterLevelPredictor` | 預測觀測井與抽水井水位 |
| `PINN_Feedback_Model` | 整合神經網路預測與物理 loss，學習 `T`、`R`、`C`、`Sy` 等參數 |

輸入是一段歷史時間窗，預設 `window_size=336`。因資料為 30 分鐘一筆，336 筆代表 7 天歷史資料。

## 重要設定

設定集中在程式前段的 `config`。

| 參數 | 預設值 | 意義 | 何時調整 |
|---|---:|---|---|
| `window_size` | 336 | 輸入歷史長度，30 分鐘資料下為 7 天 | 需要更長記憶時才調大 |
| `d_model` | 64 | Transformer 特徵維度 | 模型容量不足才調大 |
| `num_heads` | 4 | Attention heads | 通常不動 |
| `num_transformer_layers` | 4 | Transformer 層數 | 記憶體不足可調小 |
| `epochs` | 200 | 訓練輪數 | loss 未收斂可增加 |
| `batch_size` | 512 | 每批樣本數 | OOM 時優先調小 |
| `nn_lr_init` | 0.0005 | 神經網路初始學習率 | loss 震盪可降低 |
| `phys_lr` | 0.0001 | 物理參數學習率 | 物理參數不穩時降低 |
| `lambda_flow` | 150.0 | 流量 loss 權重 | 流量預測偏差大時調整 |
| `lambda_phys_final` | 2.0 | 物理 loss 最終權重 | 水位趨勢不物理時調高 |
| `warmup_epochs` | 50 | 前期暫緩物理 loss | 物理 loss 太早干擾時增加 |
| `area_A` | 3319.95 | 場域面積參數 | 換場域必改 |
| `DELTA_T` | 0.5 | 每筆資料間隔，小時 | 30 分鐘資料固定為 0.5 |
| `TRAIN_CUTOFF` | 2021-07-01 | 訓練資料截止點 | 避免使用預測期之後資料 |
| `USE_KFOLD` | False | 是否啟用 5-fold | 模型比較或論文驗證時才開 |

## 預設任務

程式 `if __name__ == "__main__"` 中的 `tasks` 會連續跑多個任務。預設包含：

| 任務 | 輸出資料夾 | 預測區間 |
|---|---|---|
| Run1 | `PINN_MAPE_Complete_Report3_Run1_56` | 2021-05-01 到 2021-05-08 |
| Run2 | `PINN_MAPE_Complete_Report3_Run2_56` | 2020-10-01 到 2020-10-08 |
| Run3 | `PINN_MAPE_Complete_Report3_Run3_56` | 2020-12-01 到 2020-12-08 |

若只想跑其中一個任務，可把其他 task 註解掉。

若要改預測日期，修改 task 內：

```python
"PREDICT_START": "2021-05-01 00:00",
"PREDICT_END":   "2021-05-08 00:00",
"TEST_START":    "2021-05-01 00:00",
"TEST_END":      "2021-06-01 00:00",
```

注意：`PREDICT_START` 前必須至少有 `window_size` 筆歷史資料，否則自回歸預測沒有足夠 context。

## 主要輸出

每個 task 會在自己的 `save_folder` 產生：

| 檔案 | 說明 | 後續用途 |
|---|---|---|
| `pinn_model.weights.h5` | 模型權重 | 二期 transfer learning 可讀取 |
| `inference_pack.pkl` | scaler、欄位順序、設定 | 後續推論或重現訓練 |
| `learned_T.npy` | 學得的 transmissivity 相關參數 | 最佳化物理計算 |
| `learned_C.npy` | 井損係數 | 最佳化物理計算 |
| `calibrated_inflow_sy.npy` | 補注量與 Sy | 最佳化動態校準 |
| `qin_series.npy`、`qin_smooth.npy` | 反推補注量序列 | 診斷與分析 |
| `background_h_7d.npy` | 7 天無抽水背景水位預測 | 最佳化基準 |
| `future_q_7d.npy` | 7 天抽水量預測 | 診斷 |
| `qin_7d_dynamic.npy` | 7 天動態補注量預測 | 最佳化與診斷 |
| `accurate_pred_h_7d.npy` | 使用實際抽水條件的水位預測 | 最佳化反向驗證 |
| `Full_Diagnostic_Report_Test.csv` | 盲測區間評估 | 判斷模型可用性 |
| `Full_Diagnostic_Report.csv` | 預測區間診斷 | 報告 |
| `Prediction_vs_Actual_Obs.png` | 觀測井預測與實測比較 | 首要檢查圖 |
| `Prediction_vs_Actual_PW.png` | 抽水井水位比較 | 輔助檢查 |
| `Prediction_vs_Actual_Flow.png` | 抽水量比較 | 輔助檢查 |
| `background_h_7d_trend.png` | 背景水位趨勢 | 最佳化前檢查 |

## 結果判讀

建議依下列順序檢查：

1. 看 `Prediction_vs_Actual_Obs.png`：趨勢應大致跟實測一致，不應整段偏移太大或劇烈震盪。
2. 看 `Full_Diagnostic_Report_Test.csv`：盲測 MAE 越小越好；若觀測井 MAE 明顯大於現場可接受誤差，需要調整資料或模型。
3. 看 `Prediction_vs_Actual_Flow.png`：流量若長期偏低或偏高，最佳化會被帶偏。
4. 看 `background_h_7d_trend.png`：無抽水背景水位應平滑，若跳動過大代表自回歸或初始條件可能有問題。

## 常見調整方式

| 現象 | 優先處理 |
|---|---|
| GPU OOM | 降 `batch_size`，再降 `num_transformer_layers` 或 `d_model` |
| 水位整體偏移 | 檢查 scaler、預測起點水位、訓練資料切分 |
| 流量預測很差 | 檢查 `Qw**` 是否與 `X_**` 對齊，調整 `lambda_flow` |
| loss 很震盪 | 降 `nn_lr_init` 或 `phys_lr` |
| 預測期前 context 不足 | 將 `PREDICT_START` 往後移，或補足前 7 天資料 |
| 最佳化找不到 PINN 產物 | 確認該 `save_folder` 內有 `learned_T.npy`、`learned_C.npy`、`calibrated_inflow_sy.npy` |
