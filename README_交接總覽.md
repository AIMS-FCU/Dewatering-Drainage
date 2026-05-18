# PINN 與抽水最佳化交接總覽

## 核心流程

```text
原始/整併資料
→ 資料前處理
→ PINN 模型訓練
→ 7 天自回歸水位預測
→ 檢查 Prediction_vs_Actual_Obs.png 與診斷 CSV
→ 匯出 learned_T / learned_C / calibrated_inflow_sy / inference_pack / weights
→ 最佳化程式讀取 PINN 報告資料夾
→ 產生抽水排程、節能率、水位驗證圖
```

## 正式版程式對應

| 程式 | 用途 | 手冊 |
|---|---|---|
| `data_preporcess.py` | 一期資料前處理，產生 PINN 訓練資料 | `00_資料前處理手冊.md` |
| `data_preprocess_phase2.py` | 二期資料前處理，產生二期微調資料 | `00_資料前處理手冊.md` |
| `PINN正式版本.py` | 一期 PINN 訓練與 7 天預測 | `01_PINN一期模型訓練手冊.md` |
| `PINN正式_Phase2_Training_Transfer_二期微調.py` | 二期 PINN 遷移學習與微調 | `02_PINN二期微調手冊.md` |
| `最佳化正式版PINN一期.py` | 一期抽水最佳化 | `03_最佳化一期操作手冊.md` |
| `最佳化正式版PINN二期.py` | 二期抽水最佳化 | `04_最佳化二期操作手冊.md` |

## 推薦執行順序

1. 確認原始資料或 30 分鐘對齊資料檔存在。
2. 執行資料前處理程式。
3. 確認已產生 `Master_Training_Data_Continuous3.csv` 或 `Phase2_Training_Data2.csv`。
4. 先跑 PINN 訓練程式，產生報告資料夾。
5. 檢查 PINN 報告資料夾內的預測圖與診斷表。
6. 將最佳化程式的 `pinn_report_path` 指到該 PINN 報告資料夾。
7. 執行最佳化程式。
8. 檢查 `Optimization_Report.png`、`Well_Open_Comparison.csv`、`Well_Power_Summary.csv`。

## 重要觀念

- `Raw MAE` 代表模型原始絕對水位預測能力。
- `Corrected MAE` 代表起始水位錨定後的 7 天趨勢預測能力。
- 最佳化不是直接訓練神經網路，而是使用 PINN 提供的物理參數與背景水位，求解抽水排程。
- 接手者調參時必須記錄：資料區間、模型報告資料夾、主要超參數、MAE、節能率、是否有起始水位校正。
