# PINN 地下水位預測與抽水最佳化交接總覽

本資料夾整理了從資料前處理、PINN 模型訓練、二期 transfer learning 微調，到抽水排程最佳化的完整流程。若是第一次接手，建議先照本頁的「標準執行順序」跑一次，再依各子手冊調整參數。

## 資料夾內容

| 位置 | 主要檔案 | 用途 |
|---|---|---|
| `00_資料前處理手冊.md` | `data_preporcess.py`、`data_preprocess_phase2.py` | 說明如何把原始監測資料整理成 PINN 可讀的訓練 CSV |
| `PINN一期內容/01_PINN一期模型訓練手冊.md` | `PINN正式版本.py` | 一期 PINN 訓練、7 天自回歸預測、輸出模型與報告 |
| `PINN二期內容/02_PINN二期微調手冊.md` | `PINN正式_Phase2_Training_Transfer_二期微調.py` | 讀取一期權重，使用二期資料做 transfer learning |
| `PINN二期內容/00_二期新資料與距離矩陣更新說明.md` | `generate_mock_distance_matrix.py` | 新二期資料進來時，說明如何更新資料欄位、井位與 `Distance_Matrix_Phase2.csv` |
| `最佳化一期內容/03_最佳化一期操作手冊.md` | `最佳化正式版PINN一期.py` | 使用一期 PINN 產物推估節能抽水排程 |
| `最佳化二期內容/04_最佳化二期操作手冊.md` | `最佳化正式版PINN二期.py` | 使用二期微調結果做二期抽水最佳化 |
| `05_常見錯誤與除錯.md` | 所有程式 | GPU、資料欄位、路徑、字型、最佳化不可行等問題排查 |

## 標準執行順序

1. 確認 Python 環境與套件。

   ```powershell
   python -c "import pandas,numpy,tensorflow,sklearn,pulp,matplotlib; print('ok')"
   ```

2. 產生一期訓練資料。

   ```powershell
   cd github\PINN一期內容
   python data_preporcess.py
   ```

   產物：`Master_Training_Data_Continuous3.csv`

3. 訓練一期 PINN。

   ```powershell
   python PINN正式版本.py
   ```

   預設會依程式中的 `tasks` 連續跑 3 個任務，輸出到：

   ```text
   PINN_MAPE_Complete_Report3_Run1_56
   PINN_MAPE_Complete_Report3_Run2_56
   PINN_MAPE_Complete_Report3_Run3_56
   ```

4. 產生二期訓練資料。

   ```powershell
   cd ..\PINN二期內容
   python data_preprocess_phase2.py
   ```

   產物：`Phase2_Training_Data2.csv`

5. 執行二期 PINN 微調。

   ```powershell
   python PINN正式_Phase2_Training_Transfer_二期微調.py
   ```

   預設讀取一期權重：

   ```text
   PINN_MAPE_Complete_Report3_Run1_56/pinn_model.weights.h5
   ```

   預設輸出到：

   ```text
   微調/Report_Full_0301_Early
   微調/Report_Full_0324_Late
   ```

6. 執行最佳化。

   一期：

   ```powershell
   cd ..\最佳化一期內容
   python 最佳化正式版PINN一期.py
   ```

   二期：

   ```powershell
   cd ..\最佳化二期內容
   python 最佳化正式版PINN二期.py
   ```

   執行前務必打開最佳化程式，確認 `opt_config["pinn_report_path"]` 指向正確的 PINN 結果資料夾。

## 最重要的輸入與輸出

| 階段 | 必要輸入 | 主要輸出 | 下一步使用者 |
|---|---|---|---|
| 一期前處理 | 原始水位、流量、電力、雨量 CSV | `Master_Training_Data_Continuous3.csv` | 一期 PINN、一期最佳化 |
| 一期 PINN | `Master_Training_Data_Continuous3.csv`、`Distance_Matrix.csv` | `pinn_model.weights.h5`、`inference_pack.pkl`、`learned_T.npy`、`learned_C.npy`、`calibrated_inflow_sy.npy`、`background_h_7d.npy`、診斷圖表 | 二期微調、一期最佳化 |
| 二期前處理 | `sensor_data_30min_aligned.csv` | `Phase2_Training_Data2.csv` | 二期 PINN、二期最佳化 |
| 二期 PINN | `Phase2_Training_Data2.csv`、`Distance_Matrix_Phase2.csv`、一期權重 | 二期權重、二期物理參數、7 天背景水位、診斷圖表 | 二期最佳化 |
| 最佳化 | PINN 結果資料夾、訓練 CSV、距離矩陣 | `Optimization_Report.png`、`Well_Open_Comparison.csv`、二期另有 `Well_Power_Summary.csv` | 操作建議與節能比較 |

## 路徑使用原則

各程式多使用相對路徑，所以建議在程式所在資料夾執行。例如要跑二期微調，就先進入 `github\PINN二期內容` 再執行 Python。若要跨資料夾使用模型權重或資料檔，請在程式內改成正確的相對路徑或絕對路徑。

最容易忘記的是最佳化程式的 `pinn_report_path`。這個路徑必須指向已經產生下列檔案的 PINN 結果資料夾：

```text
learned_T.npy
learned_C.npy
calibrated_inflow_sy.npy
background_h_7d.npy
accurate_pred_h_7d.npy
```

## 判讀結果的基本順序

1. 先看 PINN 的 `Prediction_vs_Actual_Obs.png`，確認觀測井水位趨勢是否合理。
2. 再看 `Full_Diagnostic_Report_Test.csv`，確認盲測 MAE、MAPE、WAPE 是否可接受。
3. 若有 `background_h_7d.npy` 與 `background_h_7d_trend.png`，檢查無抽水背景水位是否平滑，不應出現不合理跳動。
4. 最後看最佳化的 `Optimization_Report.png` 與 `Well_Open_Comparison.csv`，比較實際排程與最佳化排程的抽水量、開井數、電量與水位限制。

## 常見修改點

| 需求 | 修改位置 |
|---|---|
| 改預測日期 | PINN 程式中 `tasks` 的 `PREDICT_START`、`PREDICT_END` |
| 改盲測區間 | PINN 程式中 `TEST_START`、`TEST_END` |
| 改訓練截止日期 | PINN 程式中 `TRAIN_CUTOFF` |
| GPU 記憶體不足 | `batch_size` 從 512/256 降到 128/64 |
| 二期換一期權重 | 二期 PINN 的 `phase1_weights_path` |
| 最佳化換 PINN 結果 | 最佳化程式的 `pinn_report_path` |
| 最佳化換分析日期 | 最佳化程式的 `ANALYSIS_START`、`ANALYSIS_END` |
| 二期最佳化改目標水位策略 | `TARGET_MODE`、`TARGET_QUANTILE`、`FIXED_TARGET_H` |

## 建議交接檢查清單

- 已確認所有 CSV 可用 Excel 或 pandas 正常開啟。
- 已確認時間欄位是 30 分鐘或 1 小時等距資料，且沒有大量缺值。
- 已確認距離矩陣的井名和訓練資料欄位一致。
- 已保存每次訓練使用的程式版本、日期設定、輸出資料夾。
- 最佳化結果只當作建議排程，仍需搭配現場水位、安全水位與設備限制確認。
