import pandas as pd
import numpy as np
import os
import sys
import re

# 強制 Windows終端機使用 utf-8 輸出，避免 emoji 報錯
sys.stdout.reconfigure(encoding='utf-8')

# --- 第二期資料專用處理邏輯 ---

input_file = 'sensor_data_30min_aligned.csv'
output_file = 'Phase2_Training_Data2.csv'

print(f"--- 開始處理第二期資料: {input_file} ---")
if not os.path.exists(input_file):
    print(f"❌ 找不到檔案: {input_file}")
    sys.exit()

# 讀取已對齊的總表
print("載入資料中，請稍候...")
master_df = pd.read_csv(input_file, index_col=0, low_memory=False)
master_df.index = pd.to_datetime(master_df.index, errors='coerce')

# 清洗數值欄位（移除可能存在的逗號並轉為數值型態）
for col in master_df.columns:
    if master_df[col].dtype == 'object':
        master_df[col] = master_df[col].astype(str).str.replace(',', '', regex=False).str.strip()
    master_df[col] = pd.to_numeric(master_df[col], errors='coerce')

# 💡 核心修正區：過濾 PW 抽水井的異常水位
pw_cols = [c for c in master_df.columns if str(c).upper().startswith('PW') and 'KWH' not in str(c).upper()]
print(f"\n--- 執行 PW 井水位異常值過濾 (>0 或 <-30 設為空值並插補) ---")
for col in pw_cols:
    # 標記異常值為 NaN
    mask_outlier = (master_df[col] > 0) | (master_df[col] < -30)
    outlier_count = mask_outlier.sum()
    if outlier_count > 0:
        master_df.loc[mask_outlier, col] = np.nan
        print(f"  ✅ {col}: 過濾了 {outlier_count} 筆異常值")
    
    # 針對產生的 NaN 進行插值與前向填充
    master_df[col] = master_df[col].interpolate(method='linear').ffill().bfill()

# 💡 核心修正區：處理電力資料產出 P_gr 與 X，並與 QW 流量連動
kwh_cols = [c for c in master_df.columns if 'KWH' in str(c).upper()]

print("\n--- 執行電力與流量連動校正 (產生 X 欄位) ---")
for col in kwh_cols:
    # 擷取數字部分，保留字串格式以免 "010" 變成 "10"
    match = re.search(r'\d+', str(col))
    if match:
        tag = match.group()  # 例如 "01", "010", "02"
        
        # 1. 產生電力特徵 (P_gr 與 開關 X)
        delta_kwh = master_df[col].diff().clip(lower=0)
        master_df[f'P_gr_{tag}'] = (delta_kwh * 2).fillna(0)
        master_df[f'X_{tag}'] = (delta_kwh > 0.01).astype(int)
        
        # 2. 💡 [連動處理]：將流量 QW 與 開關 X 相乘
        qw_col_upper = f"QW{tag}" # 針對第二期大寫的 QW010 等
        qw_col_lower = f"Qw{tag}"
        
        target_qw = None
        if qw_col_upper in master_df.columns: target_qw = qw_col_upper
        elif qw_col_lower in master_df.columns: target_qw = qw_col_lower
            
        if target_qw:
            # 若 X 為 0，則流量強迫歸零
            master_df[target_qw] = master_df[target_qw].fillna(0) * master_df[f'X_{tag}']
            print(f"  ✅ 已校正 {target_qw}：新增 X_{tag}，開關為 0 時流量已歸零")

# --- 最終空值處理 ---
# 雨量補 0 (若有雨量欄位)
rain_cols = [c for c in master_df.columns if any(k in str(c).lower() for k in ['rain', '雨量'])]
for rc in rain_cols:
    master_df[rc] = master_df[rc].fillna(0)

# 其他欄位進行前向填充 (避免出現 NaN)
master_df = master_df.ffill().fillna(0)

# 儲存結果
print(f"\n正在儲存至 {output_file}...")
master_df.to_csv(output_file, encoding='utf-8-sig')

print(f"\n✅ 全部處理完成！")
print(f"整合修正點：")
print(f"1. 讀取了已合併的 {input_file}。")
print(f"2. 成功解析 KWH 並生成了所有的 X 欄位 (例如 X_01, X_010)。")
print(f"3. 流量連動：沒開機 (X=0) 就沒流量。")
print(f"4. 總筆數: {len(master_df)}")
print(f"5. 結果已儲存為: {output_file}")
