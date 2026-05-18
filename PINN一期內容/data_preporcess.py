import pandas as pd
import numpy as np
import os
import chardet
import re

def smart_read_csv(file_path):
    """
    專門處理觀測資料：自動偵測編碼、精確定位標題、移除指定欄位、清洗數值(去逗號)。
    """
    print(f"--- 處理中: {file_path} ---")
    
    if not os.path.exists(file_path):
        print(f"  ❌ 找不到檔案: {file_path}")
        return None

    try:
        # 1. 偵測編碼
        with open(file_path, 'rb') as f:
            rawdata = f.read(30000) 
            result = chardet.detect(rawdata)
            detected_enc = result['encoding']
            if detected_enc is None or 'utf-8' in detected_enc.lower():
                detected_enc = 'utf-8-sig'
        
        # 2. 尋找真正的標題行
        df_scan = pd.read_csv(file_path, encoding=detected_enc, header=None, nrows=10)
        header_row = 0
        for i, row in df_scan.iterrows():
            if any("時間" in str(val) for val in row.values):
                header_row = i
                break
        
        # 3. 正式讀取
        df = pd.read_csv(file_path, encoding=detected_enc, header=header_row, low_memory=False)
        
        # 4. 欄位清洗
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df.columns = df.columns.str.strip()
        
        cols_to_drop = ['溫度', '電壓', '訊號強度', 'Temp', 'Voltage']
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
        
        # 5. 時間索引處理
        time_col = df.columns[0]
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        df = df.dropna(subset=[time_col]).set_index(time_col)
        
        # 6. 數值清洗：移除逗號
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace(',', '', regex=False).str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 7. 重採樣 (Resample) 至 30 分鐘，確保時序對齊
        df = df.resample('30min').mean()
        
        return df

    except Exception as e:
        print(f"  ❌ 讀取失敗: {e}")
        return None

# --- 主要執行邏輯 ---

files = {
    'well_drawdown': '抽水水位合併總表.csv',
    'obs_level': '水位合併總表.csv',
    'rain': 'rain總表.csv',
    'power': '電力合併總表.csv',
    'flow': '抽水量合併總表.csv'
}

dfs = {key: smart_read_csv(path) for key, path in files.items()}

if all(v is not None for v in dfs.values()):
    print("\n--- 開始合併與特徵工程 ---")
    
    # 橫向合併
    master_df = pd.concat(dfs.values(), axis=1)
    
    # 💡 核心修正區：處理電力資料產出 P_gr 與 X，並與 Qw 流量連動
    kwh_cols = [c for c in master_df.columns if 'KWH' in str(c).upper()]
    
    print("--- 執行電力與流量連動校正 ---")
    for col in kwh_cols:
        match = re.search(r'\d+', str(col))
        well_num = int(match.group()) if match else None
        
        if well_num is not None:
            tag = f"{well_num:02d}"  # 產生 01, 02...
            
            # 1. 產生電力特徵 (P_gr 與 開關 X)
            delta_kwh = master_df[col].diff().clip(lower=0)
            master_df[f'P_gr_{tag}'] = (delta_kwh * 2).fillna(0)
            master_df[f'X_{tag}'] = (delta_kwh > 0.01).astype(int)
            
            # 2. 💡 [連動處理]：將流量 Qw 與 開關 X 相乘
            qw_col = f"Qw{tag}"
            if qw_col in master_df.columns:
                # 若 X 為 0，則流量 Qw 強制歸零
                master_df[qw_col] = master_df[qw_col].fillna(0) * master_df[f'X_{tag}']
                print(f"  ✅ 已校正 {qw_col}：開關為 0 時流量已歸零")

    # --- 最終空值處理 ---
    # 1. 雨量補 0
    rain_cols = [c for c in master_df.columns if any(k in str(c).lower() for k in ['rain', '雨量'])]
    for rc in rain_cols:
        master_df[rc] = master_df[rc].fillna(0)

    # 2. 其他欄位進行前向填充
    master_df = master_df.ffill().fillna(0)

    # 儲存結果 (命名為 Continuous3 以示區別)
    output_name = 'Master_Training_Data_Continuous3.csv'
    master_df.to_csv(output_name, encoding='utf-8-sig')
    
    print(f"\n✅ 全部處理完成！")
    print(f"整合修正點：")
    print(f"1. 流量連動：新增 X_{tag} 與 Qw{tag} 的連動，沒開機就沒流量。")
    print(f"2. 數據清洗：自動移除 KWH 的負增長(clip)並補齊空值。")
    print(f"3. 總筆數: {len(master_df)}")
    print(f"4. 時間範圍: {master_df.index.min()} 到 {master_df.index.max()}")
    print(f"5. 結果檔案: {output_name}")

else:
    print("\n❌ 合併失敗：請檢查所有 CSV 檔案。")