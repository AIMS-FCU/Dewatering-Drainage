import pandas as pd
import numpy as np
from pulp import *
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# ==========================================
# 1. 系統核心配置
# ==========================================
opt_config = {
    "ANALYSIS_START": "2020-09-11 11:30", 
    "ANALYSIS_END":  "2021-06-01 00:00",
    "AREA": 3319.95,                # 開挖面積 (m2)
    "SY": 0.21,                    # 比給水度 (Specific Yield) - 來自第二個程式
    "DEPTH": 19.55,                 # 開挖深度 (正值)
    "STATIC_WATER_LEVEL": -11.0, 
    "delta_t": 0.5,                 # 時間步長 (hr)
    "η": 0.4,                       # 抽水機效率
    "HEAD": 33.0,                   # 揚程 (m)
    "SF": 1.1,                      # 安全係數
    "SAFETY_BUFFER": 0.5,           # 額外安全緩衝 (公尺)
    "rw": 0.45,                     # 井徑
    "R_init": 500.0,                # 影響半徑
    "FIXED_PRICE_PER_KWH": 4.1,      
    "TARGET_FLOOR": "B6F",      
    "FIXED_TARGET_H": -20,          # 目標水位標高
    "WEIGHT_MAP": {
        "B6F": 12000.0, "B5F": 17585.1, "B4F": 22320.6, "B3F": 27011.1, "B2F": 31811.4, 
        "B1F": 37021.7, "1F": 42263.8,  "2F": 43561.7,  "10F": 51678.3, "PRF": 59291.5
    },
    "SIM_HOURS": 168,               # 預測模擬時數 (7天)
    "MIN_ACTIVE_WELLS": 0,      
    "BUFFER_HOURS": 12, 
    "pinn_report_path": "PINN_MAPE_Complete_Report3",
    "WELL_LIST": ["PW01", "PW02", "PW03", "PW04", "PW06", "PW07", "PW08", "PW09", "PW010", "PW011", "PW012", "PW013"],
    "OBS_LIST": ['PA', 'PB', 'PC', 'FPS7', 'FPS8', 'FPS9', 'FPS2', 'FPS3', 'FPS4', 'FPS5', 'FPS6']
}

# 功率轉流量轉換係數 (m3/kWh)
kW_to_m3h = (1.0 * 3600 * opt_config["η"]) / (9.81 * opt_config["HEAD"])

# ==========================================
# 2. 核心最佳化引擎
# ==========================================
def run_pulp_optimization(pinn_params, well_powers, base_h_sim, dist_matrix, target_h):
    T = pinn_params['T']
    C = pinn_params['C']
    num_wells, num_obs = len(well_powers), base_h_sim.shape[1]
    sim_hours = opt_config["SIM_HOURS"]
    
    prob = LpProblem("Smart_Pumping_Optimization", LpMinimize)
    times, wells, obs = range(sim_hours), range(num_wells), range(num_obs)
    
    x = LpVariable.dicts("X", (times, wells), cat='Binary')
    slack = LpVariable.dicts("S", (times, obs), lowBound=0, cat='Continuous')
    
    # 目標函數：極小化總耗電量 + 懲罰項（防止水位超過目標）
    total_energy_expr = lpSum([x[t][i] * well_powers[i] * opt_config["delta_t"] for t in times for i in wells])
    slack_penalty_expr = lpSum([slack[t][j] * 1000000 for t in times for j in obs])
    prob += total_energy_expr + slack_penalty_expr
    
    # 建立影響矩陣 (Influence Matrix)
    inf_matrix = np.zeros((num_wells, num_obs))
    for i in wells:
        for j in obs:
            r = dist_matrix[j, i] if dist_matrix[j, i] > 0.01 else opt_config["rw"]
            # 物理降水模型 (Theis/Thiem 簡化版)
            inf_matrix[i, j] = (kW_to_m3h / (2 * np.pi * T)) * np.log(opt_config["R_init"] / r) + \
                               (C[i] * kW_to_m3h if dist_matrix[j, i] < 1.0 else 0)

    for t in times:
        prob += lpSum([x[t][i] for i in wells]) >= opt_config["MIN_ACTIVE_WELLS"]
        for j in obs:
            start_h = base_h_sim[0, j]
            gap = start_h - target_h
            # 緩衝期內動態調整目標，避免初始解不可行
            dynamic_target = start_h - (gap * (t + 1) / opt_config["BUFFER_HOURS"]) if (gap > 0 and t < opt_config["BUFFER_HOURS"]) else target_h
            
            # 核心約束：背景水位 - 各井抽水造成的降深 <= 目標水位
            prob += (base_h_sim[t, j] - lpSum([x[t][i] * well_powers[i] * inf_matrix[i, j] for i in wells])) <= (dynamic_target + slack[t][j])
            
    prob.solve(PULP_CBC_CMD(msg=0, timeLimit=120))
    
    if LpStatus[prob.status] != 'Optimal':
        print("⚠️ 警告: 未找到最佳解，狀態為:", LpStatus[prob.status])

    schedule = np.array([[value(x[t][i]) for i in wells] for t in times])
    total_kwh = np.sum(schedule * well_powers * opt_config["delta_t"])
    avg_q_out = np.mean(np.sum(schedule * well_powers * kW_to_m3h, axis=1))
    active_wells_per_step = np.sum(schedule, axis=1)
    
    return {
        "schedule": schedule,
        "inf_matrix": inf_matrix,
        "total_kwh": total_kwh,
        "total_cost": total_kwh * opt_config["FIXED_PRICE_PER_KWH"],
        "avg_q_out": avg_q_out,
        "avg_active_wells": np.mean(active_wells_per_step),
        "min_active_wells_required": opt_config["MIN_ACTIVE_WELLS"],
        "actual_min_active_wells": np.min(active_wells_per_step),
        "optimization_goal": "Minimize energy while maintaining safety level"
    }

# ==========================================
# 3. 主流程
# ==========================================
if __name__ == "__main__":
    pinn_path = opt_config["pinn_report_path"]
    
    # 讀取資料
    df_raw = pd.read_csv("Master_Training_Data_Continuous3.csv", index_col=0)
    df_raw.index = pd.to_datetime(df_raw.index)
    dist_df = pd.read_csv("Distance_Matrix.csv", index_col=0)
    
    # ------------------------------------------
    # 🌟 整合第二個程式的 Qin 計算邏輯 🌟
    # ------------------------------------------
    qw_cols = [c for c in df_raw.columns if c.startswith('Qw')]
    obs_wells = opt_config["OBS_LIST"]
    
    # 計算每一時刻的 Q_out 與 Avg_H
    df_raw['Q_out_rate_calc'] = df_raw[qw_cols].sum(axis=1) 
    df_raw['Avg_H_calc'] = df_raw[obs_wells].mean(axis=1)
    df_raw['Delta_H_calc'] = df_raw['Avg_H_calc'].diff()
    
    # 計算動態 Qin: (ΔH * Area * Sy / Δt) + Qout
    df_raw['Q_in_rate_calc'] = (df_raw['Delta_H_calc'] * opt_config["AREA"] * opt_config["SY"] / opt_config["delta_t"]) + df_raw['Q_out_rate_calc']
    
    # 取中位數作為本次最佳化的 Qin 參考值 (避免雜訊)
    current_qin = df_raw['Q_in_rate_calc'].median()
    # ------------------------------------------

    # 讀取其餘物理參數
    try:
        learned_T = np.load(os.path.join(pinn_path, "learned_T.npy"))
        learned_C = np.load(os.path.join(pinn_path, "learned_C.npy"))
        bg_h_preds = np.load(os.path.join(pinn_path, "background_h_7d.npy"))
        pinn_params = {'T': learned_T, 'C': learned_C}
    except Exception as e:
        print(f"❌ PINN 檔案讀取失敗: {e}"); sys.exit()

    # 目標水位計算
    target_h = opt_config["FIXED_TARGET_H"]

    # --- 📈 合理化背景水位修正 ---
    print("\n--- 📈 背景水位趨勢檢查 ---")
    
    # 計算理論上升速度 (m/hr)
    rise_per_hour = (current_qin / (opt_config["AREA"] * opt_config["SY"]))
    
    # 💡 關鍵：設定一個物理上限 (例如靜態水位 -11.0m)
    # 這樣水位就不會一路漲到 50m，而是漲到平衡點就停住
    LIMIT_H = opt_config["STATIC_WATER_LEVEL"] 
    
    print(f"理論上升速度: {rise_per_hour:.4f} m/hr")
    
    # 進行修正：讓平緩的預測加上 $Q_{in}$ 趨勢，但受限於上限
    for t in range(1, opt_config["SIM_HOURS"]):
        time_passed = t * opt_config["delta_t"]
        # 計算此時刻應該漲到的高度
        theoretical_h = bg_h_preds[t-1] + (rise_per_hour * opt_config["delta_t"])
        # 取理論值與上限的最小值
        bg_h_preds[t] = np.minimum(theoretical_h, LIMIT_H)

    avg_corrected = np.mean(bg_h_preds, axis=1)
    print(f"修正後第 24 小時平均水位: {avg_corrected[48]:.2f} m")
    print(f"修正後第 168 小時平均水位: {avg_corrected[-1]:.2f} m (已受限於 {LIMIT_H}m)")
    print("---------------------------\n")



    print("\n" + "="*50)
    print(f"🔍 物理參數診斷表")
    print(f"   - 導水係數 T   : {learned_T:.4f}")
    print(f"   - 計算進流量 Qin: {current_qin:.2f} m3/h (由歷史水位變化推算)")
    print(f"   - 比給水度 Sy  : {opt_config['SY']:.3f}")
    print(f"🎯 目標水位設定: {target_h:.2f} m")
    print("="*50 + "\n")

    # 計算各井平均功率 (從歷史增量中提取)
    well_list = opt_config["WELL_LIST"]
    well_specific_powers = []
    for w in well_list:
        col = f"KWH-{int(''.join(filter(str.isdigit, w))):02d}"
        if col in df_raw.columns:
            diffs = df_raw[col].diff().dropna()
            # 過濾負值與異常值，取平均功率
            valid_diffs = diffs[diffs > 0]
            well_specific_powers.append(valid_diffs.mean() / opt_config["delta_t"] if not valid_diffs.empty else 7.5)
        else: 
            well_specific_powers.append(7.5)
    well_specific_powers = np.array(well_specific_powers)


    
    # 執行最佳化
    res = run_pulp_optimization(
        pinn_params, 
        well_specific_powers, 
        bg_h_preds, 
        dist_df.loc[opt_config["OBS_LIST"], well_list].values, 
        target_h
    )

    # 顯示結果報告
    print("\n" + "█"*60)
    print(f"  📋 智慧降水優化報告 - 基於即時 Qin 分析")
    print(f"  🔹 參考 Qin 入流量 : {current_qin:,.2f} m3/hr")
    print(f"  🔹 目標安全水位   : {target_h:.2f} m")
    print(f"  ⚡ 優化後平均抽水量 : {res['avg_q_out']:,.1f} m3/hr")
    print(f"  ⚡ 預計 7 天耗電量 : {res['total_kwh']:,.0f} kWh")
    print(f"  🛠  平均啟用井數   : {res['avg_active_wells']:.1f} 口")
    print(f"  💰 預估電費 (TWD) : $ {res['total_cost']:,.0f}")
    print("█"*60 + "\n")

    # 繪製排程與水位預測圖
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 排程熱圖
    sns.heatmap(res['schedule'].T, cmap="YlGn", cbar=False, yticklabels=well_list, ax=ax1)
    ax1.set_title("Optimal Pumping Schedule (7 Days)")
    ax1.set_xlabel("Time Step (0.5 hr)")

    # 水位預測曲線
    pred_h = np.zeros_like(bg_h_preds)
    for t in range(opt_config["SIM_HOURS"]):
        # 降深 = Σ (開關 * 功率 * 影響係數)
        drawdown = np.sum((res['schedule'][t] * well_specific_powers)[:, np.newaxis] * res['inf_matrix'], axis=0)
        pred_h[t] = bg_h_preds[t] - drawdown
    
    ax2.plot(pred_h, alpha=0.3)
    ax2.plot(np.max(pred_h, axis=1), color='red', linewidth=2, label="Predicted Max Water Level")
    ax2.axhline(y=target_h, color='black', linestyle='--', label=f"Target ({target_h:.2f}m)")
    
    ax2.set_title("Predicted Water Level Profiles")
    ax2.set_ylabel("Elevation (m)")
    ax2.legend()
    
    plt.tight_layout()
    plt.show()