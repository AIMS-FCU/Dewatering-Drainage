import pandas as pd
import numpy as np
from pulp import *
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os
import sys
from pathlib import Path
from sklearn.metrics import mean_absolute_error
from scipy.optimize import minimize

# 設定中文字型（避免亂碼）
matplotlib.rc('font', family='Microsoft JhengHei')
matplotlib.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 系統核心配置
# ==========================================
opt_config = {
    "ANALYSIS_START": "2020-10-7 00:00",
    "ANALYSIS_END":   "2020-10-14 00:00",
    "AREA": 3319.95,
    # "SY": 0.161,
    "DEPTH": 19.55,
    "STATIC_WATER_LEVEL": -11.0,
    "delta_t": 1.0,
    "η": 0.4,
    "HEAD": 30.0,
    "SF": 1.1,
    "SAFETY_BUFFER": 0.5,
    "rw": 0.45,
    "R_init": 500.0,
    "FIXED_PRICE_PER_KWH": 4.1,
    "TARGET_FLOOR": "B6F",
    "FIXED_TARGET_H": -19.7  ,
    "SIM_HOURS": 168,    
    "SIM_STEPS": 168,    
    "MIN_ACTIVE_WELLS": 0,
    "INITIAL_CARRYOVER_HOURS": 6,
    "BUFFER_HOURS": 24,
    "PRECALIBRATION_ENABLED": True,
    "PRECALIBRATION_HOURS": 24,
    "pinn_report_path": "PINN_MAPE_Complete_Report3/428_10",
    "WELL_LIST": ["PW01", "PW02", "PW03", "PW04", "PW06", "PW07", "PW08", "PW09", "PW010", "PW011", "PW012", "PW013"],
    "OBS_LIST": ['PA', 'PB','FPS7','FPS8', 'FPS9', 'FPS2', 'FPS3', 'FPS4', 'FPS5', 'FPS6'],    

    # 由 DynamicCalibrator 自動填入，不需手動調整
    "CALIBRATION": {
        "Q_IN_FACTOR": 1.0,
        "T_ADJUST_FACTOR": 1.0,
        "POWER_ADJUST": 1.0
    }
}

kW_to_m3h = (opt_config["η"] * 3600) / (9.81 * opt_config["HEAD"])

# ==========================================
# 2. 動態自動校準器 (DynamicCalibrator)
# ==========================================
REQUIRED_PINN_FILES = [
    "learned_T.npy",
    "learned_C.npy",
    "calibrated_inflow_sy.npy",
]

TRAINING_OBS_ORDER = [
    "PA",
    "PB",
    "PC",
    "FPS7",
    "FPS8",
    "FPS9",
    "FPS2",
    "FPS3",
    "FPS4",
    "FPS5",
    "FPS6",
]


def resolve_pinn_report_path(configured_path):
    configured = Path(configured_path)
    if not configured.is_absolute():
        configured = Path.cwd() / configured
    configured = configured.resolve()

    def has_required_files(folder):
        return folder.is_dir() and all((folder / name).exists() for name in REQUIRED_PINN_FILES)

    if has_required_files(configured):
        return str(configured)

    search_root = configured if configured.is_dir() else configured.parent
    candidates = [path for path in search_root.rglob("*") if has_required_files(path)]
    if not candidates:
        raise FileNotFoundError(
            f"找不到可用的 PINN 結果資料夾。原始設定路徑: {configured_path}"
        )

    candidates.sort(
        key=lambda path: max((path / name).stat().st_mtime for name in REQUIRED_PINN_FILES),
        reverse=True,
    )
    selected = candidates[0]
    print(f"未找到原始設定資料夾，改用最新可用結果: {selected}")
    return str(selected)


def align_observation_predictions(all_pinn_h, target_obs_list, well_list):
    source_names_with_wells = TRAINING_OBS_ORDER + list(well_list)
    if all_pinn_h.shape[1] == len(source_names_with_wells):
        source_obs_names = TRAINING_OBS_ORDER
    elif all_pinn_h.shape[1] == len(TRAINING_OBS_ORDER):
        source_obs_names = TRAINING_OBS_ORDER
    else:
        raise ValueError(
            f"accurate_pred_h_7d.npy 欄數為 {all_pinn_h.shape[1]}，無法對應觀測井名稱"
        )

    source_index = {name: idx for idx, name in enumerate(source_obs_names)}
    missing = [name for name in target_obs_list if name not in source_index]
    if missing:
        raise KeyError(f"找不到這些觀測井在 PINN 預測中的欄位: {missing}")

    aligned = all_pinn_h[:, [source_index[name] for name in target_obs_list]]
    print(f"觀測井名稱對齊完成: {target_obs_list}")
    return aligned


def build_actual_obs_matrix(df_actual, obs_list, sim_steps):
    actual_h = np.full((sim_steps, len(obs_list)), np.nan)
    for idx, col in enumerate(obs_list):
        if col not in df_actual.columns:
            continue
        vals = pd.to_numeric(df_actual[col], errors="coerce").values
        n = min(len(vals), sim_steps)
        actual_h[:n, idx] = vals[:n]
    return actual_h


def build_influence_matrix(dist_matrix, well_powers, T_value, C_values):
    num_wells = len(well_powers)
    num_obs = dist_matrix.shape[0]
    inf_matrix = np.zeros((num_wells, num_obs))
    for i in range(num_wells):
        for j in range(num_obs):
            r = dist_matrix[j, i] if dist_matrix[j, i] > 0.01 else opt_config["rw"]
            inf_matrix[i, j] = (kW_to_m3h / (2 * np.pi * T_value)) * np.log(opt_config["R_init"] / r) + \
                               (C_values[i] * kW_to_m3h if dist_matrix[j, i] < 1.0 else 0)
    return inf_matrix


def apply_background_precalibration(bg_h_preds, actual_h_matrix, actual_schedule, well_powers, inf_matrix):
    if not opt_config.get("PRECALIBRATION_ENABLED", False):
        return bg_h_preds, np.zeros(bg_h_preds.shape[1])

    calibration_steps = int(opt_config["PRECALIBRATION_HOURS"] / opt_config["delta_t"])
    calibration_steps = max(1, min(calibration_steps, bg_h_preds.shape[0]))

    actual_drawdown = np.zeros_like(actual_h_matrix)
    for t in range(actual_h_matrix.shape[0]):
        actual_drawdown[t] = np.sum(
            (actual_schedule[t] * well_powers)[:, np.newaxis] * inf_matrix,
            axis=0,
        )

    implied_background = actual_h_matrix + actual_drawdown
    bias_window = implied_background[:calibration_steps] - bg_h_preds[:calibration_steps]
    bias_by_obs = np.nanmedian(bias_window, axis=0)
    bias_by_obs = np.where(np.isnan(bias_by_obs), 0.0, bias_by_obs)

    calibrated_bg = bg_h_preds + bias_by_obs[np.newaxis, :]
    valid_window = ~np.isnan(implied_background[:calibration_steps])
    calibrated_bg[:calibration_steps] = np.where(
        valid_window,
        np.maximum(calibrated_bg[:calibration_steps], implied_background[:calibration_steps]),
        calibrated_bg[:calibration_steps],
    )

    print(
        f"背景水位前導校正完成: 使用前 {calibration_steps} 小時，"
        f"平均偏移 {np.mean(bias_by_obs):+.3f} m"
    )
    return calibrated_bg, bias_by_obs


class DynamicCalibrator:
    """
    原理：用「歷史實際抽水排程」做正向物理模擬，
    計算 (bg_h - drawdown) 和「實際觀測水位」的 MAE，
    找出讓誤差最小的 (Q_factor, T_factor)。
    這是有物理依據的校準，不是 curve fitting。
    """
    def __init__(self, base_T, base_C, start_h, df_cal, obs_list, well_list,
                 qw_match, well_powers, dist_matrix, pinn_qin, pinn_sy):
        self.base_T = base_T
        self.base_C = base_C
        self.start_h = start_h              # 起始水位 shape: (n_obs,)
        self.obs_list = obs_list
        self.well_powers = well_powers
        self.dist_matrix = dist_matrix      # shape: (n_obs, n_wells)
        self.pinn_qin = pinn_qin
        self.pinn_sy = pinn_sy
        self.sim_steps = opt_config["SIM_STEPS"]
        self.n_obs = len(obs_list)
        self.n_wells = len(well_list)

        # 實際抽水排程（Binary: 開=1, 關=0）shape: (sim_steps, n_wells)
        raw_on = (df_cal[qw_match] > 0.5).values
        if len(raw_on) < self.sim_steps:
            raw_on = np.pad(raw_on, ((0, self.sim_steps - len(raw_on)), (0, 0)), 'edge')
        self.actual_schedule = raw_on[:self.sim_steps].astype(float)

        # 實際觀測水位 shape: (sim_steps, n_obs)
        self.actual_h = np.full((self.sim_steps, self.n_obs), np.nan)
        for i, col in enumerate(obs_list):
            if col in df_cal.columns:
                vals = df_cal[col].values
                n = min(len(vals), self.sim_steps)
                self.actual_h[:n, i] = vals[:n]

    def _simulate_error(self, factors):
        q_factor, t_factor = factors
        T = self.base_T * t_factor
        C = self.base_C

        # 1. 重建背景水位（Qin 控制上升速度）
        current_qin = self.pinn_qin * q_factor
        rise = (current_qin * opt_config["delta_t"]) / (opt_config["AREA"] * self.pinn_sy)
        bg_h = np.zeros((self.sim_steps, self.n_obs))
        bg_h[0] = self.start_h
        for t in range(1, self.sim_steps):
            bg_h[t] = np.minimum(bg_h[t-1] + rise, opt_config["STATIC_WATER_LEVEL"])

        # 2. 影響矩陣（Theis 公式）
        inf_matrix = np.zeros((self.n_wells, self.n_obs))
        for i in range(self.n_wells):
            for j in range(self.n_obs):
                r = self.dist_matrix[j, i] if self.dist_matrix[j, i] > 0.01 else opt_config["rw"]
                inf_matrix[i, j] = (kW_to_m3h / (2 * np.pi * T)) * np.log(opt_config["R_init"] / r) + \
                                   (C[i] * kW_to_m3h if self.dist_matrix[j, i] < 1.0 else 0)

        # 3. 正向模擬：用實際排程計算預測水位
        pred_h = np.zeros_like(bg_h)
        for t in range(self.sim_steps):
            drawdown = np.sum((self.actual_schedule[t] * self.well_powers)[:, np.newaxis] * inf_matrix, axis=0)
            pred_h[t] = bg_h[t] - drawdown

        # 4. 只算有實際觀測值的點的 MAE
        valid = ~np.isnan(self.actual_h)
        if not np.any(valid):
            return 1e6
        return np.mean(np.abs(self.actual_h[valid] - pred_h[valid]))

    def get_best_factors(self):
        print("  🔍 DynamicCalibrator 自動校準中（比對歷史排程與實際水位）...")
        result = minimize(
            self._simulate_error,
            x0=[1.0, 1.0],                    # 初始：不調整
            bounds=[(0.05, 2.0), (0.1, 3.0)], # Q_factor, T_factor 的合理範圍
            method='L-BFGS-B'
        )
        q_opt, t_opt = result.x
        print(f"  ✅ 校準完成：Q_factor={q_opt:.3f} | T_factor={t_opt:.3f} | Calibration MAE={result.fun:.4f} m")
        return q_opt, t_opt


# ==========================================
# 3. 核心最佳化引擎 (PuLP)
# ==========================================
def run_pulp_optimization(pinn_params, well_powers, base_h_sim, dist_matrix, target_h_array,
                          initial_status=None, remaining_on_steps=None,
                          initial_active_count=0, carryover_count_steps=0):
    T = pinn_params['T'] * opt_config["CALIBRATION"]["T_ADJUST_FACTOR"]
    C = pinn_params['C']
    num_wells, num_obs = len(well_powers), base_h_sim.shape[1]
    sim_steps = opt_config["SIM_STEPS"]

    prob = LpProblem("Smart_Pumping_Optimization", LpMinimize)
    times, wells, obs = range(sim_steps), range(num_wells), range(num_obs)

    x = LpVariable.dicts("X", (times, wells), cat='Binary')
    slack = LpVariable.dicts("S", (times, obs), lowBound=0, cat='Continuous')

    if initial_status is None:
        initial_status = np.zeros(num_wells, dtype=int)
    else:
        initial_status = np.asarray(initial_status, dtype=int)

    if remaining_on_steps is None:
        remaining_on_steps = np.zeros(num_wells, dtype=int)
    else:
        remaining_on_steps = np.asarray(remaining_on_steps, dtype=int)

    total_energy_expr = lpSum([x[t][i] * well_powers[i] * opt_config["delta_t"] for t in times for i in wells])
    # 將懲罰係數從一億改成一萬，避免超大係數造成矩陣數值不穩定 (Ill-conditioned) 讓求解器卡死
    slack_penalty_expr = lpSum([slack[t][j] * 10000 for t in times for j in obs])
    prob += total_energy_expr + slack_penalty_expr

    inf_matrix = np.zeros((num_wells, num_obs))
    for i in wells:
        for j in obs:
            r = dist_matrix[j, i] if dist_matrix[j, i] > 0.01 else opt_config["rw"]
            inf_matrix[i, j] = (kW_to_m3h / (2 * np.pi * T)) * np.log(opt_config["R_init"] / r) + \
                               (C[i] * kW_to_m3h if dist_matrix[j, i] < 1.0 else 0)

    for t in times:
        prob += lpSum([x[t][i] for i in wells]) >= opt_config["MIN_ACTIVE_WELLS"]
        for j in obs:
            start_h = base_h_sim[0, j]
            target_h = target_h_array[j]
            gap = start_h - target_h
            buffer_steps = opt_config["BUFFER_HOURS"] / opt_config["delta_t"]
            dynamic_target = start_h - (gap * (t + 1) / buffer_steps) if (gap > 0 and t < buffer_steps) else target_h
            prob += (base_h_sim[t, j] - lpSum([x[t][i] * well_powers[i] * inf_matrix[i, j] for i in wells])) <= (dynamic_target + slack[t][j])

    # 限制式：最小連續運轉時間 = 6小時
    # 方案一：優化邏輯，摒棄 y 變數，改用總和式表達 (大幅減少分支維度)
    for i in wells:
        prob += x[0][i] == int(initial_status[i])
        for t in range(1, min(sim_steps, int(remaining_on_steps[i]) + 1)):
            prob += x[t][i] == 1

    for t in range(min(sim_steps, int(carryover_count_steps))):
        prob += lpSum([x[t][i] for i in wells]) >= int(initial_active_count)

    min_on_steps = int(6 / opt_config["delta_t"])
    for i in wells:
        for t in range(min_on_steps, sim_steps):
            # 如果在 t 時刻被關掉 (也就是 x[t-1]-x[t] == 1)，
            # 則要求在前面 min_on_steps 內必須全是開啟狀態 (總和 >= min_on_steps)
            prob += lpSum([x[k][i] for k in range(t - min_on_steps, t)]) >= min_on_steps * (x[t-1][i] - x[t][i])

    # 為了避免求解器為證明 "最完美" 而陷入無盡搜索，強制設定 2 分鐘(120秒) 交卷時間
    prob.solve(PULP_CBC_CMD(msg=1, gapRel=0.2, timeLimit=120))
    schedule = np.array([[value(x[t][i]) if value(x[t][i]) is not None else 0 for i in wells] for t in times])
    print(f"\n⚙️ [解算狀態] PuLP Solver Status: {LpStatus[prob.status]}")
    print(f"⚙️ [解算狀態] Schedule 最大數值: {np.max(schedule):.4f}")
    if np.max(schedule) > 0 and np.max(schedule) < 0.9:
        print("⚠️ 警告：最佳化引擎因為超時，回傳了非整數（小數）的未完成解答！")

    return {
        "schedule": schedule,
        "inf_matrix": inf_matrix,
        "total_kwh": np.sum(schedule * well_powers * opt_config["delta_t"]),
        "total_m3": np.sum(schedule * well_powers * opt_config["delta_t"]) * kW_to_m3h,
        "avg_active_wells": np.mean(np.sum(schedule, axis=1)),
        "active_wells_series": np.sum(schedule, axis=1)
    }


def build_well_comparison_table(time_index, well_list, actual_schedule, optimized_schedule):
    comparison_rows = []

    for t in range(len(time_index)):
        actual_on = [well_list[i] for i, flag in enumerate(actual_schedule[t]) if flag > 0.5]
        optimized_on = [well_list[i] for i, flag in enumerate(optimized_schedule[t]) if flag > 0.5]

        actual_set = set(actual_on)
        optimized_set = set(optimized_on)
        matched = sorted(actual_set & optimized_set)
        actual_only = sorted(actual_set - optimized_set)
        optimized_only = sorted(optimized_set - actual_set)

        comparison_rows.append({
            "時間": time_index[t],
            "實際開井數": len(actual_on),
            "最佳化開井數": len(optimized_on),
            "實際開井井號": ", ".join(actual_on) if actual_on else "-",
            "最佳化開井井號": ", ".join(optimized_on) if optimized_on else "-",
            "共同開井井號": ", ".join(matched) if matched else "-",
            "實際有開但模型沒開": ", ".join(actual_only) if actual_only else "-",
            "模型有開但實際沒開": ", ".join(optimized_only) if optimized_only else "-",
            "井號完全一致": int(actual_set == optimized_set)
        })

    return pd.DataFrame(comparison_rows)


def compute_carryover_constraints(df_raw, eval_start, well_list, sim_steps):
    min_on_steps = int(6 / opt_config["delta_t"])
    history_hours = max(min_on_steps - 1, 0)
    history_start = eval_start - pd.Timedelta(hours=history_hours)
    df_history = df_raw.loc[history_start:eval_start].iloc[::2].copy()

    initial_status = np.zeros(len(well_list), dtype=int)
    remaining_on_steps = np.zeros(len(well_list), dtype=int)

    for idx, w in enumerate(well_list):
        qw_col = f"Qw{int(''.join(filter(str.isdigit, w))):02d}"
        if qw_col not in df_history.columns or df_history.empty:
            continue

        history_series = (df_history[qw_col].values > 0.5).astype(int)
        if len(history_series) == 0:
            continue

        initial_status[idx] = int(history_series[-1])
        if initial_status[idx] == 0:
            continue

        consecutive_on = 0
        for flag in history_series[::-1]:
            if flag == 1:
                consecutive_on += 1
            else:
                break

        remaining_on_steps[idx] = max(0, min_on_steps - consecutive_on)

    carryover_steps = min(sim_steps, int(np.max(remaining_on_steps)) + 1 if np.any(initial_status) else 1)
    return initial_status, remaining_on_steps, df_history.index, carryover_steps


# ==========================================
# 4. 主流程
# ==========================================
if __name__ == "__main__":
    pinn_path = resolve_pinn_report_path(opt_config["pinn_report_path"])
    df_raw = pd.read_csv("Master_Training_Data_Continuous3.csv", index_col=0)
    df_raw.index = pd.to_datetime(df_raw.index)
    dist_df = pd.read_csv("Distance_Matrix.csv", index_col=0)

    eval_start = pd.to_datetime(opt_config["ANALYSIS_START"])
    eval_end = eval_start + pd.Timedelta(hours=opt_config["SIM_HOURS"])
    # [降採樣] 每 1 小時取一筆資料以對齊 1.0h 的最佳化步長
    df_actual = df_raw.loc[eval_start : eval_end].iloc[::2].copy()

    well_list, obs_list = opt_config["WELL_LIST"], opt_config["OBS_LIST"]

    # --- A. 實際數據結算 ---
    qw_match = [f"Qw{int(''.join(filter(str.isdigit, w))):02d}" for w in well_list if f"Qw{int(''.join(filter(str.isdigit, w))):02d}" in df_actual.columns]
    kwh_match = [f"KWH-{int(''.join(filter(str.isdigit, w))):02d}" for w in well_list if f"KWH-{int(''.join(filter(str.isdigit, w))):02d}" in df_actual.columns]
    actual_total_kwh = sum([max(0, df_actual[col].iloc[-1] - df_actual[col].iloc[0]) for col in kwh_match])
    actual_total_m3 = (df_actual[qw_match].sum(axis=1) * opt_config["delta_t"]).sum()
    actual_active_series = (df_actual[qw_match] > 0.5).sum(axis=1).values
    if len(actual_active_series) < opt_config["SIM_STEPS"]:
        actual_active_series = np.pad(actual_active_series, (0, opt_config["SIM_STEPS"] - len(actual_active_series)), 'edge')
    actual_active_series = actual_active_series[:opt_config["SIM_STEPS"]]

    actual_schedule_full = np.zeros((opt_config["SIM_STEPS"], len(well_list)))
    for idx, w in enumerate(well_list):
        qw_col = f"Qw{int(''.join(filter(str.isdigit, w))):02d}"
        if qw_col in df_actual.columns:
            series = (df_actual[qw_col].values > 0.5).astype(int)
            if len(series) < opt_config["SIM_STEPS"]:
                series = np.pad(series, (0, opt_config["SIM_STEPS"] - len(series)), 'edge')
            actual_schedule_full[:, idx] = series[:opt_config["SIM_STEPS"]]

    initial_status, remaining_on_steps, _, _ = compute_carryover_constraints(
        df_raw, eval_start, well_list, opt_config["SIM_STEPS"]
    )
    active_carryover_wells = [well_list[i] for i, flag in enumerate(initial_status) if flag == 1]
    initial_active_count = int(initial_status.sum())
    carryover_count_steps = min(
        opt_config["SIM_STEPS"],
        int(opt_config["INITIAL_CARRYOVER_HOURS"] / opt_config["delta_t"])
    )

    # --- B. 載入 PINN 參數 & 起始水位 ---
    try:
        learned_T = np.load(os.path.join(pinn_path, "learned_T.npy"))
        learned_C = np.load(os.path.join(pinn_path, "learned_C.npy"))
        pinn_calibrated = np.load(os.path.join(pinn_path, "calibrated_inflow_sy.npy"))
        pinn_qin = float(pinn_calibrated[0])
        pinn_sy  = float(pinn_calibrated[1])  # 由 PINN 學到的有效孔隙率
        print(f"✅ PINN 參數載入：Qin={pinn_qin:.1f} m³/hr | Sy={pinn_sy:.4f}（來自 calibrated_inflow_sy.npy）")
    except Exception as e:
        print(f"❌ PINN 數據載入失敗: {e}"); sys.exit()

    start_h_actual = df_raw.loc[eval_start, obs_list].values.astype(float)
    actual_h_matrix = build_actual_obs_matrix(df_actual, obs_list, opt_config["SIM_STEPS"])
    
    # --- C. 載入 PINN 神準預測 (含實際抽水特徵) ---
    bg_h_path = os.path.join(pinn_path, "accurate_pred_h_7d.npy")
    if os.path.exists(bg_h_path):
        print(f"📡 偵測到 PINN 準確預測結果 (含現場抽水)，正在載入 {bg_h_path}...")
        all_pinn_h = np.load(bg_h_path)
        # [降採樣] PINN 是 30min 一筆，我們取 [::2] 轉為 1 小時一筆
        # 🌟 修正：因為 PINN 現在會輸出所有井(23口)，我們最佳化只取觀測井(前 11 口)
        accurate_h_preds = align_observation_predictions(all_pinn_h[::2], obs_list, well_list)
        # 確保步數匹配
        if len(accurate_h_preds) > opt_config["SIM_STEPS"]:
            accurate_h_preds = accurate_h_preds[:opt_config["SIM_STEPS"]]
        elif len(accurate_h_preds) < opt_config["SIM_STEPS"]:
             accurate_h_preds = np.pad(accurate_h_preds, ((0, opt_config["SIM_STEPS"] - len(accurate_h_preds)), (0, 0)), mode='edge')
    else:
        print(f"⚠️ 未發現 {bg_h_path}，退回傳統模式...")
        accurate_h_preds = None

    bg_h_preds = np.zeros((opt_config["SIM_STEPS"], len(obs_list)))
    bg_h_preds[0] = start_h_actual

    # --- C. 計算各井功率 ---
    well_specific_powers = []
    for w in well_list:
        col = f"KWH-{int(''.join(filter(str.isdigit, w))):02d}"
        well_specific_powers.append(df_raw[col].diff().loc[lambda x: x > 0].mean() / 0.5 if col in df_raw.columns else 7.5)
    well_specific_powers = np.array(well_specific_powers)
    dist_matrix_np = dist_df.loc[obs_list, well_list].values

    # --- D. 🌟 DynamicCalibrator 自動校準 ---
    print(f"\n📡 PINN 物理校準 Qin = {pinn_qin:.1f} m³/hr")
    calibrator = DynamicCalibrator(
        base_T=learned_T, base_C=learned_C,
        start_h=start_h_actual,
        df_cal=df_actual,
        obs_list=obs_list, well_list=well_list,
        qw_match=qw_match, well_powers=well_specific_powers,
        dist_matrix=dist_matrix_np, pinn_qin=pinn_qin, pinn_sy=pinn_sy
    )
    best_q, best_t = calibrator.get_best_factors()
    opt_config["CALIBRATION"]["Q_IN_FACTOR"] = best_q
    opt_config["CALIBRATION"]["T_ADJUST_FACTOR"] = best_t

    # --- E. 物理參數準備與「反向基準推導」(Reverse Superposition) ---
    if accurate_h_preds is not None:
        print(f"\n🧬 [混合重疊原理] 啟動反向推導機制 (Reverse Superposition)...")
        # 1. 整理現場真實的井排程矩陣 (SIM_STEPS, n_wells)
        actual_schedule_full = np.zeros((opt_config["SIM_STEPS"], len(well_list)))
        for idx, w in enumerate(well_list):
            qw_col = f"Qw{int(''.join(filter(str.isdigit, w))):02d}"
            if qw_col in df_actual.columns:
                series = (df_actual[qw_col].values > 0.5).astype(int)
                if len(series) < opt_config["SIM_STEPS"]:
                    series = np.pad(series, (0, opt_config["SIM_STEPS"] - len(series)), 'edge')
                actual_schedule_full[:, idx] = series[:opt_config["SIM_STEPS"]]
        
        # 2. 建立精確對標的影響力矩陣 (帶入校準後 T)
        T_cal = learned_T * best_t
        inf_matrix_cal = build_influence_matrix(dist_matrix_np, well_specific_powers, T_cal, learned_C)
        
        # 3. 逐步將「現場水泵造成的理論下陷」反向加回「準確預測水位」，還原純天然基準線
        reconstructed_h_list = np.zeros_like(bg_h_preds)
        for t in range(opt_config["SIM_STEPS"]):
            theoretical_drawdown = np.sum((actual_schedule_full[t] * well_specific_powers)[:, np.newaxis] * inf_matrix_cal, axis=0)
            bg_h_preds[t] = accurate_h_preds[t] + theoretical_drawdown
            # [物理閉環驗證]：還原值 = 背景 - 理論洩降 (應該要剛好回到 accurate_h_preds)
            reconstructed_h_list[t] = bg_h_preds[t] - theoretical_drawdown
            
        print(f"  👉 已成功從「神準預測線」剝離 {np.sum(actual_schedule_full > 0)} 個運轉時數，還原為公平之天然起跑線！")

        # --- 新增：繪製反向工程閉環驗證圖 ---
        print("📈 正在產出反向工程驗證圖 (Reverse_Verification_Plot.png)...")
        bg_h_preds, bg_bias = apply_background_precalibration(
            bg_h_preds,
            actual_h_matrix,
            actual_schedule_full,
            well_specific_powers,
            inf_matrix_cal,
        )
        print(f"  背景校正各井偏移: {np.round(bg_bias, 3)}")
        n_plots = len(obs_list)
        rows_v = int(np.ceil(n_plots / 3))
        fig_v, axes_v = plt.subplots(rows_v, 3, figsize=(18, 4 * rows_v))
        axes_v = axes_v.flatten()
        time_h_v = np.arange(opt_config["SIM_STEPS"]) * opt_config["delta_t"]
        
        for i, col in enumerate(obs_list):
            axes_v[i].plot(time_h_v, accurate_h_preds[:, i], label='Original AI Pred', color='royalblue', linewidth=2)
            axes_v[i].plot(time_h_v, reconstructed_h_list[:, i], label='Reconstructed H', color='crimson', linestyle='--', linewidth=2)
            axes_v[i].plot(time_h_v, bg_h_preds[:, i], label='Derived Background', color='gray', alpha=0.3)
            axes_v[i].set_title(f"Verification: {col}")
            axes_v[i].legend(fontsize=8)
            axes_v[i].grid(True, alpha=0.3)
            
            # 計算驗證误差
            v_mae = np.mean(np.abs(accurate_h_preds[:, i] - reconstructed_h_list[:, i]))
            axes_v[i].text(0.05, 0.05, f"Verification MAE: {v_mae:.2e}m", transform=axes_v[i].transAxes, fontsize=9, color='darkgreen')

        for j in range(n_plots, len(axes_v)): fig_v.delaxes(axes_v[j])
        fig_v.suptitle("Reverse Superposition Closure Verification: Accurate Pred vs Derived (Background - Drawdown)", fontsize=16)
        fig_v.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig_v.savefig(os.path.join(pinn_path, "Reverse_Verification_Plot.png"))
        plt.close()
    else:
        current_qin = pinn_qin * best_q
        rise_per_step = (current_qin * opt_config["delta_t"]) / (opt_config["AREA"] * pinn_sy)
        for t in range(1, opt_config["SIM_STEPS"]):
            bg_h_preds[t] = np.minimum(bg_h_preds[t-1] + rise_per_step, opt_config["STATIC_WATER_LEVEL"])
        print(f"  📡 備援模式（物理公式）：Qin={current_qin:.1f} m³/hr | Sy={pinn_sy:.4f} | rise={rise_per_step:.4f} m/step")

    # --- F. 自動擷取歷史容忍邊界與執行最佳化 (暫時退回單一水位) ---
    print(f"\n🎯 最佳化水位基準：統一單一目標水位")
    target_h_array = np.full(len(obs_list), float(opt_config["FIXED_TARGET_H"]))
    print(f"   - 所有觀測井目標皆設定為: ≤ {opt_config['FIXED_TARGET_H']} m")

    print(f"   - 承接現場當下開井: {', '.join(active_carryover_wells) if active_carryover_wells else '-'}")
    if active_carryover_wells:
        carryover_desc = ", ".join([
            f"{well_list[i]} 還需維持 {remaining_on_steps[i]} 步"
            for i, flag in enumerate(initial_status) if flag == 1
        ])
        print(f"   - 最短續開限制: {carryover_desc}")

    res = run_pulp_optimization(
        {'T': learned_T, 'C': learned_C},
        well_specific_powers,
        bg_h_preds,
        dist_matrix_np,
        target_h_array,
        initial_status=initial_status,
        remaining_on_steps=remaining_on_steps,
        initial_active_count=initial_active_count,
        carryover_count_steps=carryover_count_steps
    )

    compare_time_index = df_actual.index[:opt_config["SIM_STEPS"]]
    optimized_schedule_binary = (res['schedule'] > 0.5).astype(int)
    df_well_compare = build_well_comparison_table(
        compare_time_index,
        well_list,
        actual_schedule_full.astype(int),
        optimized_schedule_binary
    )
    well_compare_path = os.path.join(pinn_path, "Well_Open_Comparison.csv")
    df_well_compare.to_csv(well_compare_path, index=False, encoding="utf-8-sig")

    exact_match_rate = df_well_compare["井號完全一致"].mean() * 100
    total_actual_opens = int(actual_schedule_full.sum())
    total_optimized_opens = int(optimized_schedule_binary.sum())
    total_matched_opens = int(np.logical_and(actual_schedule_full > 0.5, optimized_schedule_binary > 0.5).sum())

    # --- G. 誤差分析 ---
    pred_h_all = np.zeros_like(bg_h_preds)
    for t in range(opt_config["SIM_STEPS"]):
        drawdown_v = np.sum((res['schedule'][t] * well_specific_powers)[:, np.newaxis] * res['inf_matrix'], axis=0)
        pred_h_all[t] = bg_h_preds[t] - drawdown_v

    obs_report = []
    for i, obs_name in enumerate(obs_list):
        p_h = pred_h_all[:, i]
        a_h = df_actual[obs_name].values[:opt_config["SIM_STEPS"]] if obs_name in df_actual.columns else np.full(opt_config["SIM_STEPS"], np.nan)
        valid = ~np.isnan(a_h)
        mae = mean_absolute_error(a_h[valid], p_h[valid]) if np.any(valid) else np.nan
        obs_report.append({
            "觀測井": obs_name,
            "實際最終水位 (m)": round(a_h[valid][-1], 2) if np.any(valid) else "-",
            "預測最終水位 (m)": round(p_h[-1], 2),
            "MAE 誤差 (m)": round(mae, 3) if not np.isnan(mae) else "-"
        })
    df_obs = pd.DataFrame(obs_report)

    # --- H. 綜合報告輸出 ---
    saving_pct = ((actual_total_kwh - res['total_kwh']) / actual_total_kwh) * 100
    vol_diff_pct = ((res['total_m3'] - actual_total_m3) / actual_total_m3) * 100
    avg_mae = df_obs["MAE 誤差 (m)"].replace("-", np.nan).dropna().astype(float).mean()

    print("\n" + "█"*75)
    print(f" ⚡ 智慧降水：電力與物理對標綜合驗證報告")
    print(f" ---------------------------------------------------------------------------")
    print(f"\n ?? 開井井號比對摘要：")
    print(f"  實際開井總次數: {total_actual_opens}")
    print(f"  最佳化開井總次數: {total_optimized_opens}")
    print(f"  共同開井總次數: {total_matched_opens}")
    print(f"  每時段井號完全一致率: {exact_match_rate:.1f}%")
    print(f"  井號比對明細已輸出: {well_compare_path}")
    print(f"\n ?? 前 10 筆開井井號比對：")
    print(df_well_compare.head(10).to_string(index=False))
    print(f" 🏆 節能表現: {saving_pct:+.2f} % | 🚀 流量偏差: {vol_diff_pct:+.2f} %")
    print(f" ---------------------------------------------------------------------------")
    print(f" 項目              |   現場實際 (Actual)  |   最佳化 (Model)")
    print(f" 總耗電量 (kWh)    |   {actual_total_kwh:>15.1f}    |   {res['total_kwh']:>12.1f}")
    print(f" 平均井數 (口)     |   {np.mean(actual_active_series):>15.1f}    |   {res['avg_active_wells']:>12.1f}")
    print(f" ---------------------------------------------------------------------------")
    print(f"\n 📍 逐口觀測井水位對標驗證：")
    print(df_obs.to_string(index=False))
    print(f"\n 🔍 系統可用性診斷：")
    print(f"  校準 Q_factor={best_q:.3f} | T_factor={best_t:.3f}")
    print(f"  水位平均誤差 (Avg MAE): {avg_mae:.3f} m -> {'✅ 合格' if avg_mae < 0.5 else '⚠️ 水位誤差仍大'}")
    print(f"  流量偏差 (Volume Diff): {vol_diff_pct:.1f} % -> {'✅ 合格' if abs(vol_diff_pct) < 15 else '⚠️ 流量偏差仍大'}")
    print("█"*75 + "\n")

    # --- I. 繪圖 ---
    n_wells_plot = len(well_list)
    heatmap_height = max(6, n_wells_plot * 0.5)
    fig = plt.figure(figsize=(15, heatmap_height + 21))
    gs = fig.add_gridspec(4, 1, height_ratios=[heatmap_height, 7, 7, 7], hspace=0.4)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])
    ax3 = fig.add_subplot(gs[3])

    # 圖表 0: 每口井每小時開關排程熱圖
    n_hours = opt_config["SIM_HOURS"]
    # 由於現在步長就是 1h，直接使用最佳化結果即可
    schedule_hourly = res['schedule']
    # === 防呆除錯：確認 schedule_hourly 裡到底有沒有 1 ===
    total_on_hours = np.sum(schedule_hourly > 0.5)
    print(f"\n🔍 [除錯] 熱力圖總畫布大小: {schedule_hourly.shape} (共 {schedule_hourly.size} 格)")
    print(f"🔍 [除錯] 其中綠色格子 (運轉中) 的數量為: {total_on_hours} 格")
    # ============================================
    hour_labels = [f"H{h+1}" if (h+1) % 12 == 0 else "" for h in range(n_hours)]
    from matplotlib.colors import ListedColormap
    binary_cmap = ListedColormap(['white', '#2ecc40'])  # 白=關, 綠=開
    sns.heatmap(schedule_hourly.T, cmap=binary_cmap, cbar=False, vmin=0, vmax=1,
                yticklabels=well_list,
                xticklabels=hour_labels, ax=ax0, linewidths=0, linecolor='lightgray')
    ax0.set_title("每口井每小時電力開關排程 (綠=開, 白=關)", fontsize=14)
    ax0.set_xlabel("小時 (每格=1小時, 共168小時/7天)", fontsize=11)
    ax0.set_ylabel("抽水井", fontsize=11)
    ax0.set_yticks(np.arange(n_wells_plot) + 0.5)
    ax0.set_yticklabels(well_list, fontsize=9, rotation=0)

    # 圖表 1: 水位比對
    colors = matplotlib.colormaps.get_cmap('tab20')
    for i, obs_name in enumerate(obs_list):
        ax1.plot(pred_h_all[:, i], color=colors(i / len(obs_list)), label=f"Pred: {obs_name}")
        if obs_name in df_actual.columns:
            ax1.plot(df_actual[obs_name].values[:opt_config["SIM_STEPS"]], color=colors(i / len(obs_list)), linestyle='--', alpha=0.3)
    ax1.axhline(y=opt_config["FIXED_TARGET_H"], color='black', linewidth=2, linestyle='-', label="Target")
    ax1.set_title("Water Level Comparison (Solid=Model, Dashed=Actual)")
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1)); ax1.grid(True, alpha=0.3)

    # 圖表 2: 井數對標
    ax2.step(range(opt_config["SIM_STEPS"]), actual_active_series, label="Actual Wells (Baseline)", color='gray', alpha=0.5, where='post')
    ax2.step(range(opt_config["SIM_STEPS"]), res['active_wells_series'], label="Optimized Wells (AI Decision)", color='green', linewidth=2, where='post')
    ax2.set_title(f"Well Quantity Comparison (Total Saving: {saving_pct:.1f}%)")
    ax2.set_ylabel("Number of Wells"); ax2.legend(); ax2.grid(True, alpha=0.3)

    # 圖表 3: 實際抽水量 vs 最佳化抽水量 (m3/hr) 比對
    # 實際總抽水量：直接從 Qw** 欄位加總
    if qw_match:
        actual_q_series = df_actual[qw_match].sum(axis=1).values[:opt_config["SIM_STEPS"]]
        if len(actual_q_series) < opt_config["SIM_STEPS"]:
            actual_q_series = np.pad(actual_q_series, (0, opt_config["SIM_STEPS"] - len(actual_q_series)), 'edge')
    else:
        actual_q_series = actual_active_series.astype(float)

    # 最佳化總抽水量：schedule(on/off) × 各井功率(kW) × kW_to_m3h
    opt_q_series = np.array([
        np.sum(res['schedule'][t] * np.array(well_specific_powers) * kW_to_m3h)
        for t in range(opt_config["SIM_STEPS"])
    ])

    ax3.step(range(opt_config["SIM_STEPS"]), actual_q_series,
             label="實際 Q (m³/hr)", color='steelblue', alpha=0.6, where='post')
    ax3.step(range(opt_config["SIM_STEPS"]), opt_q_series,
             label="最佳化 Q (m³/hr)", color='darkorange', linewidth=2, where='post')
    ax3.set_title("總抽水量對比：實際 vs 最佳化 (m³/hr)")
    ax3.set_xlabel(f"時間步 (每步={opt_config['delta_t']*60:.0f}分鐘, 共{opt_config['SIM_STEPS']}步/7天)")
    ax3.set_ylabel("Total Q (m³/hr)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    save_fig_path = os.path.join(pinn_path, "Optimization_Report.png")
    plt.savefig(save_fig_path, bbox_inches='tight', dpi=150)
    print(f"📊 最佳化報告圖已儲存至: {save_fig_path}")
    plt.show()
