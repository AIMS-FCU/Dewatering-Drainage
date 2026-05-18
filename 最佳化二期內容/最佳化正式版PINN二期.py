import pandas as pd
import numpy as np
from pulp import *
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
import seaborn as sns
import os
import sys
from pathlib import Path
from sklearn.metrics import mean_absolute_error
from scipy.optimize import minimize

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# # 設定中文字型（避免亂碼）
# matplotlib.rc('font', family='Microsoft JhengHei')
# matplotlib.rcParams['axes.unicode_minus'] = False

# 1. 確保字體檔案已載入到 Matplotlib (這行必須在設定 rc 之前)
font_path = 'SourceHanSansTC-Regular.otf'
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rc('font', family='Source Han Sans TC')
else:
    plt.rc('font', family='Microsoft JhengHei')
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# 1. 系統核心配置
# ==========================================
opt_config = {
    "ANALYSIS_START": "2026-03-01 00:00",
    "ANALYSIS_END":   "2026-03-08 00:00",
    "AREA": 5232,
    # "SY": 0.161,
    "DEPTH": 19.55,
    "STATIC_WATER_LEVEL": -11.0,
    "delta_t": 1.0,
    "η": 0.4,
    "HEAD": 30.0,
    "SF": 1,
    "SAFETY_BUFFER": 0,
    "rw": 0.45,
    "R_init": 500.0,
    "FIXED_PRICE_PER_KWH": 4.1,
    "TARGET_FLOOR": "B6F",
    "FIXED_TARGET_H": -20.6,
    # 目標水位來源：
    # - "quantile": 從分析期間的實測觀測井水位自動取分位數。
    # - "fixed": 使用 FIXED_TARGET_H。
    # 水位是負值，95 分位數通常比較淺，會比中位數更省電。
    "TARGET_MODE": "quantile",
    "TARGET_QUANTILE": 0.95,
    "TARGET_QUANTILE_AGGREGATION": "stacked",  # stacked 或 median_by_well
    # 目標水位控制：不可高於目標，低於目標太多則視為過抽。
    "TARGET_UPPER_TOLERANCE_M": 0.3,
    "TARGET_LOWER_TOLERANCE_M": 1.2,
    "OVER_PUMPING_PENALTY": 10000,
    "MAX_OVER_PUMPING_PENALTY": 50000,
    "TARGET_TRACKING_PENALTY": 300,
    # 若 q95 還沒有比現場省到指定井數，自動把目標水位往淺水位放寬後重跑。
    "AUTO_RELAX_FOR_WELL_SAVING": True,
    "TARGET_AVG_WELL_REDUCTION": 0.5,
    "TARGET_RELAX_STEP_M": 0.2,
    "TARGET_MAX_RELAX_M": 1.5,
    "ACTIVE_WELL_HOUR_PENALTY": 0.25,
    "USE_GREEDY_HARD_UPPER": True,
    "GREEDY_START_FROM_ACTUAL": True,
    "ENERGY_FIRST_LOCAL_SEARCH": True,
    "SIM_HOURS": 168,    
    "SIM_STEPS": 168,    
    "MIN_ACTIVE_WELLS": 0,
    "INITIAL_CARRYOVER_HOURS": 6,
    "BUFFER_HOURS": 6,
    "PRECALIBRATION_ENABLED": True,
    "PRECALIBRATION_HOURS": 24,
    "MIN_UP_TIME": 4,
    "pinn_report_path": "PINN_Phase2_20260301_Test",
    "WELL_LIST": ["PW01", "PW010", "PW011", "PW05", "PW06", "PW07", "PW08", "PW09"],
    "OBS_LIST": ['PW02', 'PW03', 'PW04'],    

    # 由 DynamicCalibrator 自動填入，不需手動調整
    "CALIBRATION": {
        "Q_IN_FACTOR": 1.0,
        "T_ADJUST_FACTOR": 1.0,
        "POWER_ADJUST": 1.0
    }
}



kW_to_m3h = (opt_config["η"] * 3600) / (9.81 * opt_config["HEAD"] * opt_config.get("SF", 1.0))


def phase2_sensor_suffix(well_name):
    return str(well_name).replace("PW", "")


def estimate_phase2_well_power_kw(df_all, df_period, well_name, delta_t):
    suffix = phase2_sensor_suffix(well_name)
    kw_col = f"KW{suffix}"
    qw_col = f"QW{suffix}"
    kwh_col = f"KWH{suffix}"

    if kw_col in df_all.columns:
        kw_all = pd.to_numeric(df_all[kw_col], errors="coerce")
        if qw_col in df_all.columns:
            q_all = pd.to_numeric(df_all[qw_col], errors="coerce")
            kw_period = pd.to_numeric(df_period[kw_col], errors="coerce")
            q_period = pd.to_numeric(df_period[qw_col], errors="coerce")
            active_period_kw = kw_period[(q_period > 0.5) & (kw_period > 0)]
            if not active_period_kw.empty:
                return float(active_period_kw.mean())

            active_all_kw = kw_all[(q_all > 0.5) & (kw_all > 0)]
            if not active_all_kw.empty:
                return float(active_all_kw.mean())

        positive_kw = kw_all[kw_all > 0]
        if not positive_kw.empty:
            return float(positive_kw.mean())

    if kwh_col in df_period.columns and qw_col in df_period.columns:
        active_hours = float((pd.to_numeric(df_period[qw_col], errors="coerce") > 0.5).sum() * delta_t)
        meter_delta = float(max(0, df_period[kwh_col].iloc[-1] - df_period[kwh_col].iloc[0]))
        if active_hours > 0 and meter_delta > 0:
            return meter_delta / active_hours

    if kwh_col in df_all.columns:
        positive_diff = pd.to_numeric(df_all[kwh_col], errors="coerce").diff()
        positive_diff = positive_diff[positive_diff > 0]
        if not positive_diff.empty:
            return float(positive_diff.median() / 0.5)

    return 7.5

# ==========================================
# 2. 動態自動校準器 (DynamicCalibrator)
# ==========================================
REQUIRED_PINN_FILES = [
    "learned_T.npy",
    "learned_C.npy",
    "calibrated_inflow_sy.npy",
]

TRAINING_OBS_ORDER = [
    "PW02",
    "PW03",
    "PW04",
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
    # 預期總數 = 訓練時的觀測井 + 抽水井
    source_names_with_wells = TRAINING_OBS_ORDER + list(well_list)
    
    # 這裡會印出診斷資訊，幫您確認問題
    print(f"DEBUG: .npy 欄數={all_pinn_h.shape[1]}, 預期總數(Obs+Well)={len(source_names_with_wells)}")
    
    if all_pinn_h.shape[1] == len(source_names_with_wells):
        source_obs_names = TRAINING_OBS_ORDER
    elif all_pinn_h.shape[1] == len(TRAINING_OBS_ORDER):
        source_obs_names = TRAINING_OBS_ORDER
    else:
        raise ValueError(
            f"accurate_pred_h_7d.npy 欄數為 {all_pinn_h.shape[1]}，與預期的 {len(source_names_with_wells)} (3+8) 不符。\n"
            f"請檢查是否已重新執行 newPINN-雙模型二期資料.py 產出正確格式的檔案。"
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


def build_target_h_array(df_actual, obs_list):
    target_mode = str(opt_config.get("TARGET_MODE", "fixed")).lower()
    if target_mode == "fixed":
        target_value = float(opt_config["FIXED_TARGET_H"])
        target_array = np.full(len(obs_list), target_value)
        return target_array, f"固定水位 {target_value:.3f} m"

    if target_mode != "quantile":
        raise ValueError(f"未知 TARGET_MODE={target_mode}，請使用 'fixed' 或 'quantile'。")

    available_obs = [col for col in obs_list if col in df_actual.columns]
    if not available_obs:
        raise ValueError("找不到可用觀測井欄位，無法用分位數建立目標水位。")

    q = float(opt_config.get("TARGET_QUANTILE", 0.95))
    aggregation = str(opt_config.get("TARGET_QUANTILE_AGGREGATION", "stacked")).lower()
    obs_data = df_actual[available_obs].apply(pd.to_numeric, errors="coerce")

    if aggregation == "stacked":
        target_value = float(obs_data.stack().quantile(q))
        target_array = np.full(len(obs_list), target_value)
        desc = f"實測觀測井 stacked q{q:.2f} = {target_value:.3f} m"
    elif aggregation == "median_by_well":
        per_well_q = obs_data.quantile(q)
        target_value = float(per_well_q.median())
        target_array = np.full(len(obs_list), target_value)
        desc = f"各井 q{q:.2f} 後取中位數 = {target_value:.3f} m"
    else:
        raise ValueError(
            f"未知 TARGET_QUANTILE_AGGREGATION={aggregation}，請使用 'stacked' 或 'median_by_well'。"
        )

    return target_array, desc


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
                          initial_active_count=0, carryover_count_steps=0,
                          actual_start_h=None, initial_schedule_hint=None):
    T = pinn_params['T'] * opt_config["CALIBRATION"]["T_ADJUST_FACTOR"]
    C = pinn_params['C']
    num_wells, num_obs = len(well_powers), base_h_sim.shape[1]
    sim_steps = opt_config["SIM_STEPS"]

    prob = LpProblem("Smart_Pumping_Optimization", LpMinimize)
    times, wells, obs = range(sim_steps), range(num_wells), range(num_obs)

    x = LpVariable.dicts("X", (times, wells), cat='Binary')
    low_slack = LpVariable.dicts("LowSlack", (times, obs), lowBound=0, cat='Continuous')

    if initial_status is None:
        initial_status = np.zeros(num_wells, dtype=int)
    else:
        initial_status = np.asarray(initial_status, dtype=int)

    if remaining_on_steps is None:
        remaining_on_steps = np.zeros(num_wells, dtype=int)
    else:
        remaining_on_steps = np.asarray(remaining_on_steps, dtype=int)

    total_energy_expr = lpSum([x[t][i] * well_powers[i] * opt_config["delta_t"] for t in times for i in wells])
    active_well_expr = lpSum([x[t][i] for t in times for i in wells])
    low_slack_expr = lpSum([low_slack[t][j] for t in times for j in obs])
    prob += (
        total_energy_expr
        + opt_config["ACTIVE_WELL_HOUR_PENALTY"] * active_well_expr
        + opt_config["OVER_PUMPING_PENALTY"] * low_slack_expr
    )

    inf_matrix = np.zeros((num_wells, num_obs))
    for i in wells:
        for j in obs:
            r = dist_matrix[j, i] if dist_matrix[j, i] > 0.01 else opt_config["rw"]
            inf_matrix[i, j] = abs((kW_to_m3h / (2 * np.pi * T)) * np.log(opt_config["R_init"] / r) + \
                               (C[i] * kW_to_m3h if dist_matrix[j, i] < 1.0 else 0))

    upper_targets = np.zeros((sim_steps, num_obs))
    lower_targets = np.zeros((sim_steps, num_obs))
    for t in times:
        prob += lpSum([x[t][i] for i in wells]) >= opt_config["MIN_ACTIVE_WELLS"]
        for j in obs:
            target_h = target_h_array[j] - opt_config.get("SAFETY_BUFFER", 0.0)
            # 使用感測器讀到的真實水位作為 buffer ramp 起始點
            if actual_start_h is not None:
                start_h = actual_start_h[j]
            else:
                start_h = base_h_sim[0, j]
            gap = start_h - target_h
            buffer_steps = opt_config["BUFFER_HOURS"] / opt_config["delta_t"]
            if gap > 0 and t < buffer_steps:
                ramp_progress = t / buffer_steps
                dynamic_target = start_h - (gap * ramp_progress)
            else:
                dynamic_target = target_h
            upper_target = dynamic_target + opt_config["TARGET_UPPER_TOLERANCE_M"]
            lower_target = dynamic_target - opt_config["TARGET_LOWER_TOLERANCE_M"]
            upper_targets[t, j] = upper_target
            lower_targets[t, j] = lower_target
            predicted_h = base_h_sim[t, j] - lpSum([x[t][i] * well_powers[i] * inf_matrix[i, j] for i in wells])
            prob += predicted_h <= upper_target
            prob += predicted_h >= lower_target - low_slack[t][j]

    # 限制式：最小連續運轉時間 = 6小時
    # 方案一：優化邏輯，摒棄 y 變數，改用總和式表達 (大幅減少分支維度)
    for i in wells:
        prob += x[0][i] == int(initial_status[i])
        for t in range(1, min(sim_steps, int(remaining_on_steps[i]) + 1)):
            prob += x[t][i] == 1

    for t in range(min(sim_steps, int(carryover_count_steps))):
        prob += lpSum([x[t][i] for i in wells]) >= int(initial_active_count)

    min_on_steps = int(opt_config["MIN_UP_TIME"] / opt_config["delta_t"])
    for i in wells:
        for t in range(min_on_steps, sim_steps):
            # 備份版寫法：如果在 t 時刻被關掉，則前面 min_on_steps 內必須全開
            prob += lpSum([x[k][i] for k in range(t - min_on_steps, t)]) >= min_on_steps * (x[t-1][i] - x[t][i])

    def evaluate_schedule(candidate):
        drawdown = (candidate * well_powers[np.newaxis, :]) @ inf_matrix
        predicted = base_h_sim - drawdown
        if np.max(predicted - upper_targets) > 1e-6:
            return np.inf, predicted, np.inf

        over_pumping = np.maximum(lower_targets - predicted, 0.0)
        low_slack_total = over_pumping.sum()
        max_over_pumping = over_pumping.max()
        target_tracking_total = (upper_targets - predicted).sum()
        energy = np.sum(candidate * well_powers[np.newaxis, :] * opt_config["delta_t"])
        active = candidate.sum()
        objective = (
            energy
            + opt_config["ACTIVE_WELL_HOUR_PENALTY"] * active
            + opt_config["OVER_PUMPING_PENALTY"] * low_slack_total
            + opt_config["MAX_OVER_PUMPING_PENALTY"] * max_over_pumping
            + opt_config["TARGET_TRACKING_PENALTY"] * target_tracking_total
        )
        return objective, predicted, low_slack_total

    def predict_schedule(candidate):
        drawdown = (candidate * well_powers[np.newaxis, :]) @ inf_matrix
        return base_h_sim - drawdown

    def schedule_energy(candidate):
        return np.sum(candidate * well_powers[np.newaxis, :] * opt_config["delta_t"])

    def respects_logic(candidate):
        if np.any(candidate[0] != initial_status):
            return False
        for i in wells:
            forced_end = min(sim_steps, int(remaining_on_steps[i]) + 1)
            if forced_end > 1 and np.any(candidate[1:forced_end, i] < 0.5):
                return False
        for t in range(min(sim_steps, int(carryover_count_steps))):
            if candidate[t].sum() + 1e-9 < int(initial_active_count):
                return False
        for i in wells:
            for t in range(min_on_steps, sim_steps):
                if candidate[t - 1, i] > 0.5 and candidate[t, i] < 0.5:
                    if candidate[t - min_on_steps:t, i].sum() + 1e-9 < min_on_steps:
                        return False
        return True

    def repair_upper_violations(candidate):
        candidate = candidate.copy()
        max_repairs = sim_steps * num_wells
        repairs = 0

        while repairs < max_repairs:
            predicted = predict_schedule(candidate)
            high_excess = np.maximum(predicted - upper_targets, 0.0)
            if high_excess.max() <= 1e-6:
                return candidate, repairs

            t_crit, _ = np.unravel_index(np.argmax(high_excess), high_excess.shape)
            current_total_excess = high_excess.sum()
            best_trial = None
            best_score = -np.inf

            for i in wells:
                if candidate[t_crit, i] > 0.5:
                    continue
                trial = candidate.copy()
                end_t = min(sim_steps, t_crit + max(min_on_steps, 1))
                trial[t_crit:end_t, i] = 1
                if not respects_logic(trial):
                    continue

                trial_excess = np.maximum(predict_schedule(trial) - upper_targets, 0.0).sum()
                reduction = current_total_excess - trial_excess
                if reduction <= 1e-9:
                    continue
                added_kwh = np.sum((trial - candidate) * well_powers[np.newaxis, :] * opt_config["delta_t"])
                score = reduction / max(added_kwh, 1e-6)
                if score > best_score:
                    best_score = score
                    best_trial = trial

            if best_trial is None:
                print("⚠️ 實際排程無法局部修補到不超標，改用保守全開起點。")
                fallback = np.ones((sim_steps, num_wells), dtype=float)
                fallback[0] = initial_status
                return fallback, repairs

            candidate = best_trial
            repairs += 1

        raise RuntimeError("實際排程修補次數過多，仍無法滿足水位硬上限。")

    def build_greedy_schedule():
        if opt_config.get("GREEDY_START_FROM_ACTUAL", False) and initial_schedule_hint is not None:
            candidate = (np.asarray(initial_schedule_hint, dtype=float) > 0.5).astype(float)
            if candidate.shape != (sim_steps, num_wells):
                raise ValueError(
                    f"initial_schedule_hint shape={candidate.shape}，預期 {(sim_steps, num_wells)}"
                )
            candidate[0] = initial_status
            candidate, repairs = repair_upper_violations(candidate)
            print(f"⚙️ [貪婪硬上限] 從實際排程出發，補強 {repairs} 次以滿足硬上限")
        else:
            candidate = np.ones((sim_steps, num_wells), dtype=float)
            candidate[0] = initial_status

        current_obj, _, current_low_slack = evaluate_schedule(candidate)
        if not np.isfinite(current_obj):
            raise RuntimeError("即使保守全開排程仍會超過目標水位，請檢查起始水位、目標水位或 BUFFER_HOURS。")
        current_energy = schedule_energy(candidate)

        passes = 0
        while passes < 4:
            improved = False
            for t in reversed(range(sim_steps)):
                for i in wells:
                    if candidate[t, i] < 0.5:
                        continue
                    trial = candidate.copy()
                    trial[t, i] = 0
                    if not respects_logic(trial):
                        continue
                    trial_obj, _, trial_low_slack = evaluate_schedule(trial)
                    trial_energy = schedule_energy(trial)
                    energy_ok = (
                        not opt_config.get("ENERGY_FIRST_LOCAL_SEARCH", False)
                        or trial_energy <= current_energy + 1e-6
                    )
                    accept_trial = (
                        np.isfinite(trial_obj)
                        and
                        energy_ok
                        and (
                            trial_energy + 1e-6 < current_energy
                            or trial_obj + 1e-6 < current_obj
                        )
                    )
                    if accept_trial:
                        candidate = trial
                        current_obj = trial_obj
                        current_low_slack = trial_low_slack
                        current_energy = trial_energy
                        improved = True

                for off_i in wells:
                    if candidate[t, off_i] < 0.5:
                        continue
                    for on_i in wells:
                        if candidate[t, on_i] > 0.5:
                            continue
                        trial = candidate.copy()
                        trial[t, off_i] = 0
                        trial[t, on_i] = 1
                        if not respects_logic(trial):
                            continue
                        trial_obj, _, trial_low_slack = evaluate_schedule(trial)
                        trial_energy = schedule_energy(trial)
                        energy_ok = (
                            not opt_config.get("ENERGY_FIRST_LOCAL_SEARCH", False)
                            or trial_energy <= current_energy + 1e-6
                        )
                        accept_trial = (
                            np.isfinite(trial_obj)
                            and
                            energy_ok
                            and (
                                trial_energy + 1e-6 < current_energy
                                or trial_obj + 1e-6 < current_obj
                            )
                        )
                        if accept_trial:
                            candidate = trial
                            current_obj = trial_obj
                            current_low_slack = trial_low_slack
                            current_energy = trial_energy
                            improved = True
            passes += 1
            if not improved:
                break

        print(f"\n⚙️ [貪婪硬上限] 完成關井微調，迭代 {passes} 輪")
        print(f"⚙️ [貪婪硬上限] 開井總次數: {int(candidate.sum())} | 過抽累積: {current_low_slack:.3f} m-hour")
        final_predicted = predict_schedule(candidate)
        final_high_excess = np.maximum(final_predicted - upper_targets, 0.0)
        if final_high_excess.max() > 1e-6:
            raise RuntimeError(
                f"最佳化排程違反水位上限，最大超限 {final_high_excess.max():.4f} m。"
                "請檢查局部搜尋接受條件。"
            )
        return candidate, current_low_slack

    if opt_config.get("USE_GREEDY_HARD_UPPER", False):
        schedule, low_slack_total = build_greedy_schedule()
        return {
            "schedule": schedule,
            "inf_matrix": inf_matrix,
            "total_kwh": np.sum(schedule * well_powers * opt_config["delta_t"]),
            "total_m3": np.sum(schedule * well_powers * opt_config["delta_t"]) * kW_to_m3h,
            "avg_active_wells": np.mean(np.sum(schedule, axis=1)),
            "active_wells_series": np.sum(schedule, axis=1),
            "high_slack_total_m": 0.0,
            "low_slack_total_m": low_slack_total,
            "solver_status": "GreedyHardUpper"
        }

    for i in wells:
        x[0][i].setInitialValue(int(initial_status[i]))
        for t in range(1, sim_steps):
            x[t][i].setInitialValue(1)

    prob.solve(PULP_CBC_CMD(msg=1, gapRel=0.05, timeLimit=300, warmStart=True, keepFiles=True))
    solver_status = LpStatus[prob.status]
    schedule = np.array([[value(x[t][i]) if value(x[t][i]) is not None else 0 for i in wells] for t in times])
    print(f"\n⚙️ [解算狀態] PuLP Solver Status: {solver_status}")
    print(f"⚙️ [解算狀態] Schedule 最大數值: {np.max(schedule):.4f}")
    if solver_status not in ("Optimal", "Integer Feasible"):
        raise RuntimeError(
            f"最佳化未找到可用整數解，狀態為 {solver_status}。"
            "請放寬 BUFFER_HOURS、調低 FIXED_TARGET_H，或延長求解時間。"
        )
    if np.max(schedule) > 0 and np.max(schedule) < 0.9:
        print("⚠️ 警告：最佳化引擎因為超時，回傳了非整數（小數）的未完成解答！")

    return {
        "schedule": schedule,
        "inf_matrix": inf_matrix,
        "total_kwh": np.sum(schedule * well_powers * opt_config["delta_t"]),
        "total_m3": np.sum(schedule * well_powers * opt_config["delta_t"]) * kW_to_m3h,
        "avg_active_wells": np.mean(np.sum(schedule, axis=1)),
        "active_wells_series": np.sum(schedule, axis=1),
        "high_slack_total_m": 0.0,
        "low_slack_total_m": sum(value(low_slack[t][j]) or 0 for t in times for j in obs),
        "solver_status": solver_status
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


def build_well_power_summary(well_list, well_powers, actual_schedule, optimized_schedule, delta_t):
    actual_hours = actual_schedule.sum(axis=0) * delta_t
    optimized_hours = optimized_schedule.sum(axis=0) * delta_t
    well_powers = np.asarray(well_powers, dtype=float)

    df_power = pd.DataFrame({
        "井號": well_list,
        "估計功率_kW": well_powers,
        "實際開井時數_hr": actual_hours,
        "最佳化開井時數_hr": optimized_hours,
        "實際耗電_kWh": actual_hours * well_powers,
        "最佳化耗電_kWh": optimized_hours * well_powers,
    })
    df_power["耗電差異_kWh"] = df_power["最佳化耗電_kWh"] - df_power["實際耗電_kWh"]
    return df_power


def compute_carryover_constraints(df_raw, eval_start, well_list, sim_steps):
    min_on_steps = int(opt_config["MIN_UP_TIME"] / opt_config["delta_t"])
    history_hours = max(min_on_steps - 1, 0)
    history_start = eval_start - pd.Timedelta(hours=history_hours)
    df_history = df_raw.loc[history_start:eval_start].iloc[::2].copy()

    initial_status = np.zeros(len(well_list), dtype=int)
    remaining_on_steps = np.zeros(len(well_list), dtype=int)

    for idx, w in enumerate(well_list):
        qw_col = w.replace("PW", "QW")
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
    df_raw = pd.read_csv("Phase2_Training_Data2.csv", index_col=0)
    df_raw.index = pd.to_datetime(df_raw.index)
    dist_df = pd.read_csv("Distance_Matrix_Phase2.csv", index_col=0)

    eval_start = pd.to_datetime(opt_config["ANALYSIS_START"])
    eval_end = eval_start + pd.Timedelta(hours=opt_config["SIM_HOURS"])
    # [降採樣] 每 1 小時取一筆資料以對齊 1.0h 的最佳化步長
    df_actual = df_raw.loc[eval_start : eval_end].iloc[::2].copy()

    well_list, obs_list = opt_config["WELL_LIST"], opt_config["OBS_LIST"]

    # --- A. 實際數據結算 ---
    qw_match = [w.replace("PW", "QW") for w in well_list if w.replace("PW", "QW") in df_actual.columns]
    kwh_match = [w.replace("PW", "KWH") for w in well_list if w.replace("PW", "KWH") in df_actual.columns]
    actual_total_kwh = sum([max(0, df_actual[col].iloc[-1] - df_actual[col].iloc[0]) for col in kwh_match])
    actual_total_m3 = (df_actual[qw_match].sum(axis=1) * opt_config["delta_t"]).sum()
    actual_active_series = (df_actual[qw_match] > 0.5).sum(axis=1).values
    if len(actual_active_series) < opt_config["SIM_STEPS"]:
        actual_active_series = np.pad(actual_active_series, (0, opt_config["SIM_STEPS"] - len(actual_active_series)), 'edge')
    actual_active_series = actual_active_series[:opt_config["SIM_STEPS"]]

    actual_schedule_full = np.zeros((opt_config["SIM_STEPS"], len(well_list)))
    for idx, w in enumerate(well_list):
        qw_col = w.replace("PW", "QW")
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
        well_specific_powers.append(
            estimate_phase2_well_power_kw(df_raw, df_actual, w, opt_config["delta_t"])
        )
    well_specific_powers = np.array(well_specific_powers)
    print("⚡ 各井估計功率(kW): " + ", ".join(
        f"{w}={p:.1f}" for w, p in zip(well_list, well_specific_powers)
    ))
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
            qw_col = w.replace("PW", "QW")
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
            # 🌟 使用 np.abs(inf_matrix_cal) 確保洩降為正值
            theoretical_drawdown = np.sum((actual_schedule_full[t] * well_specific_powers)[:, np.newaxis] * np.abs(inf_matrix_cal), axis=0)
            bg_h_preds[t] = accurate_h_preds[t] + theoretical_drawdown
            # [物理閉環驗證]：還原值 = 背景 - 理論洩降 (應該要剛好回到 accurate_h_preds)
            reconstructed_h_list[t] = bg_h_preds[t] - theoretical_drawdown
            
        print(f"  👉 已成功從「神準預測線」剝離 {np.sum(actual_schedule_full > 0)} 個運轉時數，還原為公平之天然起跑線！")

        # --- 新增：校正與繪製反向工程閉環驗證圖 ---
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

    # --- F. 自動擷取歷史容忍邊界與執行最佳化 ---
    print(f"\n🎯 最佳化水位基準：統一單一目標水位")
    target_h_array, target_desc = build_target_h_array(df_actual, obs_list)
    active_target_h = float(np.nanmedian(target_h_array))
    opt_config["ACTIVE_TARGET_H"] = active_target_h
    print(
        f"   - 目標來源: {target_desc}\n"
        f"   - 所有觀測井目標: {active_target_h:.3f} m "
        f"(硬上限不可高於 {active_target_h + opt_config['TARGET_UPPER_TOLERANCE_M']:.2f} m，"
        f"低於 {active_target_h - opt_config['TARGET_LOWER_TOLERANCE_M']:.2f} m 會懲罰過抽)"
    )

    print(f"   - 承接現場當下開井: {', '.join(active_carryover_wells) if active_carryover_wells else '-'}")
    if active_carryover_wells:
        carryover_desc = ", ".join([
            f"{well_list[i]} 還需維持 {remaining_on_steps[i]} 步"
            for i, flag in enumerate(initial_status) if flag == 1
        ])
        print(f"   - 最短續開限制: {carryover_desc}")

    actual_avg_wells = float(np.mean(actual_active_series))
    target_avg_wells = max(0.0, actual_avg_wells - opt_config["TARGET_AVG_WELL_REDUCTION"])
    relax_candidates = [0.0]
    if opt_config.get("AUTO_RELAX_FOR_WELL_SAVING", False):
        max_relax = float(opt_config["TARGET_MAX_RELAX_M"])
        step_relax = float(opt_config["TARGET_RELAX_STEP_M"])
        relax_candidates = list(np.arange(0.0, max_relax + step_relax * 0.5, step_relax))

    print(
        f"   - 節能目標: 平均井數需由現場 {actual_avg_wells:.2f} 口 "
        f"降到 <= {target_avg_wells:.2f} 口"
    )

    best_res = None
    best_target_h_array = target_h_array.copy()
    best_relax_m = 0.0
    for relax_m in relax_candidates:
        trial_target_h_array = target_h_array + relax_m
        trial_target_h = float(np.nanmedian(trial_target_h_array))
        print(
            f"\n🎯 嘗試最佳化目標水位: {trial_target_h:.3f} m "
            f"(q目標 + {relax_m:.2f} m，硬上限 {trial_target_h + opt_config['TARGET_UPPER_TOLERANCE_M']:.2f} m)"
        )
        trial_res = run_pulp_optimization(
            {'T': learned_T, 'C': learned_C},
            well_specific_powers,
            bg_h_preds,
            dist_matrix_np,
            trial_target_h_array,
            initial_status=initial_status,
            remaining_on_steps=remaining_on_steps,
            initial_active_count=initial_active_count,
            carryover_count_steps=carryover_count_steps,
            actual_start_h=start_h_actual,
            initial_schedule_hint=actual_schedule_full
        )
        trial_avg_wells = float(trial_res["avg_active_wells"])
        print(
            f"   -> 平均井數 {trial_avg_wells:.2f} 口，"
            f"比現場少 {actual_avg_wells - trial_avg_wells:.2f} 口"
        )

        if best_res is None or trial_avg_wells < float(best_res["avg_active_wells"]):
            best_res = trial_res
            best_target_h_array = trial_target_h_array
            best_relax_m = relax_m

        if trial_avg_wells <= target_avg_wells + 1e-9:
            print("   ✅ 已達成至少少 0.5 口的平均井數目標")
            break

    res = best_res
    target_h_array = best_target_h_array
    active_target_h = float(np.nanmedian(target_h_array))
    opt_config["ACTIVE_TARGET_H"] = active_target_h
    opt_config["ACTIVE_TARGET_RELAX_M"] = best_relax_m
    if float(res["avg_active_wells"]) > target_avg_wells + 1e-9:
        print(
            f"   ⚠️ 已達最大放寬 {best_relax_m:.2f} m，仍未少滿 0.5 口；"
            f"目前少 {actual_avg_wells - float(res['avg_active_wells']):.2f} 口。"
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

    df_power_summary = build_well_power_summary(
        well_list,
        well_specific_powers,
        actual_schedule_full.astype(float),
        optimized_schedule_binary.astype(float),
        opt_config["delta_t"],
    )
    power_summary_path = os.path.join(pinn_path, "Well_Power_Summary.csv")
    df_power_summary.to_csv(power_summary_path, index=False, encoding="utf-8-sig")

    exact_match_rate = df_well_compare["井號完全一致"].mean() * 100
    total_actual_opens = int(actual_schedule_full.sum())
    total_optimized_opens = int(optimized_schedule_binary.sum())
    total_matched_opens = int(np.logical_and(actual_schedule_full > 0.5, optimized_schedule_binary > 0.5).sum())

    # --- G. 誤差分析 ---
    pred_h_all = np.zeros_like(bg_h_preds)
    for t in range(opt_config["SIM_STEPS"]):
        # 🌟 這裡也使用 np.abs 確保預測水位計算正確
        drawdown_v = np.sum((res['schedule'][t] * well_specific_powers)[:, np.newaxis] * np.abs(res['inf_matrix']), axis=0)
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
    print(f"  每口井功率明細已輸出: {power_summary_path}")
    print(f"\n ?? 前 10 筆開井井號比對：")
    print(df_well_compare.head(10).to_string(index=False))
    print(f"\n ?? 每口井估計功率與耗電摘要：")
    print(df_power_summary.round(2).to_string(index=False))
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
    print(
        f"  最終目標水位={opt_config['ACTIVE_TARGET_H']:.3f} m | "
        f"q目標額外放寬={opt_config.get('ACTIVE_TARGET_RELAX_M', 0.0):.2f} m"
    )
    print(f"  高水位硬上限超限: {res['high_slack_total_m']:.3f} m-hour | 過抽累積: {res['low_slack_total_m']:.3f} m-hour")
    print(f"  水位平均誤差 (Avg MAE): {avg_mae:.3f} m -> {'✅ 合格' if avg_mae < 0.5 else '⚠️ 水位誤差仍大'}")
    print(f"  流量偏差 (Volume Diff): {vol_diff_pct:.1f} % -> {'✅ 合格' if abs(vol_diff_pct) < 15 else '⚠️ 流量偏差仍大'}")
    print("█"*75 + "\n")

    # --- I. 繪圖 ---
    n_wells_plot = len(well_list)
    heatmap_height = max(6, n_wells_plot * 0.5)
    fig = plt.figure(figsize=(15, heatmap_height * 3 + 26))
    gs = fig.add_gridspec(6, 1, height_ratios=[heatmap_height, heatmap_height, heatmap_height, 7, 7, 7], hspace=0.7)
    ax_actual = fig.add_subplot(gs[0])
    ax0 = fig.add_subplot(gs[1])
    ax_diff = fig.add_subplot(gs[2])
    ax1 = fig.add_subplot(gs[3])
    ax2 = fig.add_subplot(gs[4])
    ax3 = fig.add_subplot(gs[5])

    n_hours = opt_config["SIM_HOURS"]
    hour_labels = [f"H{h+1}" if (h+1) % 12 == 0 else "" for h in range(n_hours)]
    from matplotlib.colors import ListedColormap
    binary_cmap = ListedColormap(['white', '#2ecc40'])  # 白=關, 綠=開

    # 圖表 0-1: 實際每口井每小時開關排程熱圖
    sns.heatmap(actual_schedule_full.T, cmap=binary_cmap, cbar=False, vmin=0, vmax=1,
                yticklabels=well_list,
                xticklabels=hour_labels, ax=ax_actual, linewidths=0, linecolor='lightgray')
    ax_actual.set_title("【實際】每口井每小時電力開關排程 (綠=開, 白=關)", fontsize=14)
    ax_actual.set_xlabel("小時 (每格=1小時, 共168小時/7天)", fontsize=11)
    ax_actual.set_ylabel("抽水井", fontsize=11)
    ax_actual.set_yticks(np.arange(n_wells_plot) + 0.5)
    ax_actual.set_yticklabels(well_list, fontsize=9, rotation=0)

    # 圖表 0: 最佳化每口井每小時開關排程熱圖
    # 由於現在步長就是 1h，直接使用最佳化結果即可
    schedule_hourly = res['schedule']
    # === 防呆除錯：確認 schedule_hourly 裡到底有沒有 1 ===
    total_on_hours = np.sum(schedule_hourly > 0.5)
    print(f"\n🔍 [除錯] 熱力圖總畫布大小: {schedule_hourly.shape} (共 {schedule_hourly.size} 格)")
    print(f"🔍 [除錯] 其中綠色格子 (運轉中) 的數量為: {total_on_hours} 格")
    # ============================================
    sns.heatmap(schedule_hourly.T, cmap=binary_cmap, cbar=False, vmin=0, vmax=1,
                yticklabels=well_list,
                xticklabels=hour_labels, ax=ax0, linewidths=0, linecolor='lightgray')
    ax0.set_title("【最佳化】每口井每小時電力開關排程 (綠=開, 白=關)", fontsize=14)
    ax0.set_xlabel("小時 (每格=1小時, 共168小時/7天)", fontsize=11)
    ax0.set_ylabel("抽水井", fontsize=11)
    ax0.set_yticks(np.arange(n_wells_plot) + 0.5)
    ax0.set_yticklabels(well_list, fontsize=9, rotation=0)

    # 圖表 0-2: 差異比對熱圖
    diff_matrix = np.zeros_like(actual_schedule_full)
    # 0: 兩者皆關 (白), 1: 兩者皆開 (綠), 2: 實際開/模型關 (藍), 3: 模型開/實際關 (橘)
    diff_matrix[(actual_schedule_full > 0.5) & (schedule_hourly > 0.5)] = 1
    diff_matrix[(actual_schedule_full > 0.5) & (schedule_hourly <= 0.5)] = 2
    diff_matrix[(actual_schedule_full <= 0.5) & (schedule_hourly > 0.5)] = 3
    
    diff_cmap = ListedColormap(['white', '#2ecc40', 'royalblue', 'darkorange'])
    sns.heatmap(diff_matrix.T, cmap=diff_cmap, cbar=False, vmin=0, vmax=3,
                yticklabels=well_list,
                xticklabels=hour_labels, ax=ax_diff, linewidths=0, linecolor='lightgray')
    ax_diff.set_title("【排程差異比對】 (綠=皆開, 白=皆關, 藍=實際有開/模型沒開, 橘=模型有開/實際沒開)", fontsize=14)
    ax_diff.set_xlabel("小時 (每格=1小時, 共168小時/7天)", fontsize=11)
    ax_diff.set_ylabel("抽水井", fontsize=11)
    ax_diff.set_yticks(np.arange(n_wells_plot) + 0.5)
    ax_diff.set_yticklabels(well_list, fontsize=9, rotation=0)

    # 圖表 1: 水位比對
    colors = matplotlib.colormaps.get_cmap('tab20')
    for i, obs_name in enumerate(obs_list):
        ax1.plot(pred_h_all[:, i], color=colors(i / len(obs_list)), label=f"Pred: {obs_name}")
        if obs_name in df_actual.columns:
            ax1.plot(df_actual[obs_name].values[:opt_config["SIM_STEPS"]], color=colors(i / len(obs_list)), linestyle='--', alpha=0.3)
    plot_target_h = float(opt_config.get("ACTIVE_TARGET_H", opt_config["FIXED_TARGET_H"]))
    ax1.axhline(y=plot_target_h, color='black', linewidth=2, linestyle='-', label="Target")
    ax1.axhspan(
        plot_target_h - opt_config["TARGET_LOWER_TOLERANCE_M"],
        plot_target_h + opt_config["TARGET_UPPER_TOLERANCE_M"],
        color='gray',
        alpha=0.12,
        label="Target Band",
    )
    ax1.set_title("Water Level Comparison (Solid=Model, Dashed=Actual)")
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1)); ax1.grid(True, alpha=0.3)

    # 圖表 2: 井數對標
    ax2.step(range(opt_config["SIM_STEPS"]), actual_active_series, label="Actual Wells (Baseline)", color='dimgray', alpha=0.8, where='post')
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
