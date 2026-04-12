import pandas as pd
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras import layers, Model


# ==========================================
# 💡 系統初始化與 GPU 優化
# ==========================================
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ 偵測到 GPU，已啟動加速訓練！")
    except RuntimeError as e:
        print(e)

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ==========================================
# 0. 輔助工具
# ==========================================
def calculate_wape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + 1e-10) * 100

# ==========================================
# 1. 超參數配置
# ==========================================
config = {
    "window_size": 96,       
    "d_model": 64,
    "num_heads": 4,
    "ff_dim": 128,
    "num_transformer_layers": 2,
    "dropout": 0.1,
    "nn_lr_init": 0.0005,      
    "phys_lr": 0.0001,        
    "decay_steps": 1000,      
    "decay_rate": 0.9,        
    "clip_norm": 1.0,         
    "T_init": 30.0,          
    "R_init": 100.0,         
    "C_init": 0.01,           
    "rw": 0.45,               
    "epochs": 500,            # 正式訓練
    "batch_size": 512,
    "lambda_phys_final": 2.0, 
    "lambda_flow": 2.0,       
    "warmup_epochs": 150,      
    "save_folder": "PINN_MAPE_Complete_Report3" ,
    "area_A": 3319.95,
    "DELTA_T": 0.5,
    "PREDICT_STEPS": 336,     # 7天共336步 (30分鐘一筆)
}

def map_sensor_id(w_name):
    num = "".join(filter(str.isdigit, str(w_name))) 
    return f"{int(num or 0):02d}"


class PositionalEncoding(layers.Layer):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.position_embedding = layers.Embedding(input_dim=max_len, output_dim=d_model)

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        return inputs + self.position_embedding(positions)


class TransformerEncoderBlock(layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, dropout):
        super().__init__()
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=max(1, d_model // num_heads),
            dropout=dropout,
        )
        self.ffn = tf.keras.Sequential(
            [
                layers.Dense(ff_dim, activation="gelu"),
                layers.Dropout(dropout),
                layers.Dense(d_model),
            ]
        )
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(dropout)
        self.drop2 = layers.Dropout(dropout)

    def call(self, inputs, training=False):
        attn_output = self.attention(inputs, inputs, training=training)
        attn_output = self.drop1(attn_output, training=training)
        x = self.norm1(inputs + attn_output)

        ffn_output = self.ffn(x, training=training)
        ffn_output = self.drop2(ffn_output, training=training)
        return self.norm2(x + ffn_output)

# ==========================================
# 2. 進階版 PINN 模型
# ==========================================
class PINN_Feedback_Model(Model):
    def __init__(self, dist_matrix, config, idx_map, h_min, h_max, q_min, q_max, total_samples):
        super(PINN_Feedback_Model, self).__init__()
        self.config, self.idx_map = config, idx_map
        self.dist_matrix = tf.cast(dist_matrix, tf.float32)
        self.h_min, self.h_max = tf.constant(h_min, dtype=tf.float32), tf.constant(h_max, dtype=tf.float32)
        self.q_min, self.q_max = tf.constant(q_min, dtype=tf.float32), tf.constant(q_max, dtype=tf.float32)
        self.total_samples = float(total_samples)
        
        self.T_log = tf.Variable(tf.fill([idx_map["h_end"]], tf.math.log(float(config["T_init"]))), dtype=tf.float32, name="T_log")
        self.R_log = tf.Variable(tf.fill([idx_map["h_end"]], tf.math.log(float(config["R_init"]))), dtype=tf.float32, name="R_log")
        self.C_log = tf.Variable(tf.fill([idx_map["n_wells"]], tf.math.log(float(config["C_init"]))), dtype=tf.float32, name="C_log")
        self.Sy_logit = tf.Variable(tf.fill([1], 0.0), dtype=tf.float32, name="Sy_logit")
        
        self.input_projection = layers.Dense(config["d_model"])
        self.positional_encoding = PositionalEncoding(config["window_size"], config["d_model"])
        self.transformer_blocks = [
            TransformerEncoderBlock(config["d_model"], config["num_heads"], config["ff_dim"], config["dropout"])
            for _ in range(config["num_transformer_layers"])
        ]
        self.sequence_pool = layers.GlobalAveragePooling1D()
        self.instant_dense = layers.Dense(32, activation='swish')
        self.head_h = tf.keras.Sequential([layers.Dense(128, activation='swish'), layers.Dense(64, activation='swish'), layers.Dense(idx_map["h_end"])])
        self.head_q = tf.keras.Sequential([layers.Dense(64, activation='relu'), layers.Dense(idx_map["n_wells"], activation='relu')])
        
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(config["nn_lr_init"], 1000, 0.9)
        self.opt_nn = tf.keras.optimizers.Adam(learning_rate=lr_schedule, global_clipnorm=config["clip_norm"])
        self.opt_phys = tf.keras.optimizers.Adam(learning_rate=config["phys_lr"])
        self.curr_epoch = tf.Variable(0.0, trainable=False, dtype=tf.float32)

    def call(self, inputs, training=None):
        current_C = tf.exp(self.C_log) 
        batch_size = tf.shape(inputs)[0]
        c_feature = tf.tile(tf.expand_dims(current_C, 0), [batch_size, 1]) 
        start_q = self.idx_map["n_obs"]
        current_pump = inputs[:, -1, start_q : self.idx_map["h_end"]] 
        seq_features = self.input_projection(inputs)
        seq_features = self.positional_encoding(seq_features)
        for block in self.transformer_blocks:
            seq_features = block(seq_features, training=training)
        transformer_out = self.sequence_pool(seq_features)
        combined = layers.Concatenate()([transformer_out, self.instant_dense(current_pump), c_feature])
        return self.head_h(combined), self.head_q(combined)

    def calculate_losses(self, hp, qp, y_h, y_q, X):
        T_s = tf.exp(tf.clip_by_value(self.T_log, tf.math.log(1.0), tf.math.log(500.0)))
        R_s = tf.exp(self.R_log) + 5.0
        C_s = tf.exp(tf.clip_by_value(self.C_log, -10.0, 2.0))
        Sy = tf.sigmoid(self.Sy_logit)
        h_real = hp * (self.h_max - self.h_min) + self.h_min
        q_real = tf.maximum(qp * (self.q_max - self.q_min) + self.q_min, 0.0)
        
        l_phys_spatial = 0.0
        n_o, n_w = self.idx_map["n_obs"], self.idx_map["n_wells"]
        for i in range(n_o + n_w):
            s_formation = 0.0
            for j in range(n_w):
                r = tf.where(self.dist_matrix[i, j] <= 0, self.config["rw"], self.dist_matrix[i, j])
                s_formation += (q_real[:, j] / (2.0 * 3.14159 * T_s[i] + 1e-4)) * tf.math.log(R_s[i] / (r + 1e-5))
            s_well_loss = C_s[i-n_o] * tf.square(q_real[:, i-n_o]) if i >= n_o else 0.0
            s_theo_total = s_formation + s_well_loss
            s_pred_actual = tf.maximum(self.h_max[i] - h_real[:, i], 0.0)
            l_phys_spatial += tf.reduce_mean(tf.abs(s_pred_actual - s_theo_total))

        h_prev = X[:, -1, :n_o] * (self.h_max[:n_o] - self.h_min[:n_o]) + self.h_min[:n_o]
        delta_h = tf.reduce_mean(h_real[:, :n_o], axis=1) - tf.reduce_mean(h_prev, axis=1)
        mass_balance_error = (self.config["area_A"] * Sy * delta_h) + (tf.reduce_sum(q_real, axis=1) * self.config["DELTA_T"])
        l_phys_temporal = tf.reduce_mean(tf.square(mass_balance_error - tf.reduce_mean(mass_balance_error)))
        
        return tf.reduce_mean(tf.square(y_h - hp)), (l_phys_spatial * 0.1) + l_phys_temporal, tf.reduce_mean(tf.square(y_q - qp))

    def train_step(self, data):
        X, y = data
        y_h, y_q = y[:, :self.idx_map["h_end"]], y[:, self.idx_map["h_end"]:]
        self.curr_epoch.assign_add(1.0 / (self.total_samples / self.config["batch_size"]))
        is_warmup = self.curr_epoch < float(self.config["warmup_epochs"])
        l_phys_w = tf.cond(is_warmup, lambda: 0.0, lambda: tf.minimum(self.config["lambda_phys_final"] * ((self.curr_epoch - self.config["warmup_epochs"]) / 20.0), self.config["lambda_phys_final"]))
        with tf.GradientTape(persistent=True) as tape:
            hp, qp = self(X, training=True)
            l_dat, l_phy, l_flo = self.calculate_losses(hp, qp, y_h, y_q, X)
            total_loss = (l_dat * 200.0) + (l_phys_w * (l_phy / 500.0)) + (self.config["lambda_flow"] * l_flo)
        self.opt_nn.apply_gradients(zip(tape.gradient(total_loss, self.trainable_variables), self.trainable_variables))
        def apply_phys():
            self.opt_phys.apply_gradients(zip(tape.gradient(total_loss, [self.T_log, self.R_log, self.C_log, self.Sy_logit]), [self.T_log, self.R_log, self.C_log, self.Sy_logit]))
            return tf.constant(1.0)
        tf.cond(is_warmup, lambda: tf.constant(0.0), apply_phys)
        return {"loss": total_loss, "l_dat": l_dat, "l_phy": l_phy, "l_flo": l_flo}

# ==========================================
# 3. 數據處理與執行
# ==========================================
if __name__ == "__main__":
    save_p = config["save_folder"]
    if not os.path.exists(save_p): os.makedirs(save_p)

    print("📊 數據處理中...")
    df_raw = pd.read_csv('Master_Training_Data_Continuous3.csv', index_col=0)
    df_raw.index = pd.to_datetime(df_raw.index)
    dist_df = pd.read_csv('Distance_Matrix.csv', index_col=0)

    analysis_start, analysis_end = "2020-09-11 11:30", "2021-01-31 00:00"
    df_master = df_raw.loc[analysis_start : analysis_end].copy().ffill().bfill()
    print(f"📊 數據範圍鎖定：{df_master.index.min()} 到 {df_master.index.max()}")

    obs_cols = [c for c in ['PA', 'PB', 'PC', 'FPS7', 'FPS8', 'FPS9', 'FPS2', 'FPS3', 'FPS4', 'FPS5', 'FPS6'] if c in df_raw.columns]
    wells_list = [w for w in ["PW01", "PW02", "PW03", "PW04", "PW06", "PW07", "PW08", "PW09", "PW010", "PW011", "PW012", "PW013"] if w in df_raw.columns and df_raw[w].abs().max() > 1e-6]
    flow_cols = [f"Qw{map_sensor_id(w)}" for w in wells_list]
    
    idx_map = {"n_obs": len(obs_cols), "n_wells": len(wells_list), "h_end": len(obs_cols)+len(wells_list), "flow_start": len(df_master.columns) - len(flow_cols)}
    h_min, h_max = df_master.iloc[:, :idx_map["h_end"]].min().values, df_master.iloc[:, :idx_map["h_end"]].max().values + 1e-7
    q_min, q_max = df_master[flow_cols].min().values, df_master[flow_cols].max().values + 1e-7
    scaler = MinMaxScaler().fit(df_master)

    def create_seq(df):
        data_s = scaler.transform(df)
        X, y = [], []
        for i in range(len(data_s) - config["window_size"]):
            X.append(data_s[i:i+config["window_size"]])
            y.append(np.concatenate([data_s[i+config["window_size"], :idx_map["h_end"]], data_s[i+config["window_size"], idx_map["flow_start"]:]]))
        return np.array(X), np.array(y)

    X_train, y_train = create_seq(df_master)
    model = PINN_Feedback_Model(dist_df.loc[obs_cols+wells_list, wells_list].values, config, idx_map, h_min, h_max, q_min, q_max, len(X_train))
    model.compile(); model.build(input_shape=X_train.shape)

    print(f"\n🚀 啟動強化版物理訓練 (Epochs: {config['epochs']})...")
    history = model.fit(X_train, y_train, epochs=config["epochs"], batch_size=config["batch_size"], verbose=1)

    # ==========================================
    # 🌟 4. [物理提取] 包含 Qin 與 Sy 學習
    # ==========================================
    print("\n🧹 正在進行物理診斷提取...")
    learned_sy = float(tf.sigmoid(model.Sy_logit).numpy()[0])
    hp_train_s, qp_train_s = model.predict(X_train)
    hp_train = hp_train_s * (h_max - h_min) + h_min
    qp_train = qp_train_s * (q_max - q_min) + q_min
    ht_train = y_train[:, :idx_map["h_end"]] * (h_max - h_min) + h_min
    qt_train = y_train[:, idx_map["h_end"]:] * (q_max - q_min) + q_min

    avg_h_pinn = np.mean(hp_train[:, :idx_map["n_obs"]], axis=1)
    dH_dt_pinn = np.zeros_like(avg_h_pinn)
    dH_dt_pinn[1:-1] = (avg_h_pinn[2:] - avg_h_pinn[:-2]) / (2.0 * config["DELTA_T"])
    total_Q_train = np.sum(qt_train, axis=1)

    valid_idx = slice(1, -1)
    X_reg = (config["area_A"] * dH_dt_pinn[valid_idx]).reshape(-1, 1)
    Y_reg = total_Q_train[valid_idx].reshape(-1, 1)
    qin_series = total_Q_train[valid_idx] + (config["area_A"] * learned_sy * dH_dt_pinn[valid_idx])
    qin_smooth = np.convolve(qin_series, np.ones(24)/24, mode='same')
    inflow_pinn_mean = float(np.mean(qin_smooth))

    r2_pinn_val = r2_score(total_Q_train[valid_idx], inflow_pinn_mean - (config["area_A"] * learned_sy * dH_dt_pinn[valid_idx]))

    # --- 儲存參數 ---
    np.save(f"{save_p}/learned_T.npy", np.mean(tf.exp(model.T_log).numpy()))
    np.save(f"{save_p}/learned_C.npy", tf.exp(model.C_log).numpy())
    np.save(f"{save_p}/qin_series.npy", qin_series)
    np.save(f"{save_p}/qin_smooth.npy", qin_smooth)
    np.save(f"{save_p}/calibrated_inflow_sy.npy", np.array([inflow_pinn_mean, learned_sy]))

    # --- 背景預測 (336步) ---
    print(f"\n🔮 正在自回歸預測未來 7 天 ({config['PREDICT_STEPS']}步) 背景水位...")
    curr_win = scaler.transform(df_master.tail(config["window_size"])).reshape(1, config["window_size"], -1)
    future_h = []
    for _ in range(config["PREDICT_STEPS"]):
        hp_s, _ = model(curr_win, training=False)
        future_h.append(hp_s.numpy()[0, :idx_map["n_obs"]] * (h_max[:idx_map["n_obs"]] - h_min[:idx_map["n_obs"]]) + h_min[:idx_map["n_obs"]])
        new_step = curr_win[:, -1:, :].copy()
        new_step[0, 0, :idx_map["n_obs"]] = hp_s.numpy()[0, :idx_map["n_obs"]]
        new_step[0, 0, idx_map["h_end"]:] = 0
        curr_win = np.append(curr_win[:, 1:, :], new_step, axis=1)
    np.save(f"{save_p}/background_h_7d.npy", np.array(future_h))

    # ==========================================
    # 🌟 5. [完整報告生成] 包含消失的 MAE 與 MAPE
    # ==========================================
    print("\n📊 正在生成 Full_Diagnostic_Report.csv (水位 & 流量誤差全記錄)...")
    diag_data = []
    T_res, C_res = tf.exp(model.T_log).numpy(), tf.exp(model.C_log).numpy()
    point_names = obs_cols + wells_list

    for i, name in enumerate(point_names):
        # A. 水位誤差 (所有點都有)
        h_mae = mean_absolute_error(ht_train[:, i], hp_train[:, i])
        h_mape = calculate_wape(ht_train[:, i], hp_train[:, i])
        
        # B. 初始化流量誤差與效率
        q_mae, q_mape, eff = np.nan, np.nan, 100.0
        
        # C. 抽水井專屬誤差計算
        if name in wells_list:
            w_idx = wells_list.index(name)
            # 流量誤差 (Actual qt vs Predicted qp)
            q_mae = mean_absolute_error(qt_train[:, w_idx], qp_train[:, w_idx])
            q_mape = calculate_wape(qt_train[:, w_idx], qp_train[:, w_idx])
            
            # 淤塞效率計算
            avg_q = np.mean(qp_train[:, w_idx])
            s_f = (avg_q / (2 * 3.14159 * T_res[i] + 1e-4)) * np.log(100.0 / 0.45)
            s_w = C_res[w_idx] * (avg_q ** 2)
            eff = (s_f / (s_f + s_w + 1e-6)) * 100

        diag_data.append({
            "Well": name, 
            "Type": "Pump" if name in wells_list else "Obs", 
            "MAE_Level(m)": h_mae, 
            "MAPE_Level(%)": h_mape,
            "MAE_Flow(m3/hr)": q_mae,
            "MAPE_Flow(%)": q_mape,
            "Efficiency(%)": eff
        })

    df_diag = pd.DataFrame(diag_data)
    df_diag.to_csv(f"{save_p}/Full_Diagnostic_Report.csv", index=False, encoding='utf-8-sig')
    print(df_diag.to_string(index=False, na_rep='-'))

    # ==========================================
    # 🌟 6. 繪圖邏輯 (包含 Y 軸格式化修正)
    # ==========================================
    print("📈 繪製所有診斷圖表...")
    
    # 1. 水位擬合 (1_Water_Level.png)
    rows_h = int(np.ceil(idx_map['n_obs']/3)); fig1, ax1 = plt.subplots(rows_h, 3, figsize=(18, 4*rows_h)); ax1 = ax1.flatten()
    for i in range(idx_map['n_obs']): ax1[i].plot(ht_train[:, i], alpha=0.5, label='Actual'); ax1[i].plot(hp_train[:, i], linestyle='--', label='PINN'); ax1[i].set_title(obs_cols[i]); ax1[i].legend()
    fig1.tight_layout(); fig1.savefig(f"{save_p}/1_Water_Level.png"); plt.close()

    # 2. 流量擬合 (2_Flow_Rate.png) 與 效率 (3_Efficiency_Chart.png)
    if idx_map['n_wells'] > 0:
        rows_q = int(np.ceil(idx_map['n_wells']/3)); fig2, ax2 = plt.subplots(rows_q, 3, figsize=(18, 4*rows_q)); ax2 = ax2.flatten()
        for i in range(idx_map['n_wells']): ax2[i].plot(qt_train[:, i], color='green', label='Act'); ax2[i].plot(qp_train[:, i], color='red', linestyle='--', label='Pred'); ax2[i].set_title(f"Flow: {wells_list[i]}"); ax2[i].legend()
        fig2.tight_layout(); fig2.savefig(f"{save_p}/2_Flow_Rate.png"); plt.close()
        plt.figure(figsize=(10, 6)); df_p = df_diag[df_diag['Type']=="Pump"]; plt.bar(df_p['Well'], df_p['Efficiency(%)']); plt.axhline(y=70, color='red', linestyle='--'); plt.title("Efficiency (%)"); plt.savefig(f"{save_p}/3_Efficiency_Chart.png"); plt.close()

    # 3. Loss 曲線 (4, 5, 6 號圖)
    for l_key, l_name, l_file in [('l_dat','Data Loss','4_Data_Loss.png'),('l_phy','Phys Loss','5_Phys_Loss.png'),('loss','Total Loss','6_Total_Loss.png')]:
        plt.figure(figsize=(8, 5))
        ax = plt.gca()
        plt.plot(history.history[l_key], color='black' if l_key=='loss' else None, linewidth=2)
        
        # 保持對數縮放
        plt.yscale('log')
        
        # 🌟 核心修正：強制顯示數值標籤
        from matplotlib.ticker import ScalarFormatter
        y_formatter = ScalarFormatter(useOffset=False)
        y_formatter.set_scientific(False) # 禁用 1e2 這種科學記號
        ax.yaxis.set_major_formatter(y_formatter)
        
        # 設置小數點後一位
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        
        # 自動增加刻度密度，防止範圍太小時沒刻度
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs='all', numticks=10))
        
        plt.title(l_name)
        plt.grid(True, which="both", alpha=0.3)
        plt.xlabel("Epochs")
        plt.ylabel("Loss Value")
        plt.savefig(f"{save_p}/{l_file}")
        plt.close()

    # 4. 物理直線圖 (8_PINN_Physics_Line.png)
    plt.figure(figsize=(10, 6)); plt.scatter(X_reg, Y_reg, alpha=0.3, color='gray', label='Cleaned Data (Actual)')
    x_r = np.array([np.min(X_reg), np.max(X_reg)])
    plt.plot(x_r, inflow_pinn_mean - (learned_sy * x_r), color='red', linewidth=3, label=f'PINN Physics Line (Sy={learned_sy:.3f}, R2={r2_pinn_val:.3f})')
    plt.xlabel("Area * dH/dt (m3/hr)"); plt.ylabel("Total Pumping Q (m3/hr)"); plt.title("Physical Mass Balance"); plt.legend(); plt.grid(True, alpha=0.3); plt.savefig(f"{save_p}/8_PINN_Physics_Line.png"); plt.close()

    # 5. Qin 時間序列 (9_Qin_TimeSeries.png)
    plt.figure(figsize=(12, 5)); plt.plot(qin_series, color='steelblue', alpha=0.4, label='Qin(t) Raw'); plt.plot(qin_smooth, color='darkorange', linewidth=2, label='Smoothed'); plt.axhline(inflow_pinn_mean, color='red', linestyle='--', label=f'Mean Qin={inflow_pinn_mean:.1f}'); plt.title("Derived Inflow TimeSeries"); plt.legend(); plt.grid(True, alpha=0.3); plt.savefig(f"{save_p}/9_Qin_TimeSeries.png"); plt.close()

    # 6. 背景水位恢復 (background_h_7d_curve.png)
    plt.figure(figsize=(12, 6)); plt.plot(np.arange(config["PREDICT_STEPS"])*0.5, np.array(future_h)); plt.axhline(y=-11.0, color='red', linestyle='--', label='Static SWL (-11m)'); plt.title("7-Day Background Recovery Prediction (Q=0)"); plt.xlabel("Hours"); plt.ylabel("Level (m)"); plt.grid(True, alpha=0.3); plt.savefig(f"{save_p}/background_h_7d_curve.png"); plt.close()

    print(f"✅ 任務完成！報告與圖表已儲存於 {save_p}")