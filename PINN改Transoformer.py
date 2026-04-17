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
    "window_size": 144,       # 24*7*2 如果要改14天請另外註解 2天 7天測試
    "d_model": 64,
    "num_heads": 4,
    "ff_dim": 128,
    "num_transformer_layers": 4,
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
    "epochs": 300,            # 正式訓練
    "batch_size": 256,        # 原為 512，因為 window_size 放大到 120 導致 OOM 記憶體爆掉，調降來解決
    "lambda_phys_final": 2.0, 
    "lambda_flow": 2.0,       
    "warmup_epochs": 150,      
    "save_folder": "PINN_MAPE_Complete_Report3",
    "area_A": 3319.95,
    "DELTA_T": 0.5,
    "PREDICT_START": "2021-05-14 00:00",  # 指定預測起始時間供最佳化使用
    "PREDICT_END":   "2021-05-21 00:00",  # 指定預測結束時間供最佳化使用
    "TEST_START":    "2021-05-01 00:00",  # 盲測考卷起始時間
    "TEST_END":      "2021-06-01 00:00",  # 盲測考卷結束時間
    "TRAIN_CUTOFF":  "2021-07-01 00:00",  # [新功能] 數據切斷點，忽略此日期後的低抽水資料
    "USE_KFOLD":     False,               # 是否啟用 5-Fold Cross Validation
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
        self.head_inflow = tf.keras.Sequential([layers.Dense(32, activation='swish'), layers.Dense(1, activation='softplus')]) # 動態 Qin 輸出頭
        
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
        return self.head_h(combined), self.head_q(combined), self.head_inflow(combined)

    def calculate_losses(self, hp, qp, qin_p, y_h, y_q, X):
        T_s = tf.exp(tf.clip_by_value(self.T_log, tf.math.log(1.0), tf.math.log(500.0)))
        R_s = tf.exp(self.R_log) + 5.0
        C_s = tf.exp(tf.clip_by_value(self.C_log, -10.0, 2.0))
        Sy = tf.sigmoid(self.Sy_logit)
        h_real = hp * (self.h_max - self.h_min) + self.h_min
        q_real = tf.maximum(qp * (self.q_max - self.q_min) + self.q_min, 0.0)
        qin_real = qin_p * 500.0  # 設一個參考量級（CMH）
        
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
        # 💡 [修正質量平衡]：Area * Sy * ΔH = (Qin_dynamic - Total_Pump_Q) * ΔT
        mass_balance_error = (self.config["area_A"] * Sy * delta_h) - ((qin_real[:, 0] - tf.reduce_sum(q_real, axis=1)) * self.config["DELTA_T"])
        l_phys_temporal = tf.reduce_mean(tf.square(mass_balance_error))
        
        return tf.reduce_mean(tf.square(y_h - hp)), (l_phys_spatial * 0.1) + l_phys_temporal, tf.reduce_mean(tf.square(y_q - qp))

    def train_step(self, data):
        X, y = data
        y_h, y_q = y[:, :self.idx_map["h_end"]], y[:, self.idx_map["h_end"]:]
        self.curr_epoch.assign_add(1.0 / (self.total_samples / self.config["batch_size"]))
        is_warmup = self.curr_epoch < float(self.config["warmup_epochs"])
        l_phys_w = tf.cond(is_warmup, lambda: 0.0, lambda: tf.minimum(self.config["lambda_phys_final"] * ((self.curr_epoch - self.config["warmup_epochs"]) / 20.0), self.config["lambda_phys_final"]))
        with tf.GradientTape(persistent=True) as tape:
            hp, qp, qip = self(X, training=True)
            l_dat, l_phy, l_flo = self.calculate_losses(hp, qp, qip, y_h, y_q, X)
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

    # [數據切斷]：根據使用者需求，只訓練到 2021 年 6 月底
    if config.get("TRAIN_CUTOFF"):
        cutoff_date = pd.to_datetime(config["TRAIN_CUTOFF"])
        df_raw = df_raw.loc[df_raw.index < cutoff_date]
        print(f"✂️ 數據已根據 TRAIN_CUTOFF 切斷，目前數據上限為: {df_raw.index.max()}")

    df_master = df_raw.copy().ffill().bfill()
    print(f"📊 訓練資料（完整資料集）：{df_master.index.min()} → {df_master.index.max()}")

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

    # ==========================================
    # 🌟 3.4 [資料切分] 建立 Hold-out 盲測考卷
    # ==========================================
    test_start = pd.to_datetime(config["TEST_START"])
    test_end = pd.to_datetime(config["TEST_END"])
    
    # 切成三段，避免時序接軌產生跳躍
    df_train_part1 = df_master.loc[df_master.index < test_start]
    df_test        = df_master.loc[(df_master.index >= test_start) & (df_master.index < test_end)]
    df_train_part2 = df_master.loc[df_master.index >= test_end]

    print(f"\n✂️ 資料切分完畢：")
    print(f"  - 訓練集 Part1 : {df_train_part1.index.min()} → {df_train_part1.index.max()} (共 {len(df_train_part1)} 筆)")
    print(f"  - 訓練集 Part2 : {df_train_part2.index.min()} → {df_train_part2.index.max()} (共 {len(df_train_part2)} 筆)")
    print(f"  - 盲測集 (Test): {df_test.index.min()} → {df_test.index.max()} (共 {len(df_test)} 筆)")

    # 分別做成序列
    X_tr1, y_tr1 = create_seq(df_train_part1) if len(df_train_part1) > config["window_size"] else ([], [])
    X_tr2, y_tr2 = create_seq(df_train_part2) if len(df_train_part2) > config["window_size"] else ([], [])
    X_test, y_test = create_seq(df_test)

    # 合併訓練集
    if len(X_tr1) > 0 and len(X_tr2) > 0:
        X_train_full = np.concatenate([X_tr1, X_tr2])
        y_train_full = np.concatenate([y_tr1, y_tr2])
    elif len(X_tr1) > 0:
        X_train_full, y_train_full = X_tr1, y_tr1
    else:
        X_train_full, y_train_full = X_tr2, y_tr2

    # ==========================================
    # 🌟 3.5 5-Fold Cross Validation (可開關)
    # ==========================================
    if config.get("USE_KFOLD", False):
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5, shuffle=False)
        
        cv_epochs = 300
        print(f"\n🔄 啟動 5-Fold Cross Validation (針對訓練集，每折 {cv_epochs} epochs)...")
        fold = 1
        cv_scores = []
        
        for train_idx, val_idx in kf.split(X_train_full):
            print(f"\n--- Fold {fold}/5 ---")
            X_tr, y_tr = X_train_full[train_idx], y_train_full[train_idx]
            X_val, y_val = X_train_full[val_idx], y_train_full[val_idx]
            
            cv_model = PINN_Feedback_Model(dist_df.loc[obs_cols+wells_list, wells_list].values, config, idx_map, h_min, h_max, q_min, q_max, len(X_tr))
            cv_model.compile(); cv_model.build(input_shape=X_tr.shape)
            
            print("  正在訓練...", flush=True)
            cv_model.fit(X_tr, y_tr, epochs=cv_epochs, batch_size=config["batch_size"], verbose=1)
            
            hp_val_s, _, _ = cv_model.predict(X_val, verbose=0)
            hp_val = hp_val_s * (h_max - h_min) + h_min
            y_val_h = y_val[:, :idx_map["h_end"]] * (h_max - h_min) + h_min
            
            wape_score = calculate_wape(y_val_h, hp_val)
            print(f"完成！ Validation WAPE 誤差: {wape_score:.2f}%")
            cv_scores.append(wape_score)
            fold += 1
            
        print(f"\n📊 5-Fold CV 原本的訓練體驗證平均 WAPE: {np.mean(cv_scores):.2f}%")
    else:
        print(f"\n⏭️ 根據設定，已跳過 5-Fold Cross Validation 流程。")

    # ==========================================
    # 🌟 3.6 最終全訓練集正式訓練
    # ==========================================
    print(f"\n🚀 啟動最終訓練集正式訓練 (Epochs: {config['epochs']}) 產出實體參數...")
    model = PINN_Feedback_Model(dist_df.loc[obs_cols+wells_list, wells_list].values, config, idx_map, h_min, h_max, q_min, q_max, len(X_train_full))
    model.compile(); model.build(input_shape=X_train_full.shape)

    history = model.fit(X_train_full, y_train_full, epochs=config["epochs"], batch_size=config["batch_size"], verbose=1)

    # ==========================================
    # 🌟 3.7 [盲測驗證] 最終模型成績發表
    # ==========================================
    print(f"\n🏆 正在對 {config['TEST_START']} 至 {config['TEST_END']} 進行盲測評分...")
    hp_test_s, _, _ = model.predict(X_test, verbose=0)
    hp_test = hp_test_s * (h_max - h_min) + h_min
    y_test_h = y_test[:, :idx_map["h_end"]] * (h_max - h_min) + h_min
    
    test_wape = calculate_wape(y_test_h, hp_test)
    test_mae = mean_absolute_error(y_test_h, hp_test)
    print(f"==========================================")
    print(f"🏅 [盲測成績] 10月份整體預測準確率 (WAPE) : {test_wape:.2f}%")
    print(f"🏅 [盲測成績] 10月份整體預測平均誤差 (MAE) : {test_mae:.3f} m")
    print(f"==========================================\n")

    # ==========================================
    # 🌟 4. [物理提取] 包含 Qin 與 Sy 學習 (仍使用訓練集提取物理參數)
    # ==========================================
    print("\n🧹 正在進行物理診斷提取...")
    learned_sy = float(tf.sigmoid(model.Sy_logit).numpy()[0])
    hp_train_s, qp_train_s, _ = model.predict(X_train_full)
    hp_train = hp_train_s * (h_max - h_min) + h_min
    qp_train = qp_train_s * (q_max - q_min) + q_min
    ht_train = y_train_full[:, :idx_map["h_end"]] * (h_max - h_min) + h_min
    qt_train = y_train_full[:, idx_map["h_end"]:] * (q_max - q_min) + q_min

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

    # --- 自回歸預測（不看答案）---
    predict_start = pd.to_datetime(config["PREDICT_START"])
    predict_end = pd.to_datetime(config["PREDICT_END"])
    predict_steps = int((predict_end - predict_start).total_seconds() / (config["DELTA_T"] * 3600))
    print(f"\n🔮 自回歸預測：{config['PREDICT_START']} → {config['PREDICT_END']}（共 {predict_steps} 步，不看答案）...")

    # 取 PREDICT_START 前的 window_size 筆作為初始上下文
    context_data = df_master.loc[df_master.index < predict_start].tail(config["window_size"])
    assert len(context_data) >= config["window_size"], \
        f"❌ 上下文不足：PREDICT_START 前需要至少 {config['window_size']} 筆資料，僅有 {len(context_data)} 筆"

    curr_win = scaler.transform(context_data).reshape(1, config["window_size"], -1)

    # ✅ 初始條件錨定：將 context window 最後一步的水位欄位替換為 PREDICT_START 的真實觀測值
    # 這確保自回歸預測從「真實已知的初始水位」出發，而不是讓模型猜起始點
    actual_start_row = df_master.loc[df_master.index <= predict_start].iloc[-1]
    actual_start_h_raw = actual_start_row[obs_cols].values.astype(float)
    # 用與訓練相同的 scaler 將真實水位歸一化後寫入 context window 最後一步
    full_start_row = actual_start_row.values.astype(float).reshape(1, -1)
    full_start_scaled = scaler.transform(full_start_row)[0]
    curr_win[0, -1, :idx_map["n_obs"]] = full_start_scaled[:idx_map["n_obs"]]
    print(f"  📍 初始條件錨定完成：PREDICT_START 實際水位 = {np.round(actual_start_h_raw, 2)}")

    future_h, future_qin = [], []
    for _ in range(predict_steps):
        hp_s, _, qip_s = model(curr_win, training=False)
        future_h.append(hp_s.numpy()[0, :idx_map["n_obs"]] * (h_max[:idx_map["n_obs"]] - h_min[:idx_map["n_obs"]]) + h_min[:idx_map["n_obs"]])
        future_qin.append(qip_s.numpy()[0, 0] * 500.0)
        
        new_step = curr_win[:, -1:, :].copy()
        new_step[0, 0, :idx_map["n_obs"]] = hp_s.numpy()[0, :idx_map["n_obs"]]
        new_step[0, 0, idx_map["h_end"]:] = 0
        curr_win = np.append(curr_win[:, 1:, :], new_step, axis=1)
    future_h, future_qin = np.array(future_h), np.array(future_qin)
    np.save(f"{save_p}/background_h_7d.npy", future_h)
    np.save(f"{save_p}/qin_7d_dynamic.npy", future_qin)

    # --- 新增：反向基準推導專用 --- 產生含現場真實抽水的極度準確預測水位
    print("\n🎯 [混合重疊原理] 正在產生含電力特徵的神準預測水位（Reverse Superposition Baseline）...")
    # 擷取包含 Context Window + 欲預測區間 的真實資料
    delta_hours = config["window_size"] * config["DELTA_T"]
    actual_predict_data = df_master.loc[(df_master.index >= (predict_start - pd.Timedelta(hours=delta_hours))) & (df_master.index < predict_end)]
    if len(actual_predict_data) > config["window_size"]:
        X_acc, _ = create_seq(actual_predict_data)
        # 確保輸出的長度剛好等於 predict_steps
        X_acc = X_acc[:predict_steps]
        hp_acc_s, _, _ = model.predict(X_acc, verbose=0)
        accurate_h = hp_acc_s[:, :idx_map["n_obs"]] * (h_max[:idx_map["n_obs"]] - h_min[:idx_map["n_obs"]]) + h_min[:idx_map["n_obs"]]
        np.save(f"{save_p}/accurate_pred_h_7d.npy", accurate_h)
        print(f"  📥 已儲存：accurate_pred_h_7d.npy (用於最佳化反向推導)")
    else:
        print(f"  ⚠️ 無法產生 accurate_pred_h_7d.npy，真實資料量不足以支撐 {predict_steps} 步预测。")
    # ----------------------------------------------------

    # --- 預測 vs 實際比對 ---
    actual_test_df = df_master.loc[predict_start:predict_end][obs_cols]
    if len(actual_test_df) > predict_steps:
        actual_test_df = actual_test_df.iloc[:predict_steps]
    actual_h_test = actual_test_df.values if len(actual_test_df) > 0 else None

    if actual_h_test is not None and len(actual_h_test) > 0:
        n_compare = min(len(future_h), len(actual_h_test))
        print(f"\n📊 [背景水位(抽水=0)] vs 實際對比（共 {n_compare} 步 = {n_compare * config['DELTA_T']:.0f} 小時）：")
        print(f"{'觀測井':<12} | {'MAE (m)':<10} | {'RMSE (m)':<10}")
        print("-" * 40)
        for i, col in enumerate(obs_cols):
            mae_val = mean_absolute_error(actual_h_test[:n_compare, i], future_h[:n_compare, i])
            rmse_val = np.sqrt(np.mean((actual_h_test[:n_compare, i] - future_h[:n_compare, i])**2))
            print(f"{col:<12} | {mae_val:<10.4f} | {rmse_val:<10.4f}")
        overall_mae = mean_absolute_error(actual_h_test[:n_compare].flatten(), future_h[:n_compare].flatten())
        print(f"{'整體':<12} | {overall_mae:<10.4f}")
        
        # 額外印出神準水位的比對
        if 'accurate_h' in locals() and len(accurate_h) >= n_compare:
            print(f"\n🎯 [神準預測水位(含真實抽水)] vs 實際對比：")
            acc_overall_mae = mean_absolute_error(actual_h_test[:n_compare].flatten(), accurate_h[:n_compare].flatten())
            print(f"{'整體 MAE':<12} | {acc_overall_mae:<10.4f}  <-- 最佳化將以這條線為基準進行反向推導！")
    else:
        actual_h_test = None
        print("⚠️ 預測區間內無實際資料可比對")

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
    # 重新準備 X_reg, Y_reg 以繪製散佈圖 (Area*dH/dt vs Q_pump)
    try:
        X_reg_plot = (config["area_A"] * dH_dt_pinn[valid_idx])
        Y_reg_plot = total_Q_train[valid_idx]
        plt.figure(figsize=(10, 6)); plt.scatter(X_reg_plot, Y_reg_plot, alpha=0.3, color='gray', label='Cleaned Data (Actual)')
        x_r = np.array([np.min(X_reg_plot), np.max(X_reg_plot)])
        plt.plot(x_r, inflow_pinn_mean - (learned_sy * x_r), color='red', linewidth=3, label=f'PINN Physics Line (Sy={learned_sy:.3f}, R2={r2_pinn_val:.3f})')
        plt.xlabel("Area * dH/dt (m3/hr)"); plt.ylabel("Total Pumping Q (m3/hr)"); plt.title("Physical Mass Balance"); plt.legend(); plt.grid(True, alpha=0.3); plt.savefig(f"{save_p}/8_PINN_Physics_Line.png"); plt.close()
    except Exception as e:
        print(f"⚠️ 物理直線圖繪製失敗: {e}")

    # 5. Qin 時間序列 (9_Qin_TimeSeries.png)
    plt.figure(figsize=(12, 5)); plt.plot(qin_series, color='steelblue', alpha=0.4, label='Qin(t) Train'); plt.plot(qin_smooth, color='darkorange', linewidth=2, label='Smoothed'); 
    plt.plot(np.arange(len(future_qin)) + len(qin_series), future_qin, color='red', linewidth=2, linestyle='--', label='7D Dynamic Pred')
    plt.axhline(inflow_pinn_mean, color='gray', linestyle=':', label=f'Mean Qin={inflow_pinn_mean:.1f}'); plt.title("Derived & Predicted Dynamic Inflow TimeSeries"); plt.legend(); plt.grid(True, alpha=0.3); plt.savefig(f"{save_p}/9_Qin_TimeSeries.png"); plt.close()

    # 6. 預測 vs 實際 對比圖 (Prediction_vs_Actual.png)
    n_obs_plot = len(obs_cols)
    rows_test = int(np.ceil(n_obs_plot / 3))
    fig_test, axes_test = plt.subplots(rows_test, 3, figsize=(18, 4 * rows_test))
    axes_test = axes_test.flatten()
    time_hours = np.arange(len(future_h)) * config["DELTA_T"]
    # 計算全域 Y 軸範圍，確保所有小圖比例一致
    all_vals = []
    if 'accurate_h' in locals() and len(accurate_h) > 0: all_vals.append(accurate_h)
    else: all_vals.append(future_h)
    if actual_h_test is not None: all_vals.append(actual_h_test)
    y_min_total = np.min([np.min(v) for v in all_vals]) - 0.5
    y_max_total = np.max([np.max(v) for v in all_vals]) + 0.5

    for i, col in enumerate(obs_cols):
        # 繪製神準預測水位（包含實際抽水特徵）
        if 'accurate_h' in locals() and len(accurate_h) > 0:
             pred_to_plot = accurate_h
             label_name = 'Predicted (w/ Actual Pumps)'
        else:
             pred_to_plot = future_h
             label_name = 'Predicted (Pure Background)'
             
        axes_test[i].plot(time_hours[:len(pred_to_plot)], pred_to_plot[:, i], color='darkorange', linewidth=2, label=label_name)
        if actual_h_test is not None:
            n_c = min(len(pred_to_plot), len(actual_h_test))
            axes_test[i].plot(time_hours[:n_c], actual_h_test[:n_c, i], color='steelblue', alpha=0.7, label='Actual')
            mae_i = mean_absolute_error(actual_h_test[:n_c, i], pred_to_plot[:n_c, i])
            axes_test[i].set_title(f"{col} (MAE={mae_i:.3f}m)")
        else:
            axes_test[i].set_title(col)
        axes_test[i].set_xlabel("Hours")
        axes_test[i].set_ylabel("Level (m)")
        axes_test[i].set_ylim(y_min_total, y_max_total) # 固定 Y 軸比例
        axes_test[i].legend()
        axes_test[i].grid(True, alpha=0.3)
    for j in range(n_obs_plot, len(axes_test)):
        fig_test.delaxes(axes_test[j])
    fig_test.suptitle(f"Prediction vs Actual: {config['PREDICT_START']} ~ {config['PREDICT_END']}", fontsize=14)
    fig_test.tight_layout()
    fig_test.savefig(f"{save_p}/Prediction_vs_Actual.png", bbox_inches='tight')
    plt.close()

    # 7. 背景水位回升趨勢圖 (background_h_7d_trend.png)
    print("📈 正在繪製 7 天背景水位回升趨勢圖...")
    plt.figure(figsize=(12, 6))
    time_h = np.arange(len(future_h)) * config["DELTA_T"]
    colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, len(obs_cols)))
    for i, col in enumerate(obs_cols):
        plt.plot(time_h, future_h[:, i], color=colors[i], label=col, alpha=0.8)
    
    plt.axhline(y=-11.0, color='red', linestyle='--', linewidth=2, label='Static SWL (-11m)')
    plt.title(f"7-Day Background Water Level Recovery (Predicted by AI)\n{config['PREDICT_START']} ~ {config['PREDICT_END']}")
    plt.xlabel("Hours (Autoregressive Steps)")
    plt.ylabel("Water Level (m)")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1, fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_p}/background_h_7d_trend.png", bbox_inches='tight')
    plt.close()

    print(f"✅ 任務完成！報告與圖表已儲存於 {save_p}")

    print(f"✅ 任務完成！報告與圖表已儲存於 {save_p}")