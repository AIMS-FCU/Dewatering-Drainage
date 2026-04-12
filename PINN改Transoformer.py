import pandas as pd
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
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
    # 分母改用總流量的和，避免單點為 0 的問題
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
    "epochs": 3,            # 測試建議設小，正式建議 300+
    "batch_size": 512,
    "lambda_phys_final": 2.0, 
    "lambda_flow": 2.0,       
    "warmup_epochs": 150,      
    "save_folder": "PINN_MAPE_Complete_Report3" ,
    "area_A": 3319.95,
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
            TransformerEncoderBlock(
                config["d_model"],
                config["num_heads"],
                config["ff_dim"],
                config["dropout"],
            )
            for _ in range(config["num_transformer_layers"])
        ]
        self.sequence_pool = layers.GlobalAveragePooling1D()
        self.instant_dense = layers.Dense(32, activation='swish')
        self.head_h = tf.keras.Sequential([layers.Dense(128, activation='swish'), layers.Dense(64, activation='swish'), layers.Dense(idx_map["h_end"])])
        self.head_q = tf.keras.Sequential([
            layers.Dense(64, activation='relu'), 
            layers.Dense(idx_map["n_wells"], activation='relu') 
        ])
        
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
        # --- A. 提取物理參數 ---
        T_s = tf.exp(tf.clip_by_value(self.T_log, tf.math.log(1.0), tf.math.log(500.0)))
        R_s = tf.exp(self.R_log) + 5.0
        C_s = tf.exp(tf.clip_by_value(self.C_log, -10.0, 2.0))
        
        # 🌟 這裡讓 Sy 參與運算：使用 Sigmoid 限制在 0~1 之間
        Sy = tf.sigmoid(self.Sy_logit)
        
        # --- B. 反標準化為物理量 ---
        h_real = hp * (self.h_max - self.h_min) + self.h_min
        q_real_raw = qp * (self.q_max - self.q_min) + self.q_min
        q_real = tf.maximum(q_real_raw, 0.0) # 強制非負
        
        # --- C. 空間物理損失 (Thiem 公式: 穩態降深) ---
        l_phys_spatial = 0.0
        n_o, n_w = self.idx_map["n_obs"], self.idx_map["n_wells"]
        for i in range(n_o + n_w):
            s_formation = 0.0
            for j in range(n_w):
                r = tf.where(self.dist_matrix[i, j] <= 0, self.config["rw"], self.dist_matrix[i, j])
                s_formation += (q_real[:, j] / (2.0 * 3.14159 * T_s[i] + 1e-4)) * tf.math.log(R_s[i] / (r + 1e-5))
            
            s_well_loss = 0.0
            if i >= n_o:
                well_idx = i - n_o
                s_well_loss = C_s[well_idx] * tf.square(q_real[:, well_idx])
            
            s_theo_total = s_formation + s_well_loss
            s_pred_actual = tf.maximum(self.h_max[i] - h_real[:, i], 0.0)
            l_phys_spatial += tf.reduce_mean(tf.abs(s_pred_actual - s_theo_total))

        # --- D. 🌟 關鍵新增：時間物理損失 (質量守恆: 學習 Sy) ---
        # 1. 取得前一個時段的水位 (從輸入 X 的最後一格提取)
        h_prev_std = X[:, -1, :self.idx_map["n_obs"]]
        h_prev = h_prev_std * (self.h_max[:n_o] - self.h_min[:n_o]) + self.h_min[:n_o]
        
        # 2. 計算觀測井平均水位的變化 (Delta_H)
        avg_h_now = tf.reduce_mean(h_real[:, :n_o], axis=1)
        avg_h_prev = tf.reduce_mean(h_prev, axis=1)
        delta_h = avg_h_now - avg_h_prev
        
        # 3. 計算總抽水量 (Q_out)
        total_q_out = tf.reduce_sum(q_real, axis=1)
        
        # 4. 質量守恆公式: (Qin - Qout) * dt = Area * Sy * Delta_H
        # 這裡假設 Qin 是一個需要被滿足的常數(或基準)，我們計算 Delta_H 是否符合抽水引起的變化
        # 為了簡化，我們約束：Area * Sy * Delta_H + Qout * dt 應該要等於進水量補給
        # 因為我們不知道精確的 Qin，我們讓模型去最小化這個變化的不穩定性
        dt = self.config["DELTA_T"] if "DELTA_T" in self.config else 0.5
        area = self.config["area_A"]
        
        # 物理公式項
        mass_balance_error = (area * Sy * delta_h) + (total_q_out * dt)
        
        # 我們讓這個項逼近之前回歸得到的平均 Qin，或者讓它在 batch 內保持穩定
        # 這裡採取最直接的做法：讓它逼近一個合理的進水範圍或減少其波動
        l_phys_temporal = tf.reduce_mean(tf.square(mass_balance_error - tf.reduce_mean(mass_balance_error)))

        # --- E. 綜合損失 ---
        # 空間損失(學T,C) + 時間損失(學Sy)
        total_phys_loss = (l_phys_spatial * 0.1) + (l_phys_temporal * 1)
            
        return tf.reduce_mean(tf.square(y_h - hp)), total_phys_loss, tf.reduce_mean(tf.square(y_q - qp))

    def train_step(self, data):
        X, y = data
        y_h = y[:, :self.idx_map["h_end"]]
        y_q = y[:, self.idx_map["h_end"]:]
        
        self.curr_epoch.assign_add(1.0 / (self.total_samples / self.config["batch_size"]))
        is_warmup = self.curr_epoch < float(self.config["warmup_epochs"])
        
        l_phys_w = tf.cond(
            is_warmup, 
            lambda: 0.0, 
            lambda: tf.minimum(
                self.config["lambda_phys_final"] * ((self.curr_epoch - self.config["warmup_epochs"]) / 20.0), 
                self.config["lambda_phys_final"]
            )
        )

        with tf.GradientTape(persistent=True) as tape:
            hp, qp = self(X, training=True)
            # 🌟 修改點：將 X 傳入 calculate_losses
            l_dat, l_phy, l_flo = self.calculate_losses(hp, qp, y_h, y_q, X)
            
            l_phy_normalized = l_phy / 500.0 
            total_loss = (l_dat * 200.0) + (l_phys_w * l_phy_normalized) + (self.config["lambda_flow"] * l_flo)

        nn_vars = self.trainable_variables
        self.opt_nn.apply_gradients(zip(tape.gradient(total_loss, nn_vars), nn_vars))
        
        def apply_phys():
            p_vars = [self.T_log, self.R_log, self.C_log, self.Sy_logit]
            self.opt_phys.apply_gradients(zip(tape.gradient(total_loss, p_vars), p_vars))
            return tf.constant(1.0)
        
        tf.cond(is_warmup, lambda: tf.constant(0.0), apply_phys)
        
        return {"loss": total_loss, "l_dat": l_dat, "l_phy": l_phy, "l_flo": l_flo, "phys_w": l_phys_w}

# ==========================================
# 3. 數據處理與執行訓練
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(config["save_folder"]): os.makedirs(config["save_folder"])

    print("📊 數據處理中...")
    df_raw = pd.read_csv('Master_Training_Data_Continuous3.csv', index_col=0)
    df_raw.index = pd.to_datetime(df_raw.index)
    dist_df = pd.read_csv('Distance_Matrix.csv', index_col=0)

    obs_cols = [c for c in ['PA', 'PB', 'PC', 'FPS7', 'FPS8', 'FPS9', 'FPS2', 'FPS3', 'FPS4', 'FPS5', 'FPS6'] if c in df_raw.columns]
    potential_wells = ["PW01", "PW02", "PW03", "PW04", "PW06", "PW07", "PW08", "PW09", "PW010", "PW011", "PW012", "PW013"]
    wells_list = [w for w in potential_wells if w in df_raw.columns and df_raw[w].abs().max() > 1e-6]
    flow_cols = [f"Qw{map_sensor_id(w)}" for w in wells_list if f"Qw{map_sensor_id(w)}" in df_raw.columns]
    
    needed_cols = obs_cols + wells_list + [f"P_gr_{map_sensor_id(w)}" for w in wells_list if f"P_gr_{map_sensor_id(w)}" in df_raw.columns] + ['ERN/hr1'] + flow_cols
    # 🌟 強制固定時間範圍，與最佳化模型一致
    analysis_start = "2020-09-11 11:30"
    analysis_end   = "2021-06-01 00:00" # 👈 必須跟最佳化起點完全一致
    
    # 只保留到 4/16 之前的資料
    df_master = df_raw.loc[analysis_start : analysis_end].copy().ffill().bfill()
    
    print(f"📊 數據範圍已鎖定：{df_master.index.min()} 到 {df_master.index.max()}")

    test_start = pd.to_datetime('2021-04-01')
    test_end   = pd.to_datetime('2021-06-01')
    test_mask = (df_master.index >= test_start) & (df_master.index <= test_end)
    train_df, test_df = df_master.loc[~test_mask].copy(), df_master.loc[test_mask].copy()

    idx_map = {"n_obs": len(obs_cols), "n_wells": len(wells_list), "h_end": len(obs_cols)+len(wells_list), "flow_start": len(df_master.columns) - len(flow_cols), "pwr_start": len(obs_cols)+len(wells_list)}
    h_min, h_max = df_master.iloc[:, :idx_map["h_end"]].min().values, df_master.iloc[:, :idx_map["h_end"]].max().values + 1e-7
    q_min, q_max = df_master[flow_cols].min().values, df_master[flow_cols].max().values + 1e-7
    scaler = MinMaxScaler().fit(df_master)

    def create_seq(df):
        data_s = scaler.transform(df)
        X, y = [], []
        if len(data_s) <= config["window_size"]: return np.zeros((0, config["window_size"], data_s.shape[1])), np.zeros((0,0))
        for i in range(len(data_s) - config["window_size"]):
            X.append(data_s[i:i+config["window_size"]])
            y.append(np.concatenate([data_s[i+config["window_size"], :idx_map["h_end"]], data_s[i+config["window_size"], idx_map["flow_start"]:]]))
        return np.array(X), np.array(y)

    X_train, y_train = create_seq(train_df)
    X_test, y_test = create_seq(test_df)

    model = PINN_Feedback_Model(dist_df.loc[obs_cols+wells_list, wells_list].values, config, idx_map, h_min, h_max, q_min, q_max, len(X_train))
    model.compile()

    # ==========================================
    # 🏗️ 強化版模型架構視覺化 (手動追蹤版)
    # ==========================================
    print("\n🏗️ 正在生成完整架構流程圖...")

    def plot_enhanced_pinn_structure(model, config, idx_map, save_path):
        # 手動定義架構節點，確保順序與邏輯正確
        nodes = [
            {"name": "Input Time Series", "info": f"Shape: (Batch, {config['window_size']}, Features)"},
            {"name": "Input Projection (Dense)", "info": f"Dim: {config['d_model']}"},
            {"name": "Positional Encoding", "info": "Add Time Context"},
            {"name": "Transformer Block 1", "info": f"Heads: {config['num_heads']}"},
            {"name": "Transformer Block 2", "info": f"Heads: {config['num_heads']}"},
            {"name": "Global Avg Pooling", "info": "Sequence -> Vector"},
            {"name": "Concatenate Layer", "info": "Combine: [Trans_Out, Pump_Feature, Physics_C]"},
            {"name": "Head_H (Sequential)", "info": f"Predict Level: {idx_map['h_end']} points"},
            {"name": "Head_Q (Sequential)", "info": f"Predict Flow: {idx_map['n_wells']} wells"},
            {"name": "PINN Loss Function", "info": "Physics Constraints (T, R, C, Sy)"}
        ]

        plt.figure(figsize=(10, 12))
        plt.axis('off')
        
        n = len(nodes)
        for i, node in enumerate(nodes):
            y = n - i
            # 繪製方框
            color = 'lightgreen' if 'Loss' in node['name'] else 'aliceblue'
            if 'Head' in node['name']: color = 'lightyellow'
            
            plt.text(0.5, y, f"{node['name']}\n{node['info']}", 
                    ha='center', va='center', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.8', facecolor=color, edgecolor='steelblue'))
            
            # 繪製箭頭 (分叉處理)
            if i < n - 3: # 一般線性流向
                plt.annotate('', xy=(0.5, y - 0.4), xytext=(0.5, y - 0.6),
                            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
            elif i == 6: # Concatenate 分叉到兩個 Head
                plt.annotate('', xy=(0.3, y - 0.6), xytext=(0.5, y - 0.4),
                            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
                plt.annotate('', xy=(0.7, y - 0.6), xytext=(0.5, y - 0.4),
                            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
            elif i >= 7 and i < 9: # Head 到 Loss
                target_y = n - 9 + 0.4
                plt.annotate('', xy=(0.5, target_y), xytext=(0.5 if i==7 else 0.5, y - 0.4),
                            arrowprops=dict(arrowstyle='->', color='red', linestyle='--', lw=1))

        plt.title(f"PINN Model Architecture\n(Total Params: {model.count_params():,})", fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"✅ 強化版架構圖已儲存至: {save_path}")

    # 呼叫函數
    plot_enhanced_pinn_structure(model, config, idx_map, f"{config['save_folder']}/pinn_full_architecture.png")

    # ==========================================
    # 💡 4. [新增] 提取並儲存物理參數供 Optimization.py 使用
    # ==========================================
    print("\n💾 正在提取物理參數並儲存檔案...")
    save_p = config["save_folder"]
    
    # 提取 T (場址平均) 與 C (13口井陣列)
    learned_T_single = np.mean(tf.exp(model.T_log).numpy())
    learned_C_array = tf.exp(model.C_log).numpy()
    
    np.save(f"{save_p}/learned_T.npy", learned_T_single)
    np.save(f"{save_p}/learned_C.npy", learned_C_array)
    
    with open(f"{save_p}/physics_summary.txt", "w") as f:
        f.write(f"Learned site-wide T (mean): {learned_T_single:.6f}\n")
        f.write(f"Learned Well Loss C (13 wells):\n{str(learned_C_array)}")
    print(f"✅ 物理參數已儲存：T={learned_T_single:.4f}")

    # ==========================================
    # 5. 產出報告與自動診斷 (包含水位與流量誤差分析)
    # ==========================================
    print("\n📊 正在生成完整診斷報告與圖表...")
    hp_s, qp_s = model.predict(X_test)
    hp = hp_s * (h_max - h_min) + h_min

    qp_raw = qp_s * (q_max - q_min) + q_min
    qp = qp_s * (q_max - q_min) + q_min
    ht = y_test[:, :idx_map["h_end"]] * (h_max - h_min) + h_min
    qt = y_test[:, idx_map["h_end"]:] * (q_max - q_min) + q_min
    T_res, C_res = tf.exp(model.T_log).numpy(), tf.exp(model.C_log).numpy()
    point_names = obs_cols + wells_list

    diag_data = []
    for i, name in enumerate(point_names):
        # --- A. 水位誤差 (Level MAE/MAPE) ---
        h_mae = mean_absolute_error(ht[:, i], hp[:, i])
        h_mape = calculate_wape(ht[:, i], hp[:, i])
        
        is_pump = 1 if name in wells_list else 0
        q_mae, q_mape = 0.0, 0.0 # 預設值
        eff, diagnosis = 100.0, "觀測井：監測中"
        
        # --- B. 流量誤差與效率診斷 (僅針對抽水井) ---
        if is_pump:
            well_idx = wells_list.index(name)
            # 計算該井的流量 MAE 與 MAPE
            q_mae = mean_absolute_error(qt[:, well_idx], qp[:, well_idx])
            q_mape = calculate_wape(qt[:, well_idx], qp[:, well_idx])
            
            c_val = C_res[well_idx]
            avg_q = np.mean(qp[:, well_idx])
            # 計算淤塞效率
            s_form = (avg_q / (2 * 3.14159 * T_res[i] + 1e-4)) * np.log(100.0 / 0.45)
            s_well = c_val * (avg_q ** 2)
            eff = (s_form / (s_form + s_well + 1e-6)) * 100
            
            if eff < 70: diagnosis = "🔴 嚴重淤塞"
            elif eff < 85: diagnosis = "🟡 輕微淤塞"
            else: diagnosis = "🔵 狀態良好"
            
        diag_data.append({
            "Well": name, 
            "Type": "Pump" if is_pump else "Obs", 
            "MAE_Level(m)": h_mae, 
            "MAPE_Level(%)": h_mape,
            "MAE_Flow(m3/hr)": q_mae if is_pump else np.nan,  # 新增
            "MAPE_Flow(%)": q_mape if is_pump else np.nan,    # 新增
            "Efficiency(%)": eff, 
            "Status_Diagnosis": diagnosis
        })

    df_final = pd.DataFrame(diag_data)
    df_final.to_csv(f"{save_p}/Full_Diagnostic_Report.csv", index=False, encoding='utf-8-sig')
    
    # 打印報告時，隱藏觀測井無意義的流量欄位
    print(df_final.to_string(index=False, na_rep='-'))

    def plot_loss_final(data, title, fname, color):
        plt.figure(figsize=(9, 5)); ax = plt.gca(); plt.plot(data, color=color, linewidth=2)
        plt.yscale('log'); ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
        plt.axvline(x=config["warmup_epochs"], color='gray', linestyle=':'); plt.title(title); plt.tight_layout(); plt.savefig(f"{save_p}/{fname}", dpi=150); plt.close()

    # 圖表 1-6 繪製
    rows_h = int(np.ceil(len(point_names)/3))
    fig1, axes1 = plt.subplots(rows_h, 3, figsize=(18, 4*rows_h))
    axes1 = axes1.flatten()
    for i in range(len(point_names)):
        axes1[i].plot(ht[:, i], color='steelblue'); axes1[i].plot(hp[:, i], color='orange', linestyle='--')
        axes1[i].set_title(f"{point_names[i]}\nMAE: {df_final.iloc[i]['MAE_Level(m)']:.3f}m"); axes1[i].grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{save_p}/1_Water_Level.png", dpi=150); plt.close()

    if len(wells_list) > 0:
        rows_q = int(np.ceil(len(wells_list)/3))
        fig2, axes2 = plt.subplots(rows_q, 3, figsize=(18, 4*rows_q))
        axes2 = axes2.flatten()
        for i in range(len(wells_list)):
            axes2[i].plot(qt[:, i], color='green'); axes2[i].plot(qp[:, i], color='red', linestyle='--')
            axes2[i].set_title(f"Flow: {wells_list[i]}"); axes2[i].grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(f"{save_p}/2_Flow_Rate.png", dpi=150); plt.close()

        plt.figure(figsize=(12, 6)); plt.bar(df_final[df_final['Type']=="Pump"]['Well'], df_final[df_final['Type']=="Pump"]['Efficiency(%)'], color='teal')
        plt.axhline(y=70, color='red', linestyle='--'); plt.title("Efficiency (%)"); plt.savefig(f"{save_p}/3_Efficiency_Chart.png", dpi=150); plt.close()

    plot_loss_final(history.history['l_dat'], "Data Loss", "4_Data_Loss.png", "blue")
    plot_loss_final(history.history['l_phy'], "Physics Loss", "5_Phys_Loss.png", "red")
    plot_loss_final(history.history['loss'], "Total Loss", "6_Total_Loss.png", "black")

    # ==========================================
    # 💡 6. [新增] 預測未來 168 小時背景水位 (Q=0)
    # ==========================================
    
    print("\n🔮 正在預測未來 168 小時背景水位...")
    last_window = scaler.transform(df_master.tail(config["window_size"]))
    current_window = last_window.reshape(1, config["window_size"], -1)
    future_h_preds = []

    for t in range(168):
        hp_s, qp_s = model(current_window, training=False)
        hp_real = hp_s.numpy() * (h_max - h_min) + h_min
        future_h_preds.append(hp_real[0][:idx_map["n_obs"]])
        
        new_step = current_window[:, -1:, :].copy()
        new_step[0, 0, :idx_map["n_obs"]] = hp_s.numpy()[0, :idx_map["n_obs"]]
        new_step[0, 0, idx_map["pwr_start"]:idx_map["pwr_start"]+13] = 0 
        new_step[0, 0, idx_map["flow_start"]:] = 0
        current_window = np.append(current_window[:, 1:, :], new_step, axis=1)

    np.save(f"{save_p}/background_h_7d.npy", np.array(future_h_preds))
    print(f"✅ 任務完成！背景預測已儲存至 {save_p}/background_h_7d.npy")

    # ==========================================
    # 💡 8. [修復版] 利用 PINN 學習成果進行物理診斷與圖表繪製
    # ==========================================
    print("\n🧹 正在利用 PINN 學習成果進行物理關係提取...")

    # 1. 取得 PINN 學習到的 Sy
    learned_sy = float(tf.sigmoid(model.Sy_logit).numpy()[0])
    
    # 2. 預測並反標準化水位
    hp_train_s, _ = model.predict(X_train)
    hp_train = hp_train_s * (h_max - h_min) + h_min

    # 3. 計算觀測點平均水位與變化率 dH/dt
    avg_h_pinn = np.mean(hp_train[:, :idx_map["n_obs"]], axis=1)
    dt = config.get("DELTA_T", 0.5)
    dH_dt_pinn = np.zeros_like(avg_h_pinn)
    # 中央差分 (m/hr)
    dH_dt_pinn[1:-1] = (avg_h_pinn[2:] - avg_h_pinn[:-2]) / (2.0 * dt)

    # 4. 取得總抽水量 (Q_total)
    q_train_raw = y_train[:, idx_map["h_end"]:] * (q_max - q_min) + q_min
    total_Q_train = np.sum(q_train_raw, axis=1)

    # 5. 定義回歸繪圖所需的變數 (修復 NameError)
    valid_idx = slice(1, -1)
    # X 軸: Area * dH/dt (儲存量變化)
    X_reg = (config["area_A"] * dH_dt_pinn[valid_idx]).reshape(-1, 1)
    # Y 軸: 總抽水量
    Y_reg = total_Q_train[valid_idx].reshape(-1, 1)

    # 6. 推算 Qin 序列
    # Qin = Qout + Area * Sy * dH/dt
    qin_series = total_Q_train[valid_idx] + (config["area_A"] * learned_sy * dH_dt_pinn[valid_idx])
    
    # 7. 平滑處理
    smooth_window = min(24, len(qin_series))
    if smooth_window >= 3:
        kernel = np.ones(smooth_window) / smooth_window
        qin_smooth = np.convolve(qin_series, kernel, mode='same')
    else:
        qin_smooth = qin_series.copy()

    inflow_pinn_mean = float(np.mean(qin_smooth))
    inflow_pinn_median = float(np.median(qin_smooth))

    # 8. 計算 R-squared (驗證 Sy 是否合適)
    from sklearn.metrics import r2_score
    # 如果物理參數正確，Qout 應該要接近 Qin_mean - (Area * Sy * dH/dt)
    expected_Qout = inflow_pinn_mean - (config["area_A"] * learned_sy * dH_dt_pinn[valid_idx])
    r2_pinn_val = r2_score(total_Q_train[valid_idx], expected_Qout)

    print(f"✨ [PINN 物理參數學習報告]:")
    print(f"   - 學習到的 Sy (給水度): {learned_sy:.4f}")
    print(f"   - 動態質量平衡 R-squared: {r2_pinn_val:.4f}")
    print(f"   - 推估平均 Qin (進流量): {inflow_pinn_mean:.2f} m3/hr")
    print(f"   - 推估中位數 Qin: {inflow_pinn_median:.2f} m3/hr")

    # 儲存結果
    save_p = config["save_folder"]
    np.save(f"{save_p}/qin_series.npy", qin_series)
    np.save(f"{save_p}/qin_smooth.npy", qin_smooth)
    np.save(f"{save_p}/calibrated_inflow_sy.npy", np.array([inflow_pinn_mean, learned_sy]))

    # 6. 繪製物理直線圖
    # --- 繪圖 8: 物理直線圖 (修復後的直線繪製) ---
    plt.figure(figsize=(10, 6))
    plt.scatter(X_reg, Y_reg, alpha=0.3, color='gray', label='Cleaned Data (Actual)')
    
    # 繪製由 PINN 參數決定的物理斜線： Y = Qin - Sy * X
    # 我們取 X 的極值來畫一條直線
    x_range = np.array([np.min(X_reg), np.max(X_reg)])
    y_range = inflow_pinn_mean - (learned_sy * x_range)
    
    plt.plot(x_range, y_range, color='red', linewidth=3, 
             label=f'PINN Physics Line (Sy={learned_sy:.3f}, R2={r2_pinn_val:.3f})')
    
    plt.xlabel("Area * dH/dt (m3/hr)")
    plt.ylabel("Total Pumping Q (m3/hr)")
    plt.title("Physical Mass Balance: Storage Change vs. Pumping Rate")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{save_p}/8_PINN_Physics_Line.png", dpi=150)
    plt.show()

    # --- 繪圖 9: Qin 時間序列圖 ---
    plt.figure(figsize=(12, 5))
    plt.plot(qin_series, color='steelblue', alpha=0.45, label='Qin(t) Raw')
    plt.plot(qin_smooth, color='darkorange', linewidth=2, label='Smoothed Qin(t)')
    plt.axhline(inflow_pinn_mean, color='red', linestyle='--', label=f'Mean Qin={inflow_pinn_mean:.1f}')
    plt.xlabel("Time Step")
    plt.ylabel("Inflow (m3/hr)")
    plt.title("PINN-derived Dynamic Qin (Mass Balance Recovery)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_p}/9_Qin_TimeSeries.png", dpi=150)
    plt.show()