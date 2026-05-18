import pandas as pd
import numpy as np
import tensorflow as tf
import os
import sys
import warnings
import random

# 強制 Windows終端機使用 utf-8 輸出，避免 emoji 報錯
sys.stdout.reconfigure(encoding='utf-8')

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras import layers, Model
import joblib
import h5py
from sklearn.preprocessing import MinMaxScaler

# MICROTUNE_ENHANCED: make fine-tuning runs more reproducible.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception as e:
    print(f"MICROTUNE_ENHANCED: deterministic ops unavailable: {e}")

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

# 🌟 新增：開啟混合精度訓練 (Mixed Precision) 與 XLA 加速
# 這兩項技術可以大幅降低 VRAM 使用量，並在現代 GPU (如 RTX 系列) 上讓訓練速度提升 1.5 倍到 3 倍！
from tensorflow.keras import mixed_precision
ENABLE_XLA = False
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
tf.config.optimizer.set_jit(ENABLE_XLA) # XLA can fail on some GPU/mixed precision fusion kernels.

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ==========================================
# 0. 輔助工具
# ==========================================
def calculate_wape(y_true, y_pred, is_flow=False):
    # Weighted Absolute Percentage Error (巨觀體積水量的百分比誤差，不除以每一個時刻)
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    if is_flow:
        mask = y_true >= 1.0
        if np.sum(mask) == 0:
            return np.nan  # 該井沒有抽水，WAPE 無定義
        return np.sum(np.abs(y_true[mask] - y_pred[mask])) / (np.sum(np.abs(y_true[mask])) + 1e-10) * 100.0
    else:
        return np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + 1e-10) * 100.0

def calculate_mape(y_true, y_pred, is_flow=False):
    # Mean Absolute Percentage Error (正統 MAPE：精算每一個時刻的百分比殘差取平均)
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    if is_flow:
        # 流量專用：忽略停機時段 (只計算真實抽水 >= 1.0 的點，否則會發生除以 0 的無限大錯誤)
        mask = np.abs(y_true) >= 1.0
        if np.sum(mask) == 0:
            return 0.0 if np.mean(y_pred) < 5.0 else 100.0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0
    else:
        # 水位 MAPE
        return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-10))) * 100.0

# ==========================================
# 1. 超參數配置
# ==========================================
config = {
    "window_size": 336,       # 24*7*2 如果要改14天請另外註解 2天 7天測試
    "d_model": 64,
    "num_heads": 4,
    "ff_dim": 128,
    "num_transformer_layers": 4,
    "dropout": 0.1,
    "nn_lr_init": 0.0001,      # 🌟 遷移學習微調時，建議降低學習率 (原為 0.0005)
    "phys_lr": 0.0001,        
    "decay_steps": 1000,      
    "decay_rate": 0.9,        
    "clip_norm": 1.0,         
    "T_init": 30.0,          
    "R_init": 100.0,         
    "C_init": 0.01,           
    "rw": 0.45,               
    "epochs": 300,          # 與 PINN4層線性336.py 對齊
    "flow_epochs": 100,
    "batch_size": 256,
    "lambda_phys_final": 2.0, 
    "lambda_flow": 150.0,      # 🌟 與一期模型同步提高，平衡流量預測與水位預測的權重
    "warmup_epochs": 150,
    "save_folder": "PINN_Phase2_Report",
    "area_A": 5232,
    "DELTA_T": 0.5,
    "PREDICT_START": "2026-04-17 00:00",  # 指定預測起始時間供最佳化使用 (4/17~4/24 共 7 天)
    "PREDICT_END":   "2026-04-24 00:00",  # 指定預測結束時間供最佳化使用
    "TEST_START":    "2026-03-24 00:00",  # 盲測考卷起始時間 (3/24)
    "TEST_END":      "2026-04-24 00:00",  # 盲測考卷結束時間 (4/24)
    "TRAIN_CUTOFF":  "2026-03-01 00:00",  # [新功能] 數據切斷點，忽略此日期後的低抽水資料 (訓練只看2月底前)
    "USE_KFOLD":     False,               # 是否啟用 5-Fold Cross Validation
    "FINETUNE_START": None,
    "FINETUNE_END":   None,
    "HEAD_WARMUP_EPOCHS": 30,
    "EARLY_STOPPING_PATIENCE": 40,
    "SHUFFLE_TRAINING": False,
    "START_LEVEL_CORRECTION": True,
    "START_LEVEL_CORRECTION_MAX_ABS": None,
    "training_data_path": "Phase2_Training_Data2.csv",
    "allow_partial_transfer": True,
    
    # 🌟 遷移學習設定 (Transfer Learning)
    # 請將此路徑指向一期模型訓練出來的權重檔 (.h5)
    "phase1_weights_path": "PINN_MAPE_Complete_Report3_Run1_56/pinn_model.weights.h5",
}

def map_sensor_id(w_name):
    num = "".join(filter(str.isdigit, str(w_name))) 
    return num.zfill(2)


def is_probably_compatible_weights(weights_path):
    """更寬鬆且聰明的檢查，確保能載入不同環境產出的雙模型權重。"""
    try:
        import h5py
        with h5py.File(weights_path, "r") as f:
            names = []
            f.visit(names.append)
        
        # 只要有這幾個核心標記，就代表是我們定義的雙模型架構
        required_markers = [
            "head_h_1",
            "head_q_1",
            "head_inflow_1"
        ]
        
        missing = []
        for marker in required_markers:
            if not any(marker.lower() in name.lower() for name in names):
                missing.append(marker)
        
        if not missing:
            print(f"✅ [遷移學習] 偵測到相容的模型架構，準備載入...")
            return True
        else:
            print(f"\n🔍 [除錯資訊] 權重檔相容性檢查失敗！")
            print(f"❌ 遺失標記: {missing}")
            # 如果只缺了 projection，我們也試試看載入 (因為有時候 Keras 會改名)
            is_only_projection_missing = all("projection" in m for m in missing)
            if is_only_projection_missing and len(missing) > 0:
                print(f"💡 [遷移學習] 僅遺失投影層標記，將嘗試強制載入...")
                return True
            return False
            
    except Exception as e:
        print(f"⚠️ 無法檢查權重檔相容性: {e}")
        return False


def load_transfer_weights_safely(model, weights_path, allow_partial=False):
    """
    修正版：針對「打平後的權重檔」進行精準載入。
    分別對子模型進行 load_weights，避開層級路徑不匹配的問題。
    """
    print(f"📥 [遷移學習] 正在嘗試進行精準子模型層對齊載入...")
    
    try:
        # 直接對整個 PINN_Feedback_Model 進行載入，讓 Keras 自動遞迴匹配子模型的層名稱
        model.load_weights(weights_path, skip_mismatch=True)
        return True
    except Exception as e:
        print(f"⚠️ [遷移學習] 載入失敗: {e}")
        return False


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
        self.ffn_1 = layers.Dense(ff_dim, activation="gelu", name="ffn_dense_1")
        self.ffn_drop = layers.Dropout(dropout)
        self.ffn_2 = layers.Dense(d_model, name="ffn_dense_2")
        
        self.norm1 = layers.LayerNormalization(epsilon=1e-6, name="norm1")
        self.norm2 = layers.LayerNormalization(epsilon=1e-6, name="norm2")
        self.drop1 = layers.Dropout(dropout)
        self.drop2 = layers.Dropout(dropout)


    def call(self, inputs, training=False):
        attn_output = self.attention(inputs, inputs, training=training)
        attn_output = self.drop1(attn_output, training=training)
        x = self.norm1(inputs + attn_output)

        ffn_output = self.ffn_1(x)
        ffn_output = self.ffn_drop(ffn_output, training=training)
        ffn_output = self.ffn_2(ffn_output)
        
        ffn_output = self.drop2(ffn_output, training=training)
        return self.norm2(x + ffn_output)

# ==========================================
# 2. 進階版 PINN 模型 (統一為一期架構並加入 LoRA)
# ==========================================
class FlowPredictor(Model):
    def __init__(self, config, idx_map):
        super(FlowPredictor, self).__init__()
        self.config, self.idx_map = config, idx_map
        self.input_projection = layers.Dense(config["d_model"], name="flow_input_projection")
        self.positional_encoding = PositionalEncoding(config["window_size"], config["d_model"])
        self.transformer_blocks = [
            TransformerEncoderBlock(config["d_model"], config["num_heads"], config["ff_dim"], config["dropout"])
            for _ in range(config["num_transformer_layers"])
        ]
        self.sequence_pool = layers.GlobalAveragePooling1D(name="flow_sequence_pool")
        self.instant_dense = layers.Dense(32, activation='swish', name="flow_instant_dense")
        self.head_q_1 = layers.Dense(128, activation='swish', name="head_q_1")
        self.head_q_2 = layers.Dense(64, activation='swish', name="head_q_2")
        self.head_q_out = layers.Dense(idx_map["n_wells"], activation='relu', name="head_q_out")
        self.head_inflow_1 = layers.Dense(32, activation='swish', name="head_inflow_1")
        self.head_inflow_out = layers.Dense(1, activation='softplus', name="head_inflow_out")

    def call(self, inputs, c_feature, training=False):
        start_q = self.idx_map["n_obs"]
        current_pump = inputs[:, -1, start_q:self.idx_map["h_end"]]
        current_h = inputs[:, -1, :self.idx_map["h_end"]]

        seq_features = self.input_projection(inputs)
        seq_features = self.positional_encoding(seq_features)
        for block in self.transformer_blocks:
            seq_features = block(seq_features, training=training)
        transformer_out = self.sequence_pool(seq_features)

        combined = layers.Concatenate()([transformer_out, self.instant_dense(current_pump), c_feature, current_h])
        q_out = self.head_q_1(combined)
        q_out = self.head_q_2(q_out)
        q_out = self.head_q_out(q_out)
        qin_out = self.head_inflow_1(combined)
        qin_out = self.head_inflow_out(qin_out)
        return q_out, qin_out


class WaterLevelPredictor(Model):
    def __init__(self, config, idx_map):
        super(WaterLevelPredictor, self).__init__()
        self.config, self.idx_map = config, idx_map
        self.input_projection = layers.Dense(config["d_model"], name="h_input_projection")
        self.positional_encoding = PositionalEncoding(config["window_size"], config["d_model"])
        self.transformer_blocks = [
            TransformerEncoderBlock(config["d_model"], config["num_heads"], config["ff_dim"], config["dropout"])
            for _ in range(config["num_transformer_layers"])
        ]
        self.sequence_pool = layers.GlobalAveragePooling1D(name="h_sequence_pool")
        self.instant_dense = layers.Dense(32, activation='swish', name="h_instant_dense")
        self.head_h_1 = layers.Dense(256, activation='swish', name="head_h_1")
        self.head_h_2 = layers.Dense(128, activation='swish', name="head_h_2")
        self.head_h_3 = layers.Dense(64, activation='swish', name="head_h_3")
        self.head_h_4 = layers.Dense(32, activation='swish', name="head_h_4")
        self.head_h_out = layers.Dense(idx_map["h_end"], name="head_h_out")
        self.cp_1 = layers.Dense(64, activation='swish', name="cp_1")
        self.cp_out = layers.Dense(32, name="cp_out")

    def call(self, inputs, c_feature, training=False):
        start_q = self.idx_map["n_obs"]
        current_pump = inputs[:, -1, start_q:self.idx_map["h_end"]]
        current_h = inputs[:, -1, :self.idx_map["h_end"]]

        seq_features = self.input_projection(inputs)
        seq_features = self.positional_encoding(seq_features)
        for block in self.transformer_blocks:
            seq_features = block(seq_features, training=training)
        transformer_out = self.sequence_pool(seq_features)

        combined = layers.Concatenate()([transformer_out, self.instant_dense(current_pump), c_feature, current_h])
        contrastive_feat = self.cp_1(transformer_out)
        contrastive_feat = self.cp_out(contrastive_feat)

        h_out = self.head_h_1(combined)
        h_out = self.head_h_2(h_out)
        h_out = self.head_h_3(h_out)
        h_out = self.head_h_4(h_out)
        h_out = self.head_h_out(h_out)
        return h_out, contrastive_feat


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
        self.h_bias = self.add_weight(
            name="h_bias",
            shape=(idx_map["h_end"],),
            initializer="zeros",
            trainable=True,
            dtype=tf.float32,
        )
        self.flow_model = FlowPredictor(config, idx_map)
        self.h_model = WaterLevelPredictor(config, idx_map)
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(config["nn_lr_init"], 1000, 0.9)
        self.opt_nn = tf.keras.optimizers.Adam(learning_rate=lr_schedule, global_clipnorm=config["clip_norm"])
        self.opt_phys = tf.keras.optimizers.Adam(learning_rate=config["phys_lr"])
        self.curr_epoch = tf.Variable(0.0, trainable=False, dtype=tf.float32)

    def build(self, input_shape):
        dummy_inputs = tf.zeros((1, input_shape[1], input_shape[2]), dtype=tf.float32)
        dummy_c = tf.zeros((1, self.idx_map["n_wells"]), dtype=tf.float32)
        self.flow_model(dummy_inputs, dummy_c, training=False)
        self.h_model(dummy_inputs, dummy_c, training=False)
        super(PINN_Feedback_Model, self).build(input_shape)

    def call(self, inputs, training=None):
        current_C = tf.exp(self.C_log)
        batch_size = tf.shape(inputs)[0]
        c_feature = tf.tile(tf.expand_dims(current_C, 0), [batch_size, 1])
        q_out, qin_out = self.flow_model(inputs, c_feature, training=training)
        h_out, contrastive_feat = self.h_model(inputs, c_feature, training=training)
        h_out = tf.clip_by_value(tf.cast(h_out, tf.float32) + self.h_bias, 0.0, 1.0)
        return h_out, q_out, qin_out, contrastive_feat

    def nt_xent_loss(self, z1, z2, temperature=0.1):
        z1 = tf.math.l2_normalize(z1, axis=1)
        z2 = tf.math.l2_normalize(z2, axis=1)
        batch_size = tf.shape(z1)[0]
        representations = tf.concat([z1, z2], axis=0) # shape: (2N, 32)
        similarity_matrix = tf.matmul(representations, representations, transpose_b=True) / temperature
        
        mask = tf.eye(2 * batch_size, dtype=tf.bool)
        similarity_matrix = tf.where(mask, -1e9, similarity_matrix)
        
        labels = tf.concat([tf.range(batch_size, 2 * batch_size), tf.range(0, batch_size)], axis=0)
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, similarity_matrix, from_logits=True)
        return tf.reduce_mean(loss)

    def calculate_losses(self, hp, qp, qin_p, y_h, y_q, X):
        T_s = tf.exp(tf.clip_by_value(self.T_log, tf.math.log(1.0), tf.math.log(500.0)))
        R_s = tf.exp(self.R_log) + 5.0
        C_s = tf.exp(tf.clip_by_value(self.C_log, -10.0, 2.0))
        Sy = tf.sigmoid(self.Sy_logit)
        h_real = hp * (self.h_max - self.h_min) + self.h_min
        q_real = tf.maximum(qp * (self.q_max - self.q_min) + self.q_min, 0.0)
        qin_real = qin_p * 500.0
        
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
        mass_balance_error = (self.config["area_A"] * Sy * delta_h) - ((qin_real[:, 0] - tf.reduce_sum(q_real, axis=1)) * self.config["DELTA_T"])
        l_phys_temporal = tf.reduce_mean(tf.square(mass_balance_error))
        
        return tf.reduce_mean(tf.square(y_h - hp)), (l_phys_spatial * 0.1) + l_phys_temporal, tf.reduce_mean(tf.square(y_q - qp))

    def train_step(self, data):
        X, y = data
        y_h, y_q = y[:, :self.idx_map["h_end"]], y[:, self.idx_map["h_end"]:]
        self.curr_epoch.assign_add(1.0 / (self.total_samples / self.config["batch_size"]))
        is_warmup = self.curr_epoch < float(self.config["warmup_epochs"])
        l_phys_w = tf.cond(is_warmup, lambda: 0.0, lambda: tf.minimum(self.config["lambda_phys_final"] * ((self.curr_epoch - self.config["warmup_epochs"]) / 20.0), self.config["lambda_phys_final"]))
        
        ss_prob = tf.clip_by_value(
            (self.curr_epoch - float(self.config["warmup_epochs"])) / float(self.config["epochs"]),
            0.0, 0.5
        )
        
        X_aug = X + tf.random.normal(shape=tf.shape(X), mean=0.0, stddev=0.015)

        with tf.GradientTape(persistent=True) as tape:
            hp, qp, qip, z_clean = self(X, training=True)
            _, _, _, z_aug = self(X_aug, training=True)
            
            l_dat, l_phy, l_flo = self.calculate_losses(hp, qp, qip, y_h, y_q, X)
            
            hp_detached = tf.stop_gradient(hp)
            h_end = self.idx_map["h_end"]
            last_step_ss = tf.concat([hp_detached, X[:, -1, h_end:]], axis=-1)
            X_ss = tf.concat([X[:, :-1, :], tf.expand_dims(last_step_ss, 1)], axis=1)
            hp_ss, qp_ss, _, _ = self(X_ss, training=True)
            l_dat_ss = tf.reduce_mean(tf.square(y_h - hp_ss))
            
            l_cl_raw = self.nt_xent_loss(z_clean, z_aug, temperature=0.1)
            l_cl_weight = tf.cond(is_warmup, lambda: 0.0, lambda: 1.0)
            l_cl = l_cl_raw * l_cl_weight
            
            total_loss = (l_dat * 200.0) + (ss_prob * l_dat_ss * 200.0) + (l_phys_w * (l_phy / 500.0)) + (self.config["lambda_flow"] * l_flo) + (l_cl * 0.1)
            
        phys_vars = [self.T_log, self.R_log, self.C_log, self.Sy_logit]
        nn_vars = [v for v in self.trainable_variables if all(v is not p for p in phys_vars)]
        nn_grads = tape.gradient(total_loss, nn_vars)
        nn_grads_and_vars = [(g, v) for g, v in zip(nn_grads, nn_vars) if g is not None]
        if nn_grads_and_vars:
            self.opt_nn.apply_gradients(nn_grads_and_vars)

        def apply_phys():
            phys_grads = tape.gradient(total_loss, phys_vars)
            phys_grads_and_vars = [(g, v) for g, v in zip(phys_grads, phys_vars) if g is not None]
            if phys_grads_and_vars:
                self.opt_phys.apply_gradients(phys_grads_and_vars)
            return tf.constant(1.0)
        tf.cond(is_warmup, lambda: tf.constant(0.0), apply_phys)
        return {"loss": total_loss, "l_dat": l_dat, "l_phy": l_phy, "l_flo": l_flo, "l_cl": l_cl, "l_ss": l_dat_ss, "ss_prob": ss_prob}


# ==========================================
# 3. 數據處理與執行
# ==========================================
if __name__ == "__main__":

    base_config = config.copy()
    tasks = [
        # === 第一組：全資料集 (不限時間範圍，看 3/1 或 3/24 以前的所有歷史資料) ===
        {
            "save_folder": "微調/Report_Full_0301_Early",
            "FINETUNE_START": None, "FINETUNE_END": None,
            "TRAIN_CUTOFF":  "2026-03-01 00:00",
            "PREDICT_START": "2026-03-01 00:00", "PREDICT_END": "2026-03-08 00:00",
            "TEST_START": "2026-02-01 00:00", "TEST_END": "2026-02-15 00:00",
        },
        {
            "save_folder": "微調/Report_Full_0324_Late",
            "FINETUNE_START": None, "FINETUNE_END": None,
            "TRAIN_CUTOFF":  "2026-03-01 00:00",
            "PREDICT_START": "2026-03-24 00:00", "PREDICT_END": "2026-03-31 00:00",
            "TEST_START": "2026-03-01 00:00", "TEST_END": "2026-03-15 00:00",
        },
        # === 第二組：一個月微調 (僅使用 2026年 1 月份的資料) ===
        # {
        #     "save_folder": "微調/Report_1Month_0301_Early",
        #     "FINETUNE_START": "2026-01-01 00:00", "FINETUNE_END": "2026-02-01 00:00",
        #     "TRAIN_CUTOFF":  None,
        #     "PREDICT_START": "2026-03-01 00:00", "PREDICT_END": "2026-03-08 00:00",
        #     "TEST_START": "2026-02-15 00:00", "TEST_END": "2026-03-01 00:00",
        # },
        # {
        #     "save_folder": "微調/Report_1Month_0324_Late",
        #     "FINETUNE_START": "2026-01-01 00:00", "FINETUNE_END": "2026-02-01 00:00",
        #     "TRAIN_CUTOFF":  None,
        #     "PREDICT_START": "2026-03-24 00:00", "PREDICT_END": "2026-03-31 00:00",
        #     "TEST_START": "2026-03-10 00:00", "TEST_END": "2026-03-24 00:00",
        # },
        # # === 第三組：三個月微調 (使用 2026年 1月 到 3月 預測前夕的資料) ===
        # {
        #     "save_folder": "微調/Report_3Month_0301_Early",
        #     "FINETUNE_START": "2026-01-01 00:00", "FINETUNE_END": "2026-03-01 00:00",
        #     "TRAIN_CUTOFF":  None,
        #     "PREDICT_START": "2026-03-01 00:00", "PREDICT_END": "2026-03-08 00:00",
        #     "TEST_START": "2026-02-15 00:00", "TEST_END": "2026-03-01 00:00",
        # },
        # {
        #     "save_folder": "微調/Report_3Month_0324_Late",
        #     "FINETUNE_START": "2026-01-01 00:00", "FINETUNE_END": "2026-03-24 00:00",
        #     "TRAIN_CUTOFF":  None,
        #     "PREDICT_START": "2026-03-24 00:00", "PREDICT_END": "2026-03-31 00:00",
        #     "TEST_START": "2026-03-10 00:00", "TEST_END": "2026-03-24 00:00",
        # }
    ]

    for task_idx, task_params in enumerate(tasks):
        print(f"\n\n{'='*60}\n🚀 啟動獨立任務 {task_idx+1}/3: 儲存至 {task_params['save_folder']}\n{'='*60}\n")
        config = base_config.copy()
        config.update(task_params)
        tf.keras.backend.clear_session()
        save_p = config["save_folder"]
        if not os.path.exists(save_p): os.makedirs(save_p)

        print("📊 數據處理中...")
        data_path = config.get("training_data_path", "Phase2_Training_Data2.csv")
        if not os.path.exists(data_path):
            print(f"⚠️ 找不到 {data_path}，改用原始 Phase2_Training_Data.csv")
            data_path = "Phase2_Training_Data.csv"
        print(f"📄 使用訓練資料: {data_path}")
        df_raw = pd.read_csv(data_path, index_col=0)
        df_raw.index = pd.to_datetime(df_raw.index)
        dist_df = pd.read_csv('Distance_Matrix_Phase2.csv', index_col=0)

        # [數據切斷]：將 TRAIN_CUTOFF 移到後段僅針對訓練集切割，保留 df_master 完整性供 TEST 與 PREDICT 使用
        # if config.get("TRAIN_CUTOFF"):
        #     cutoff_date = pd.to_datetime(config["TRAIN_CUTOFF"])
        #     df_raw = df_raw.loc[df_raw.index < cutoff_date]
        #     print(f"✂️ 數據已根據 TRAIN_CUTOFF 切斷，目前數據上限為: {df_raw.index.max()}")

        df_master = df_raw.copy().ffill().bfill()
        print(f"📊 訓練資料（完整資料集）：{df_master.index.min()} → {df_master.index.max()}")

        phase1_obs_candidates = ['PA', 'PB', 'PC', 'FPS7', 'FPS8', 'FPS9', 'FPS2', 'FPS3', 'FPS4', 'FPS5', 'FPS6']
        phase2_obs_fallback = ['PW02', 'PW03', 'PW04']
        obs_cols = [c for c in phase1_obs_candidates if c in df_raw.columns]
        if not obs_cols:
            obs_cols = [c for c in phase2_obs_fallback if c in df_raw.columns]

        phase2_well_candidates = ["PW01", "PW010", "PW011", "PW05", "PW06", "PW07", "PW08", "PW09"]
        wells_list = [
            w for w in phase2_well_candidates
            if w in df_raw.columns and pd.to_numeric(df_raw[w], errors="coerce").abs().max() > 1e-6
        ]

        flow_cols = []
        for w in wells_list:
            sensor_id = map_sensor_id(w)
            for candidate in (f"Qw{sensor_id}", f"QW{sensor_id}"):
                if candidate in df_raw.columns:
                    flow_cols.append(candidate)
                    break

        wells_with_flow = [w for w, q in zip(wells_list, flow_cols)]
        wells_list = wells_with_flow
        obs_cols = [c for c in obs_cols if c in dist_df.index]
        wells_list = [w for w in wells_list if w in dist_df.index and w in dist_df.columns]
        flow_cols = [q for w, q in zip(wells_with_flow, flow_cols) if w in wells_list]

        if not obs_cols:
            raise ValueError("沒有可用的水位控制點同時存在於訓練資料與 Distance_Matrix_Phase2.csv。")
        if not wells_list or not flow_cols:
            raise ValueError("沒有可用的抽水井/流量欄位同時存在於訓練資料與 Distance_Matrix_Phase2.csv。")
    
        h_cols = obs_cols + wells_list

        # 🌟 重要修正：重新排列 DataFrame 欄位順序，確保 [水位] 在最前面，接著是 [流量]，最後是其他特徵
        # 這是為了確保 idx_map["h_end"] 索引到正確的水位資料 (避免誤抓到電力資料)
        other_cols = [c for c in df_master.columns if c not in h_cols and c not in flow_cols]
        df_master = df_master[h_cols + flow_cols + other_cols]

        # ==========================================
        # 1. & 2. 計算靜態偏差 (Static Bias Calculation) 與特徵增強 (Feature Augmentation)
        # ==========================================
        # MICROTUNE_ENHANCED: fit scaler and static offsets on the requested fine-tune window.
        finetune_start = pd.to_datetime(config["FINETUNE_START"]) if config.get("FINETUNE_START") else None
        finetune_end = pd.to_datetime(config["FINETUNE_END"]) if config.get("FINETUNE_END") else None
        df_fit_base = df_master
        if finetune_start is not None:
            df_fit_base = df_fit_base.loc[df_fit_base.index >= finetune_start]
        if finetune_end is not None:
            df_fit_base = df_fit_base.loc[df_fit_base.index < finetune_end]
        if len(df_fit_base) <= config["window_size"]:
            raise ValueError(
                f"微調資料不足：{config.get('FINETUNE_START')} 到 {config.get('FINETUNE_END')} "
                f"只有 {len(df_fit_base)} 筆，需大於 window_size={config['window_size']}。"
            )
        print(
            f"MICROTUNE_ENHANCED: fit window {df_fit_base.index.min()} -> {df_fit_base.index.max()} "
            f"({len(df_fit_base)} rows)"
        )

        overall_mean = df_fit_base[h_cols].mean().mean()
        well_mean = df_fit_base[h_cols].mean()
        diff_cols = [f"DIFF_{w}" for w in h_cols]
        for w, diff_w in zip(h_cols, diff_cols):
            df_master[diff_w] = well_mean[w] - overall_mean

        idx_map = {
            "n_obs": len(obs_cols), 
            "n_wells": len(wells_list), 
            "h_end": len(h_cols),
            "flow_start": df_master.columns.get_loc(flow_cols[0]),
            "diff_start": df_master.columns.get_loc(diff_cols[0])
        }
        print(f"📌 水位控制點: {obs_cols}")
        print(f"📌 抽水井輸出: {wells_list}")
        print(f"📌 抽水流量欄位: {flow_cols}")
        
        # 計算反歸一化參數 (MinMaxScaler: min, max) 以對齊一期
        h_min, h_max = df_master.iloc[:, :idx_map["h_end"]].min().values, df_master.iloc[:, :idx_map["h_end"]].max().values + 1e-7
        df_fit = df_master.loc[df_fit_base.index]
        h_min, h_max = df_fit.iloc[:, :idx_map["h_end"]].min().values, df_fit.iloc[:, :idx_map["h_end"]].max().values + 1e-7
        q_min, q_max = df_fit[flow_cols].min().values, df_fit[flow_cols].max().values + 1e-7
        # 後段輸出與 inference_pack 沿用 mean/std 命名；MinMax 反轉時 std 等同 range，mean 等同 min。
        h_mean, h_std = h_min, h_max - h_min
        q_mean, q_std = q_min, q_max - q_min
        h_max_phys = h_max
        
        scaler = MinMaxScaler().fit(df_fit)

        diff_max = np.max([df_fit[c].values[0] for c in diff_cols]) + 1e-6
        diff_min = np.min([df_fit[c].values[0] for c in diff_cols]) - 1e-6

        def scale_with_diff(df):
            scaled = scaler.transform(df)
            for c in diff_cols:
                col_idx = df_master.columns.get_loc(c)
                scaled[:, col_idx] = (df_master[c].iloc[0] - diff_min) / (diff_max - diff_min)
            return scaled

        def create_seq(df):
            data_s = scale_with_diff(df)
            X, y = [], []
            for i in range(len(data_s) - config["window_size"]):
                X.append(data_s[i:i+config["window_size"]])
                y.append(np.concatenate([data_s[i+config["window_size"], :idx_map["h_end"]], data_s[i+config["window_size"], idx_map["flow_start"]:idx_map["flow_start"]+idx_map["n_wells"]]]))
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

        # 🌟 套用 TRAIN_CUTOFF，僅拋棄超出範圍的訓練資料 (不影響 Test / Predict)
        if config.get("TRAIN_CUTOFF"):
            cutoff_date = pd.to_datetime(config["TRAIN_CUTOFF"])
            df_train_part1 = df_train_part1.loc[df_train_part1.index < cutoff_date]
            df_train_part2 = df_train_part2.loc[df_train_part2.index < cutoff_date]

        if finetune_start is not None:
            df_train_part1 = df_train_part1.loc[df_train_part1.index >= finetune_start]
            df_train_part2 = df_train_part2.loc[df_train_part2.index >= finetune_start]
        if finetune_end is not None:
            df_train_part1 = df_train_part1.loc[df_train_part1.index < finetune_end]
            df_train_part2 = df_train_part2.loc[df_train_part2.index < finetune_end]

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

        if len(X_train_full) == 0:
            raise ValueError("訓練序列為 0，請檢查 TRAIN_CUTOFF/TEST_START/TEST_END/window_size 設定。")
        if len(X_test) == 0:
            raise ValueError("盲測序列為 0，請拉長 TEST 區間或縮短 window_size。")
        print(f"🧩 序列資料: X_train={X_train_full.shape}, y_train={y_train_full.shape}, X_test={X_test.shape}, y_test={y_test.shape}")

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
            
                # 💡 重要：cv_model 開始的所有步驟都必須縮排在 for 裡面
                cv_model = PINN_Feedback_Model(
                    dist_df.loc[obs_cols+wells_list, wells_list].values, 
                    config, idx_map, h_min, h_max, q_min, q_max, len(X_tr)
                )
                cv_model.compile()
                cv_model.build(input_shape=X_tr.shape)
        
                print(f"  正在訓練 Fold {fold}...", flush=True)
                cv_model.fit(X_tr, y_tr, epochs=cv_epochs, batch_size=config["batch_size"], verbose=1)
        
                # 執行預測與反歸一化
                hp_val_s, _, _, _ = cv_model.predict(X_val, verbose=0)
                hp_val = hp_val_s * (h_max - h_min) + h_min
                y_val_h = y_val[:, :idx_map["h_end"]] * (h_max - h_min) + h_min
            
                # 修正這裡的 IndentationError
                wape_score = calculate_wape(y_val_h, hp_val)
                print(f"完成！ Fold {fold} Validation WAPE 誤差: {wape_score:.2f}%")
                cv_scores.append(wape_score)
                fold += 1
            
            print(f"\n📊 5-Fold CV 驗證平均 WAPE: {np.mean(cv_scores):.2f}%")
        else:
            print(f"\n⏭️ 根據設定，已跳過 5-Fold Cross Validation 流程。")

        # ==========================================
        # 🌟 3.6 最終全訓練集正式訓練 (整合分階段模式 + 遷移學習)
        # ==========================================
        model = PINN_Feedback_Model(dist_df.loc[obs_cols+wells_list, wells_list].values, config, idx_map, h_min, h_max, q_min, q_max, len(X_train_full))
        model.build(input_shape=X_train_full.shape)

        # ------------------------------------------
        # 📥 載入一期模型進行遷移學習 (Transfer Learning)
        # ------------------------------------------
        phase1_weights_path = config.get("phase1_weights_path", "")
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            abs_weights_path = os.path.join(base_dir, phase1_weights_path)
        except NameError:
            abs_weights_path = os.path.abspath(phase1_weights_path)
            
        if os.path.exists(abs_weights_path):
            print(f"\n🧠 [遷移學習] 偵測到一期模型權重 {abs_weights_path}！")
            if is_probably_compatible_weights(abs_weights_path):
                try:
                    load_transfer_weights_safely(
                        model,
                        abs_weights_path,
                        allow_partial=config.get("allow_partial_transfer", False),
                    )
                    print(f"✅ [遷移學習] 已嚴格載入相容的一期權重！")
                    
                    # 🚨 強制重置物理參數 (不同工地，地質不同)
                    model.T_log.assign(tf.fill([idx_map["h_end"]], tf.math.log(float(config["T_init"]))))
                    model.R_log.assign(tf.fill([idx_map["h_end"]], tf.math.log(float(config["R_init"]))))
                    model.C_log.assign(tf.fill([idx_map["n_wells"]], tf.math.log(float(config["C_init"]))))
                    model.Sy_logit.assign(tf.fill([1], 0.0))
                    print(f"🧹 [遷移學習] 已將物理參數 (T, R, C, Sy) 強制重置為初始值！")
                except Exception as e:
                    print(f"⚠️ 載入一期權重失敗，將從目前二期架構重新訓練。錯誤訊息: {e}")
            else:
                print("⚠️ [遷移學習] 權重檔架構與目前二期模型不相容，已跳過載入，改從目前二期架構重新訓練。")
        else:
            print(f"\n⚠️ [遷移學習] 未找到一期權重檔 (路徑: {abs_weights_path})，將從頭訓練。")
        # ------------------------------------------

        # --- Phase 1: 獨立流量預測訓練 ---
        print(f"\n🔧 [Phase 1] 已透過遷移學習載入一期權重，跳過獨立流量訓練...")
        
        # --- Phase 2: PINN 物理特徵訓練與微調 ---
        # ⚠️ 修正：因為二期 input_projection 重新初始化，若凍結 Transformer 會導致輸入雜訊。
        # 因此改為解凍 Transformer (trainable = True)，讓它能適應新的特徵投影空間。
        def reset_model_optimizers():
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(config["nn_lr_init"], 1000, 0.9)
            model.opt_nn = tf.keras.optimizers.Adam(learning_rate=lr_schedule, global_clipnorm=config["clip_norm"])
            model.opt_phys = tf.keras.optimizers.Adam(learning_rate=config["phys_lr"])

        def set_backbone_trainable(is_trainable):
            for block in model.flow_model.transformer_blocks:
                block.trainable = is_trainable
            for block in model.h_model.transformer_blocks:
                block.trainable = is_trainable
            model.flow_model.input_projection.trainable = is_trainable
            model.h_model.input_projection.trainable = is_trainable
            model.flow_model.positional_encoding.trainable = False
            model.h_model.positional_encoding.trainable = False

        # 保留 head 可訓練，讓 Phase 2 依現地資料調整輸出層。
        model.flow_model.head_q_1.trainable = True
        model.flow_model.head_q_2.trainable = True
        model.flow_model.head_q_out.trainable = True
        model.flow_model.head_inflow_1.trainable = True
        model.flow_model.head_inflow_out.trainable = True
        model.h_model.head_h_1.trainable = True
        model.h_model.head_h_2.trainable = True
        model.h_model.head_h_3.trainable = True
        model.h_model.head_h_4.trainable = True
        model.h_model.head_h_out.trainable = True
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="loss",
            patience=int(config.get("EARLY_STOPPING_PATIENCE", 40)),
            restore_best_weights=True,
            verbose=1,
        )
        head_warmup_epochs = int(config.get("HEAD_WARMUP_EPOCHS", 0))
        shuffle_training = bool(config.get("SHUFFLE_TRAINING", False))

        if head_warmup_epochs > 0:
            set_backbone_trainable(False)
            reset_model_optimizers()
            model.compile()
            trainable_count = int(np.sum([np.prod(v.shape) for v in model.trainable_variables]))
            print(f"MICROTUNE_ENHANCED: Stage 1 Head/Bias trainable params: {trainable_count:,}")
            model.fit(
                X_train_full,
                y_train_full,
                epochs=min(head_warmup_epochs, config["epochs"]),
                batch_size=config["batch_size"],
                verbose=1,
                shuffle=shuffle_training,
            )

        set_backbone_trainable(True)
        reset_model_optimizers()
        model.compile()
        trainable_count = int(np.sum([np.prod(v.shape) for v in model.trainable_variables]))
        print(f"MICROTUNE_ENHANCED: Stage 2 full-model trainable params: {trainable_count:,}")
        
        history = model.fit(
            X_train_full,
            y_train_full,
            epochs=config["epochs"],
            initial_epoch=min(head_warmup_epochs, config["epochs"]),
            batch_size=config["batch_size"],
            verbose=1,
            shuffle=shuffle_training,
            callbacks=[early_stop],
        )

        # ==========================================
        # 🌟 3.7 [盲測驗證] 最終模型成績發表
        # ==========================================
        print(f"\n🏆 正在對 {config['TEST_START']} 至 {config['TEST_END']} 進行盲測評分...")
        hp_test_s, qp_test_s, _, _ = model.predict(X_test, verbose=0)
        hp_test = hp_test_s * (h_max - h_min) + h_min
        qp_test = qp_test_s * (q_max - q_min) + q_min
        y_test_h = y_test[:, :idx_map["h_end"]] * (h_max - h_min) + h_min
        qt_test = y_test[:, idx_map["h_end"]:] * (q_max - q_min) + q_min
    
        test_wape = calculate_wape(y_test_h, hp_test)
        test_mae = mean_absolute_error(y_test_h, hp_test)
        print(f"==========================================")
        print(f"🏅 [盲測成績] 10月份整體預測準確率 (WAPE) : {test_wape:.2f}%")
        print(f"🏅 [盲測成績] 10月份整體預測平均誤差 (MAE) : {test_mae:.3f} m")
        print(f"==========================================\n")

        print("📊 正在生成 Full_Diagnostic_Report_Test.csv (TEST 區間診斷)...")
        test_diag_data = []
        test_point_names = obs_cols + wells_list
        for i, name in enumerate(test_point_names):
            h_mae = mean_absolute_error(y_test_h[:, i], hp_test[:, i])
            h_mape = calculate_mape(y_test_h[:, i], hp_test[:, i])

            q_mae, q_wape = np.nan, np.nan
            if name in wells_list:
                w_idx = wells_list.index(name)
                q_mae = mean_absolute_error(qt_test[:, w_idx], qp_test[:, w_idx])
                q_wape = calculate_wape(qt_test[:, w_idx], qp_test[:, w_idx], is_flow=True)

            test_diag_data.append({
                "Well": name,
                "Type": "Pump" if name in wells_list else "Obs",
                "MAE_Level(m)": h_mae,
                "MAPE_Level(%)": h_mape,
                "MAE_Flow(m3/hr)": q_mae,
                "WAPE_Flow(%)": q_wape,
            })

        df_diag_test = pd.DataFrame(test_diag_data)
        df_diag_test.to_csv(f"{save_p}/Full_Diagnostic_Report_Test.csv", index=False, encoding='utf-8-sig')

        # ==========================================
        # 🌟 4. [物理提取] 包含 Qin 與 Sy 學習 (仍使用訓練集提取物理參數)
        # ==========================================
        print("\n🧹 正在進行物理診斷提取...")
        learned_sy = float(tf.sigmoid(model.Sy_logit).numpy()[0])
        hp_train_s, qp_train_s, _, _ = model.predict(X_train_full)
        hp_train = hp_train_s * h_std + h_mean
        qp_train = qp_train_s * q_std + q_mean
        ht_train = y_train_full[:, :idx_map["h_end"]] * h_std + h_mean
        qt_train = y_train_full[:, idx_map["h_end"]:] * q_std + q_mean

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

        # --- 儲存模型權重與推論包給推論腳本使用 ---
        model.save_weights(f"{save_p}/pinn_model.weights.h5")
        inference_pack = {
            "scaler": scaler, "diff_max": diff_max, "diff_min": diff_min,
            "h_mean": h_mean, "h_std": h_std, "q_mean": q_mean, "q_std": q_std,
            "idx_map": idx_map, "obs_cols": obs_cols, "wells_list": wells_list, "diff_cols": diff_cols,
            "config": config
        }
        joblib.dump(inference_pack, f"{save_p}/inference_pack.pkl")

        # --- 自回歸預測（不看答案）---
        predict_start = pd.to_datetime(config["PREDICT_START"])
        predict_end = pd.to_datetime(config["PREDICT_END"])
        predict_steps = int((predict_end - predict_start).total_seconds() / (config["DELTA_T"] * 3600))
        print(f"\n🔮 自回歸預測：{config['PREDICT_START']} → {config['PREDICT_END']}（共 {predict_steps} 步，不看答案）...")

        # 取 PREDICT_START 前的 window_size 筆作為初始上下文
        context_data = df_master.loc[df_master.index < predict_start].tail(config["window_size"])
        assert len(context_data) >= config["window_size"], \
            f"❌ 上下文不足：PREDICT_START 前需要至少 {config['window_size']} 筆資料，僅有 {len(context_data)} 筆"

        curr_win = scale_with_diff(context_data).reshape(1, config["window_size"], -1)

        # ✅ 初始條件錨定：將 context window 最後一步替換為 PREDICT_START 的真實狀態 (包含抽水井與流量)
        # 這確保自回歸預測從「真實已知的完整起始狀態」出發，防止歷史資料斷層導致的 5m 初始巨大落差
        actual_start_row = df_master.loc[df_master.index <= predict_start].iloc[-1]
        actual_start_h_raw = actual_start_row[obs_cols].values.astype(float)
        actual_start_h_all_raw = actual_start_row[obs_cols + wells_list].values.astype(float)
        
        # 用與訓練相同的 scaler 將真實狀態歸一化後寫入 context window 最後一步
        full_start_scaled = scale_with_diff(actual_start_row.to_frame().T)[0]
        curr_win[0, -1, :] = full_start_scaled  # 🌟 修正：全部特徵(包含抽水井、流量)都必須對齊到最新一刻！
        print(f"  📍 初始條件錨定完成：PREDICT_START 實際水位 = {np.round(actual_start_h_raw, 2)}")

        future_feedback_df = df_master.loc[(df_master.index > predict_start) & (df_master.index <= predict_end)].copy()
    
        # 取消 1/3 真實值輸入，強制作為 0，進行全程盲測
        teacher_forcing_steps = 0
        da_interval = int(7 * 24 / config["DELTA_T"]) # 💡 [路線A] Data Assimilation: 每 7 天 (336步) 校正一次軌道
        print(f"  🎯 [路線A] 滾動校正：每 {da_interval * config['DELTA_T']} 小時注入一次現場真實水位，消除累積誤差")

        future_h, future_q, future_qin = [], [], []
        for step in range(predict_steps):
            hp_s, qp_s, qip_s, _ = model(curr_win, training=False)

            hp_pred_full = hp_s.numpy()[0, :idx_map["h_end"]] # 取得全部 24 口井水位
            qp_pred = qp_s.numpy()[0, :idx_map["n_wells"]]
            qin_pred = qip_s.numpy()[0, 0]

            future_h.append(hp_pred_full * h_std + h_mean)
            future_q.append(qp_pred * q_std + q_mean)
            future_qin.append(qin_pred * 500.0)

            if step < teacher_forcing_steps:
                next_step_scaled = scale_with_diff(future_feedback_df.iloc[[step]])[0]
                new_step = next_step_scaled.reshape(1, 1, -1)
            else:
                new_step = curr_win[:, -1:, :].copy()
                new_step[0, 0, :idx_map["n_obs"]] = hp_pred_full[:idx_map["n_obs"]]
                
                if step < len(future_feedback_df):
                    next_step_scaled_real = scale_with_diff(future_feedback_df.iloc[[step]])[0]
                    # 強制將抽水井 (PW) 的水位於每一步覆蓋為真實資料，避免誤差累積
                    new_step[0, 0, idx_map["n_obs"]:idx_map["h_end"]] = next_step_scaled_real[idx_map["n_obs"]:idx_map["h_end"]]
                    if idx_map["flow_start"] < new_step.shape[-1]:
                        new_step[0, 0, idx_map["flow_start"]:idx_map["flow_start"]+idx_map["n_wells"]] = next_step_scaled_real[idx_map["flow_start"]:idx_map["flow_start"]+idx_map["n_wells"]]
                    # 覆蓋其餘所有外部特徵 (例如 X, KW) 讓模型感知抽水機開關變化
                    if new_step.shape[-1] > idx_map["flow_start"] + idx_map["n_wells"]:
                        new_step[0, 0, idx_map["flow_start"]+idx_map["n_wells"]:] = next_step_scaled_real[idx_map["flow_start"]+idx_map["n_wells"]:]
                else:
                    new_step[0, 0, idx_map["n_obs"]:idx_map["h_end"]] = curr_win[0, -1, idx_map["n_obs"]:idx_map["h_end"]]
                    if idx_map["flow_start"] < new_step.shape[-1]:
                        new_step[0, 0, idx_map["flow_start"]:idx_map["flow_start"]+idx_map["n_wells"]] = curr_win[0, -1, idx_map["flow_start"]:idx_map["flow_start"]+idx_map["n_wells"]]
                        new_step[0, 0, idx_map["flow_start"]:idx_map["flow_start"] + idx_map["n_wells"]] = qp_pred
                
                # 💡 [路線A] 每 7 天強制使用「真實水位」來修正起點，消除長程自回歸的滾雪球誤差
                if step > 0 and step % da_interval == 0 and step < len(future_feedback_df):
                    new_step[0, 0, :idx_map["n_obs"]] = next_step_scaled_real[:idx_map["n_obs"]]
                    print(f"    🔄 第 {step} 步 (累積 {step*config['DELTA_T']} 小時)：執行 Data Assimilation，匯入真實水位校正軌道！")

            curr_win = np.append(curr_win[:, 1:, :], new_step, axis=1)
            
        future_h = np.array(future_h)
        future_q = np.array(future_q)
        future_qin = np.array(future_qin)

        # Anchor the autoregressive trajectory to the measured level at PREDICT_START.
        # The model still predicts the 7-day shape, but the absolute datum is corrected
        # by the known initial condition. Keep the raw forecast for comparison.
        if bool(config.get("START_LEVEL_CORRECTION", False)) and len(future_h) > 0:
            future_h_raw = future_h.copy()
            start_level_bias = actual_start_h_all_raw - future_h_raw[0, :idx_map["h_end"]]
            max_abs_bias = config.get("START_LEVEL_CORRECTION_MAX_ABS")
            if max_abs_bias is not None:
                start_level_bias = np.clip(start_level_bias, -float(max_abs_bias), float(max_abs_bias))
            future_h = future_h_raw.copy()
            future_h[:, :idx_map["h_end"]] = future_h[:, :idx_map["h_end"]] + start_level_bias

            bias_df = pd.DataFrame({
                "Well": obs_cols + wells_list,
                "Actual_Start_Level(m)": actual_start_h_all_raw,
                "Raw_First_Pred_Level(m)": future_h_raw[0, :idx_map["h_end"]],
                "Applied_Bias(m)": start_level_bias,
            })
            bias_df.to_csv(f"{save_p}/start_level_bias_correction.csv", index=False, encoding="utf-8-sig")
            np.save(f"{save_p}/background_h_7d_raw.npy", future_h_raw)
            print("\nSTART_LEVEL_CORRECTION enabled")
            print("  Saved raw forecast: background_h_7d_raw.npy")
            print("  Saved bias table: start_level_bias_correction.csv")
            print(f"  Obs bias range: {start_level_bias[:idx_map['n_obs']].min():.3f} ~ {start_level_bias[:idx_map['n_obs']].max():.3f} m")
        
        np.save(f"{save_p}/background_h_7d.npy", future_h)
        np.save(f"{save_p}/future_q_7d.npy", future_q)
        np.save(f"{save_p}/qin_7d_dynamic.npy", future_qin)

        # --- 新增：反向基準推導專用 --- 產生含現場真實抽水的極度準確預測水位
        print("\n🎯 [混合重疊原理] 正在產生含電力特徵的神準預測水位（Reverse Superposition Baseline）...")
        delta_hours = config["window_size"] * config["DELTA_T"]
        actual_predict_data = df_master.loc[(df_master.index >= (predict_start - pd.Timedelta(hours=delta_hours))) & (df_master.index < predict_end)]
        if len(actual_predict_data) > config["window_size"]:
            X_acc, _ = create_seq(actual_predict_data)
            X_acc = X_acc[:predict_steps]
            hp_acc_s, _, _, _ = model.predict(X_acc, verbose=0)
            # 神準預測也取全部井
            accurate_h = hp_acc_s[:, :idx_map["h_end"]] * h_std + h_mean
            np.save(f"{save_p}/accurate_pred_h_7d.npy", accurate_h)
            print(f"  📥 已儲存：accurate_pred_h_7d.npy (包含觀測井與抽水井)")
        else:
            print(f"  ⚠️ 無法產生 accurate_pred_h_7d.npy")
        # ----------------------------------------------------

        # --- 預測 vs 實際比對 (包含抽水井水位) ---
        actual_test_df = df_master.loc[(df_master.index > predict_start) & (df_master.index <= predict_end)][obs_cols + wells_list]
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

            # ==========================================
            # 💡 4. 移除後處理：預測完後，直接輸出數值，不再做任何手動平移
            # 背景水位與神準預測直接以模型原始輸出的 feature-learned state 為準
            # ==========================================
            print(f"\n✅ 已依據靜態偏移特徵 (DIFF) 直出預測結果，取消原有 Bias 後處理。")
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
            h_mape = calculate_mape(ht_train[:, i], hp_train[:, i])
        
            # B. 初始化流量誤差與效率
            q_mae, q_wape, eff = np.nan, np.nan, 100.0
        
            # C. 抽水井專屬誤差計算
            if name in wells_list:
                w_idx = wells_list.index(name)
                # 流量誤差 (Actual qt vs Predicted qp)
                q_mae = mean_absolute_error(qt_train[:, w_idx], qp_train[:, w_idx])
                q_wape = calculate_wape(qt_train[:, w_idx], qp_train[:, w_idx], is_flow=True)
            
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
                "WAPE_Flow(%)": q_wape,
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
        print(f"History keys: {list(history.history.keys())}")
        for l_key, l_name, l_file in [('l_dat','Data Loss','4_Data_Loss.png'),('l_phy','Phys Loss','5_Phys_Loss.png'),('loss','Total Loss','6_Total_Loss.png')]:
            if l_key not in history.history:
                print(f"⚠️ Skip {l_name}: history key '{l_key}' not found.")
                continue
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

        # 6. 預測 vs 實際 對比圖 (Prediction_vs_Actual)
        print("📊 正在繪製水位與流量對比圖...")
        time_hours = np.arange(len(future_h)) * config["DELTA_T"]
        
        # --- 6.1 觀測井水位 (Level_Obs) ---
        n_obs = len(obs_cols)
        rows_obs = int(np.ceil(n_obs / 3))
        fig_obs, axes_obs = plt.subplots(rows_obs, 3, figsize=(18, 4 * rows_obs))
        axes_obs = axes_obs.flatten()
        for i, col in enumerate(obs_cols):
            axes_obs[i].plot(time_hours, future_h[:, i], color='forestgreen', linewidth=2, label='Pred')
            if actual_h_test is not None:
                n_f = min(len(future_h), len(actual_h_test))
                mae_val = mean_absolute_error(actual_h_test[:n_f, i], future_h[:n_f, i])
                axes_obs[i].plot(time_hours[:n_f], actual_h_test[:n_f, i], color='steelblue', alpha=0.6, label='Actual')
                axes_obs[i].set_title(f"{col} (MAE={mae_val:.3f}m)")
            else:
                axes_obs[i].set_title(f"Obs Level: {col}")
            axes_obs[i].set_ylim([-25, -10]) # 固定 Y 軸範圍
            axes_obs[i].legend(); axes_obs[i].grid(True, alpha=0.3)
        for j in range(n_obs, len(axes_obs)): fig_obs.delaxes(axes_obs[j])
        fig_obs.tight_layout(); fig_obs.savefig(f"{save_p}/Prediction_vs_Actual_Obs.png"); plt.close()

        # --- 6.2 抽水井水位 (Level_PW) ---
        n_pw = len(wells_list)
        rows_pw = int(np.ceil(n_pw / 3))
        fig_pw, axes_pw = plt.subplots(rows_pw, 3, figsize=(18, 4 * rows_pw))
        axes_pw = axes_pw.flatten()
        for i, col in enumerate(wells_list):
            idx = n_obs + i
            axes_pw[i].plot(time_hours, future_h[:, idx], color='darkorange', linewidth=2, label='Pred')
            if actual_h_test is not None:
                n_f = min(len(future_h), len(actual_h_test))
                mae_val = mean_absolute_error(actual_h_test[:n_f, idx], future_h[:n_f, idx])
                axes_pw[i].plot(time_hours[:n_f], actual_h_test[:n_f, idx], color='dimgray', alpha=0.6, label='Actual')
                axes_pw[i].set_title(f"{col} (MAE={mae_val:.3f}m)")
            else:
                axes_pw[i].set_title(f"PW Level: {col}")
            axes_pw[i].legend(); axes_pw[i].grid(True, alpha=0.3)
        for j in range(n_pw, len(axes_pw)): fig_pw.delaxes(axes_pw[j])
        fig_pw.tight_layout(); fig_pw.savefig(f"{save_p}/Prediction_vs_Actual_PW.png"); plt.close()

        # --- 6.3 抽水量 (Flow) ---
        rows_q = int(np.ceil(n_pw / 3))
        fig_q, axes_q = plt.subplots(rows_q, 3, figsize=(18, 4 * rows_q))
        axes_q = axes_q.flatten()
        actual_q_test = df_test[flow_cols].values if 'df_test' in locals() else None
        for i, col in enumerate(wells_list):
            axes_q[i].plot(time_hours, future_q[:, i], color='crimson', linewidth=2, label='Pred')
            if actual_q_test is not None:
                n_f = min(len(future_q), len(actual_q_test))
                mae_val = mean_absolute_error(actual_q_test[:n_f, i], future_q[:n_f, i])
                axes_q[i].plot(time_hours[:n_f], actual_q_test[:n_f, i], color='black', alpha=0.4, label='Actual')
                axes_q[i].set_title(f"{col} (MAE={mae_val:.2f} m3/hr)")
            else:
                axes_q[i].set_title(f"Flow: {col}")
            axes_q[i].legend(); axes_q[i].grid(True, alpha=0.3)
        for j in range(n_pw, len(axes_q)): fig_q.delaxes(axes_q[j])
        fig_q.tight_layout(); fig_q.savefig(f"{save_p}/Prediction_vs_Actual_Flow.png"); plt.close()

        # ==========================================
        # 移除 bias correction 圖表繪製
        # ==========================================

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

        # ==========================================
        # 🌟 最新功能：儲存推論專用的大腦與測量器 (供推論程式瞬間啟動使用)
        # ==========================================
        print(f"\n💾 正在匯出神經網路權重與縮放器至 {save_p} ...")
        model.save_weights(f"{save_p}/pinn_model.weights.h5")
        
        # 匯出所需的重要變數
        import joblib
        # 確保這些變數在你的訓練腳本中已經計算好
        inference_pack = {
            "scaler": scaler,  # 直接存入 sklearn 對象
            "diff_max": float(diff_max),
            "diff_min": float(diff_min),
            
            # 確保水位與流量的統計量被正確截取與轉換
            "h_mean": h_mean.astype(float), 
            "h_std":  h_std.astype(float),
            "q_mean": q_mean.astype(float),
            "q_std":  q_std.astype(float),
            
            "h_max_phys": h_max_phys.astype(float),
            "idx_map": idx_map,
            "obs_cols": obs_cols,
            "wells_list": wells_list,
            "diff_cols": diff_cols,
            "config": config  # 包含 window_size, DELTA_T 等
        }
        
        save_path = f"{save_p}/inference_pack.pkl"
        joblib.dump(inference_pack, save_path)
        print(f"✅ 推論包已成功匯出至: {save_path}")
        print(f"📊 包含水位口數: {len(h_mean)}, 抽水井口數: {len(q_mean)}")
