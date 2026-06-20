import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# 日本語表示のためのフォント設定 (macOS用)
plt.rcParams['font.family'] = 'Hiragino Sans'

# ==========================================
# 1. CSVデータ（スルメイカ資源量時系列）の読み込みと前処理
# ==========================================
csv_path = '/Users/fukuokashunya/MSY/スルメイカ秋季 資源量・漁獲量時系列データ - スルメイカ秋季 資源量・漁獲量時系列データ.csv'
df = pd.read_csv(csv_path)

# 欠損行の除外
df_clean = df.dropna(subset=['資源量（千トン）']).copy()

# 時間軸（年）の設定
t_real = (df_clean['漁期年'] - df_clean['漁期年'].min()).values.astype(float)

# スルメイカ資源量 (x1) の取得（百万トン単位）
scale_factor = 1000.0
x1_data = df_clean['資源量（千トン）'].values / scale_factor

# 漁獲圧 f_x1 の計算とクリッピング
raw_fishing_rates = df_clean['漁獲量（千トン）'].values / df_clean['資源量（千トン）'].values
fishing_rates = np.clip(raw_fishing_rates, 0.0, 0.95)
f_x1_interp = interp1d(t_real, fishing_rates, kind='linear', fill_value="extrapolate")

# ==========================================
# 2. PINNs思想に基づくデータ駆動型の数値微分（左辺の確定）
# ==========================================
dx1dt_data = np.gradient(x1_data, t_real)
left_side_data = (dx1dt_data / x1_data) + fishing_rates

# ==========================================
# 3. 潜在変数の軌道生成と残差関数
# ==========================================
def bi_partite_lv_ode_latent(t, state, latent_params, l11, l12):
    """裏の3種（x2, y1, y2）を動かす微分方程式（修正版：l11, l12を正しく連動）"""
    x2, y1, y2 = state
    r_x2, r_y1, r_y2, l21, l22, c1, c2, d1, d2 = latent_params
    
    x2 = max(0.0, x2)
    y1 = max(0.0, y1)
    y2 = max(0.0, y2)
    
    # スルメイカの実データx1(t)をタイムステップごとにリアルタイム内挿
    x1_val = np.interp(t, t_real, x1_data)
    
    dx2dt = r_x2 * x2 - l21 * x2 * y1 - l22 * x2 * y2
    
    # 【修正】勝手な固定値を排除し、最適化中の l11, l12 がリアルタイムに反映されるように変更
    dy1dt = -r_y1 * y1 + c1 * l11 * x1_val * y1 + d1 * l21 * x2 * y1
    dy2dt = -r_y2 * y2 + c2 * l12 * x1_val * y2 + d2 * l22 * x2 * y2
    
    return [dx2dt, dy1dt, dy2dt]

def residuals_pinn_philosophy(estimation_vector, t_real, left_side_data, x1_data):
    # 推定ベクトルの展開
    r_x1, l11, l12 = estimation_vector[0:3]
    latent_params = estimation_vector[3:12]
    init_latent = estimation_vector[12:15]
    
    # 1. 現在のパラメータ候補（l11, l12を含む）で裏の3種の時系列（軌道）を解く
    sol = solve_ivp(
        bi_partite_lv_ode_latent,
        t_span=[t_real[0], t_real[-1]],
        y0=init_latent,
        t_eval=t_real,
        args=(latent_params, l11, l12), # 【修正】ここでl11, l12をしっかり引き渡す
        method='RK45'
    )
    
    if sol.status != 0 or sol.y.shape[1] != len(t_real):
        return np.ones(len(t_real)) * 1e6
    
    x2_sol, y1_sol, y2_sol = sol.y
    
    # 2. スルメイカの数式ルール（右辺）を計算
    right_side_model = r_x1 - l11 * y1_sol - l12 * y2_sol
    
    # -------------------------------------------------------------
    # 【追加項目】生態学的生存ペナルティ（他3種の絶滅を物理的に禁止する）
    # -------------------------------------------------------------
    # 各タイムステップにおいて、資源量が 0.01（微小値）を下回っている度合いを計算
    extinction_penalty = np.zeros_like(t_real)
    for latent_species in [x2_sol, y1_sol, y2_sol]:
        # 0.01を下回った分だけ、ペナルティ（大きなエラー値）を加算する
        extinction_penalty += np.where(latent_species < 0.01, (0.01 - latent_species) * 100.0, 0.0)
    # -------------------------------------------------------------
    
    # 3. スルメイカのズレに、絶滅ペナルティを足し算して Scipy に返す
    return (right_side_model - left_side_data) + extinction_penalty
    
    # 3. 左辺（データ駆動）と右辺（モデル駆動）の代数的なズレを返す
    return right_side_model - left_side_data

# ==========================================
# 4. パラメータ推定の実行
# ==========================================
# 初期推測値（15変数、すべて動かします）
initial_guess = [1.5, 0.4, 0.3,  1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,  0.5, 0.5, 0.5]

# 現実的な境界条件の設定
lower_bounds = [0.1, 0.01, 0.01,  0.1, 0.1, 0.1, 0.01, 0.01, 0.1, 0.1, 0.1, 0.1,  0.01, 0.01, 0.01]
upper_bounds = [5.0, 2.0,  2.0,   3.0, 3.0, 3.0, 2.0,  2.0,  1.5, 1.5, 1.5, 1.5,  5.0,  5.0,  5.0]
bounds = (lower_bounds, upper_bounds)

print("--- PINNs思想に基づく完全連動フィッティングを開始します ---")
res = least_squares(
    residuals_pinn_philosophy,
    initial_guess,
    bounds=bounds,
    args=(t_real, left_side_data, x1_data),
    verbose=1
)

# ==========================================
# 5. 完全な4種系ODEでのフォワード検証（答え合わせ）
# ==========================================
final_params = res.x[0:12]
final_inits = [x1_data[0], res.x[12], res.x[13], res.x[14]]

def full_bi_partite_lv_ode(t, state, params, f_x1_func):
    x1, x2, y1, y2 = state
    r_x1, r_x2, r_y1, r_y2, l11, l12, l21, l22, c1, c2, d1, d2 = params
    x1 = max(0.0, x1); x2 = max(0.0, x2); y1 = max(0.0, y1); y2 = max(0.0, y2)
    f_x1 = f_x1_func(t)
    dx1dt = (r_x1 - f_x1) * x1 - l11 * x1 * y1 - l12 * x1 * y2
    dx2dt = r_x2 * x2 - l21 * x2 * y1 - l22 * x2 * y2
    dy1dt = -r_y1 * y1 + c1 * l11 * x1 * y1 + d1 * l21 * x2 * y1
    dy2dt = -r_y2 * y2 + c2 * l12 * x1 * y2 + d2 * l22 * x2 * y2
    return [dx1dt, dx2dt, dy1dt, dy2dt]

sol_verify = solve_ivp(
    full_bi_partite_lv_ode,
    [t_real[0], t_real[-1]],
    final_inits,
    t_eval=t_real,
    args=(final_params, f_x1_interp),
    method='RK45'
)

print("\n--- パラメータ推定完了 ---")
param_names = ["r_x1", "r_x2", "r_y1", "r_y2", "l11", "l12", "l21", "l22", "c1", "c2", "d1", "d2"]
for name, val in zip(param_names, final_params):
    print(f"{name:<6} : {val:.4f}")

# ==========================================
# 6. 可視化
# ==========================================
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
species_labels = [['x1 (スルメイカ資源量)', 'x2 (競合種 - 潜在変数)'], 
                  ['y1 (捕食者1 - 潜在変数)', 'y2 (捕食者2 - 潜在変数)']]

for i in range(2):
    for j in range(2):
        idx = i * 2 + j
        ax = axes[i, j]
        
        ax.plot(t_real, sol_verify.y[idx] * scale_factor, 'b-', linewidth=2, label='PINNs思想による推定動態')
        if idx == 0:
            ax.plot(t_real, df_clean['資源量（千トン）'].values, 'ro', alpha=0.6, label='実際の観測データ (csv)')
            
        ax.set_title(species_labels[i][j])
        ax.set_ylabel('資源量（千トン）' if idx < 2 else '個体数/資源量')
        ax.grid(True)
        ax.legend()

plt.tight_layout()
plt.show()