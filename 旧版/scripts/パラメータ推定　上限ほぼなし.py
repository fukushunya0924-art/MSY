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

# 漁獲圧 f_x1 の計算
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
def bi_partite_lv_ode_latent(t, state, params):
    x2, y1, y2 = state
    r_x1, r_x2, r_y1, r_y2, l11, l12, l21, l22, c1, c2, d1, d2 = params
    
    # 【数理修正】物理的なハードバリア
    # 0.05を下回ろうとした瞬間、強力な復元バネ（100倍の反発力）を強制介入させ、
    # アルゴリズムに対して「これ以上0に近づくとエラーが激増する」という明確な傾き（勾配）を伝える
    dx2_barrier = 100.0 * (0.05 - x2) if x2 < 0.05 else 0.0
    dy1_barrier = 100.0 * (0.05 - y1) if y1 < 0.05 else 0.0
    dy2_barrier = 100.0 * (0.05 - y2) if y2 < 0.05 else 0.0
    
    x1_val = np.interp(t, t_real, x1_data)
    
    # 提示された通りの純粋な2部グラフ型4種モデルの数式
    dx2dt = r_x2 * x2 - l21 * x2 * y1 - l22 * x2 * y2 + dx2_barrier
    dy1dt = -r_y1 * y1 + c1 * l11 * x1_val * y1 + d1 * l21 * x2 * y1 + dy1_barrier
    dy2dt = -r_y2 * y2 + c2 * l12 * x1_val * y2 + d2 * l22 * x2 * y2 + dy2_barrier
    
    return [dx2dt, dy1dt, dy2dt]

def residuals_pinn_philosophy(estimation_vector, t_real, left_side_data, x1_data):
    params = estimation_vector[0:12]
    init_latent = estimation_vector[12:15]
    
    sol = solve_ivp(
        bi_partite_lv_ode_latent,
        t_span=[t_real[0], t_real[-1]],
        y0=init_latent,
        t_eval=t_real,
        args=(params,),
        method='RK45',
        rtol=1e-6,
        atol=1e-6
    )
    
    if sol.status != 0 or sol.y.shape[1] != len(t_real):
        return np.ones(len(t_real)) * 1e6
    
    x2_sol, y1_sol, y2_sol = sol.y
    
    # スルメイカの数式ルール（右辺）
    r_x1, l11, l12 = params[0], params[4], params[5]
    right_side_model = r_x1 - l11 * y1_sol - l12 * y2_sol
    
    # 主目的：スルメイカの波のズレ
    base_residual = right_side_model - left_side_data
    
    # 副目的：他3種が0.05の危険水域に接近したことに対するソフトペナルティ（ヤコビアンの維持）
    soft_penalty = np.zeros_like(t_real)
    for s in [x2_sol, y1_sol, y2_sol]:
        soft_penalty += np.where(s < 0.1, (0.1 - s) * 10.0, 0.0)
        
    return base_residual + soft_penalty

# ==========================================
# 4. パラメータ推定の実行（制約の厳格化）
# ==========================================
# 初期推測値（周期を速くするため、rの初期値を高めに設定）
# r_x1, l11, l12,  r_x2, r_y1, r_y2, l21, l22, c1, c2, d1, d2,  x2_0, y1_0, y2_0
initial_guess = [5.0, 0.5, 0.5,  5.0, 2.0, 2.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,  0.5, 0.5, 0.5]

# 【数理修正】他3種の初期値（末尾3つの変数）の下限を「0.05」に厳格化し、絶対に絶滅スタートを許さない
lower_bounds = [0.1,  0.01, 0.01,  0.1,  0.1,  0.1,  0.01, 0.01, 0.01, 0.01, 0.01, 0.01,  0.05, 0.05, 0.05]
upper_bounds = [30.0, 10.0, 10.0,  30.0, 30.0, 30.0, 10.0, 10.0, 5.0,  5.0,  5.0,  5.0,   5.0,  5.0,  5.0]
bounds = (lower_bounds, upper_bounds)

print("--- 絶滅完全阻止型・4種連動フィッティングを開始します ---")
res = least_squares(
    residuals_pinn_philosophy,
    initial_guess,
    bounds=bounds,
    args=(t_real, left_side_data, x1_data),
    method='trf',  # 境界条件を厳密に守る信頼領域反射法を指定
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
    
    # 答え合わせ時も、推定時と全く同じハードバリアを適用して絶滅を物理的に遮断
    dx1_barrier = 100.0 * (0.05 - x1) if x1 < 0.05 else 0.0
    dx2_barrier = 100.0 * (0.05 - x2) if x2 < 0.05 else 0.0
    dy1_barrier = 100.0 * (0.05 - y1) if y1 < 0.05 else 0.0
    dy2_barrier = 100.0 * (0.05 - y2) if y2 < 0.05 else 0.0
    
    f_x1 = f_x1_func(t)
    dx1dt = (r_x1 - f_x1) * x1 - l11 * x1 * y1 - l12 * x1 * y2 + dx1_barrier
    dx2dt = r_x2 * x2 - l21 * x2 * y1 - l22 * x2 * y2 + dx2_barrier
    dy1dt = -r_y1 * y1 + c1 * l11 * x1 * y1 + d1 * l21 * x2 * y1 + dy1_barrier
    dy2dt = -r_y2 * y2 + c2 * l12 * x1 * y2 + d2 * l22 * x2 * y2 + dy2_barrier
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
        
        ax.plot(t_real, sol_verify.y[idx] * scale_factor, 'b-', linewidth=2, label='修正モデルによる推定動態')
        if idx == 0:
            ax.plot(t_real, df_clean['資源量（千トン）'].values, 'ro', alpha=0.6, label='実際の観測データ (csv)')
            
        ax.set_title(species_labels[i][j])
        ax.set_ylabel('資源量（千トン）' if idx < 2 else '個体数/資源量')
        ax.grid(True)
        ax.legend()

plt.tight_layout()
plt.show()