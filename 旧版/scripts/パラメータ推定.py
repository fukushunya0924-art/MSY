import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# 日本語表示のためのフォント設定 (macOS用)
plt.rcParams['font.family'] = 'Hiragino Sans'

# ==========================================
# 1. 2部グラフ型4種系ODEモデルの定義
# ==========================================
def bi_partite_lv_ode(t, state, params, f_x1_func):
    x1, x2, y1, y2 = state
    
    # パラメータの展開（12個）
    r_x1, r_x2, r_y1, r_y2, l11, l12, l21, l22, c1, c2, d1, d2 = params
    
    # 絶滅ガード（負の値にならないように制御）
    x1 = max(0.0, x1)
    x2 = max(0.0, x2)
    y1 = max(0.0, y1)
    y2 = max(0.0, y2)
    
    # 2部グラフ型4種系の数式
    # 漁獲圧 f_x1(t) をデータから取得して反映
    f_x1 = f_x1_func(t)
    dx1dt = (r_x1 - f_x1) * x1 - l11 * x1 * y1 - l12 * x1 * y2
    
    dx2dt = r_x2 * x2 - l21 * x2 * y1 - l22 * x2 * y2
    
    dy1dt = -r_y1 * y1 + c1 * l11 * x1 * y1 + d1 * l21 * x2 * y1
    dy2dt = -r_y2 * y2 + c2 * l12 * x1 * y2 + d2 * l22 * x2 * y2
    
    return [dx1dt, dx2dt, dy1dt, dy2dt]

# ==========================================
# 2. 最小二乗法のための残差（エラー）関数
# ==========================================
def residuals_latent(estimation_vector, t_eval, real_x1_data, f_x1_func):
    """
    estimation_vector の内訳:
    - [0:12]  : 12個のモデルパラメータ
    - [12]    : x2 の初期値 (x2_0)
    - [13]    : y1 の初期値 (y1_0)
    - [14]    : y2 の初期値 (y2_0)
    """
    params = estimation_vector[0:12]
    x1_0 = real_x1_data[0] # x1(スルメイカ)の初期値は実データの最初の値を使用
    x2_0 = estimation_vector[12]
    y1_0 = estimation_vector[13]
    y2_0 = estimation_vector[14]
    
    init_state = [x1_0, x2_0, y1_0, y2_0]
    
    # 与えられた初期値とパラメータ候補でODEを解く
    sol = solve_ivp(
        bi_partite_lv_ode,
        t_span=[t_eval[0], t_eval[-1]],
        y0=init_state,
        t_eval=t_eval,
        args=(params, f_x1_func),
        method='RK45'
    )
    
    # ソルバーが失敗した場合は大きなペナルティ
    if sol.status != 0 or sol.y.shape[1] != len(t_eval):
        return np.ones(len(t_eval)) * 1e6
    
    # 【重要】実データが存在する「x1（スルメイカ）のみ」の残差を計算して返す
    estimated_x1 = sol.y[0]
    return estimated_x1 - real_x1_data

# ==========================================
# 3. CSVデータ（スルメイカ資源量時系列）の読み込み
# ==========================================
csv_path = '/Users/fukuokashunya/MSY/スルメイカ秋季 資源量・漁獲量時系列データ - スルメイカ秋季 資源量・漁獲量時系列データ.csv'
df = pd.read_csv(csv_path)

# 資源量データが欠損している行を除外し、コピーを作成
df_clean = df.dropna(subset=['資源量（千トン）']).copy()

# 時間軸（年）の設定：最初の観測年を0年目とする
t_real = (df_clean['漁期年'] - df_clean['漁期年'].min()).values.astype(float)

# スルメイカ資源量 (x1) の取得。計算安定化のため1000で割り「百万トン」単位として扱う
scale_factor = 1000.0
real_x1_data = df_clean['資源量（千トン）'].values / scale_factor

# 漁獲圧 f_x1 の計算 (漁獲量 / 資源量) と補間関数の作成
# 漁獲量と資源量の単位が同じ(千トン)なので、そのまま割ることで比率（漁獲圧）が得られる
fishing_rates = df_clean['漁獲量（千トン）'].values / df_clean['資源量（千トン）'].values
f_x1_interp = interp1d(t_real, fishing_rates, kind='linear', fill_value="extrapolate")

# ==========================================
# 4. パラメータ ＋ 潜在初期値の同時推定実行
# ==========================================
# 推定対象（12個のパラメータ + 3つの隠れた初期値 = 計15変数）の初期推測値
# パラメータのguess(1.0や0.5) + 初期値のguess(すべて0.5と仮定)
initial_guess = [1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,   0.5, 0.5, 0.5]

# すべて正（> 0）の制約条件を設定
bounds = (1e-4 * np.ones(15), np.inf * np.ones(15))

print("--- 潜在変数を含むパラメータ推定を開始します ---")
res = least_squares(
    residuals_latent, 
    initial_guess, 
    bounds=bounds, 
    args=(t_real, real_x1_data, f_x1_interp),
    verbose=1
)

# 結果の分解
estimated_params = res.x[0:12]
estimated_inits = [real_x1_data[0], res.x[12], res.x[13], res.x[14]]

print("\n--- 推定・逆算完了 ---")
param_names = ["r_x1", "r_x2", "r_y1", "r_y2", "l11", "l12", "l21", "l22", "c1", "c2", "d1", "d2"]
for name, val in zip(param_names, estimated_params):
    print(f"{name:<6} : {val:.4f}")
print(f"\n逆算された初期値 x2(0): {res.x[12]:.4f}, y1(0): {res.x[13]:.4f}, y2(0): {res.x[14]:.4f}")

# ==========================================
# 5. 再現シミュレーションと可視化
# ==========================================
sol_estimated = solve_ivp(
    bi_partite_lv_ode, 
    [t_real[0], t_real[-1]], 
    estimated_inits, 
    t_eval=t_real, 
    args=(estimated_params, f_x1_interp)
)

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
species_labels = [['x1 (スルメイカ資源量)', 'x2 (競合種 - 潜在変数)'], 
                  ['y1 (捕食者1 - 潜在変数)', 'y2 (捕食者2 - 潜在変数)']]

for i in range(2):
    for j in range(2):
        idx = i * 2 + j
        ax = axes[i, j]
        
        # 推定された動態のプロット
        ax.plot(t_real, sol_estimated.y[idx] * scale_factor, 'b-', linewidth=2, label='Estimated Dynamics')
        
        if idx == 0:
            # x1の場合のみ、実データをプロット
            ax.plot(t_real, real_x1_data * scale_factor, 'ro', alpha=0.5, label='Actual Data (csv)')
            
        ax.set_title(species_labels[i][j])
        ax.set_ylabel('資源量（千トン）' if idx < 2 else '個体数/資源量')
        ax.grid(True)
        ax.legend()

plt.tight_layout()
plt.show()