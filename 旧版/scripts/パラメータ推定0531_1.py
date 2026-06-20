import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# 日本語表示のためのフォント設定 (macOS用)
plt.rcParams['font.family'] = 'Hiragino Sans'

# ==========================================
# 1. 整理された4魚種のデータシートの読み込み
# ==========================================
# 各魚種のCSVデータを個別に読み込み、年（Year）で結合します。
df_sardine = pd.read_csv('/Users/fukuokashunya/MSY/data/マイワシ時系列データ_資源量・漁獲量・漁獲係数 - マイワシ時系列データ_資源量・漁獲量・漁獲係数.csv')
df_yellowtail = pd.read_csv('/Users/fukuokashunya/MSY/data/ブリ時系列データ_資源量・漁獲量・漁獲係数 - ブリ時系列データ_資源量・漁獲量・漁獲係数.csv')
df_mackerel = pd.read_csv('/Users/fukuokashunya/MSY/data/マサバ時系列データ_資源量・漁獲量・漁獲係数 - マサバ時系列データ_資源量・漁獲量・漁獲係数.csv')
df_squid = pd.read_csv('/Users/fukuokashunya/MSY/data/スルメイカ秋季 資源量・漁獲量時系列データ - スルメイカ秋季 資源量・漁獲量時系列データ.csv')

# 「漁期年」を「年」に統一して結合キーとする
df_mackerel.rename(columns={'漁期年': '年'}, inplace=True)
df_squid.rename(columns={'漁期年': '年'}, inplace=True)

# 共通する「年」でデータをマージ（内部結合）
df_merged = df_squid[['年', '資源量（千トン）', '漁獲量（千トン）']].merge(
    df_sardine[['年', '資源量（万トン）', '漁獲量（万トン）']], on='年'
).merge(
    df_yellowtail[['年', '資源量（トン）', '漁獲量（トン）']], on='年'
).merge(
    df_mackerel[['年', '資源量（万トン）', '漁獲量（万トン）']], on='年', suffixes=('_sardine', '_mackerel')
)

df_clean = df_merged.dropna().copy()

# 時間軸（年）の設定
t_real = (df_clean['年'] - df_clean['年'].min()).values.astype(float)
N_data = len(t_real)

# スケールダウン（計算安定化のため：百万トン単位などへ）
scale_factor = 1000.0

# 4種の資源量・漁獲圧データを抽出（単位を百万トンに統一してスケーリング）
x1_data = df_clean['資源量（千トン）'].values / 1000.0
x2_data = df_clean['資源量（万トン）_sardine'].values / 100.0
y1_data = df_clean['資源量（トン）'].values / 1000000.0
y2_data = df_clean['資源量（万トン）_mackerel'].values / 100.0

f_x1 = np.clip(df_clean['漁獲量（千トン）'].values / df_clean['資源量（千トン）'].values, 0.0, 0.95)
f_x2 = np.clip(df_clean['漁獲量（万トン）_sardine'].values / df_clean['資源量（万トン）_sardine'].values, 0.0, 0.95)
f_y1 = np.clip(df_clean['漁獲量（トン）'].values / df_clean['資源量（トン）'].values, 0.0, 0.95)
f_y2 = np.clip(df_clean['漁獲量（万トン）_mackerel'].values / df_clean['資源量（万トン）_mackerel'].values, 0.0, 0.95)

# フォワード検証（答え合わせ）用の漁獲圧補間関数
f_x1_interp = interp1d(t_real, f_x1, kind='linear', fill_value="extrapolate")
f_x2_interp = interp1d(t_real, f_x2, kind='linear', fill_value="extrapolate")
f_y1_interp = interp1d(t_real, f_y1, kind='linear', fill_value="extrapolate")
f_y2_interp = interp1d(t_real, f_y2, kind='linear', fill_value="extrapolate")

# ==========================================
# 2. ODEモデルの定義
# ==========================================
def full_system_ode(t, state, params):
    x1, x2, y1, y2 = state
    r_x1, r_x2, r_y1, r_y2, l11, l12, l21, l22, c1, c2, d1, d2 = params
    
    # 時間tにおける各魚種の漁獲圧を内挿
    fx1_t = f_x1_interp(t)
    fx2_t = f_x2_interp(t)
    fy1_t = f_y1_interp(t)
    fy2_t = f_y2_interp(t)
    
    dx1dt = (r_x1 - fx1_t) * x1 - l11 * x1 * y1 - l12 * x1 * y2
    # x2(マイワシ)にも漁獲圧 fx2_t を反映
    dx2dt = (r_x2 - fx2_t) * x2 - l21 * x2 * y1 - l22 * x2 * y2
    dy1dt = (-r_y1 - fy1_t) * y1 + c1 * l11 * x1 * y1 + d1 * l21 * x2 * y1
    dy2dt = (-r_y2 - fy2_t) * y2 + c2 * l12 * x1 * y2 + d2 * l22 * x2 * y2
    return [dx1dt, dx2dt, dy1dt, dy2dt]

# ==========================================
# 3. 積分軌道を用いた残差関数の定義
# ==========================================
def residuals_ode_integration(params, t_real, x1_data, x2_data, y1_data, y2_data, init_conditions):
    sol = solve_ivp(
        full_system_ode,
        [t_real[0], t_real[-1]],
        init_conditions,
        t_eval=t_real,
        args=(params,),
        method='RK45'
    )
    
    # ソルバーが失敗した場合や長さが合わない場合は大きなペナルティ
    if sol.status != 0 or sol.y.shape[1] != len(t_real):
        return np.ones(len(t_real) * 4) * 1e6
        
    sim_x1, sim_x2, sim_y1, sim_y2 = sol.y
    
    res_x1 = sim_x1 - x1_data
    res_x2 = sim_x2 - x2_data
    res_y1 = sim_y1 - y1_data
    res_y2 = sim_y2 - y2_data
    
    return np.concatenate([res_x1, res_x2, res_y1, res_y2])

# ==========================================
# 4. 最適化の実行
# ==========================================
# 初期推測値
# r_x1, r_x2, r_y1, r_y2, l11, l12, l21, l22, c1, c2, d1, d2
initial_guess = [1.5, 1.5, 0.5, 0.5,  0.5, 0.5, 0.5, 0.5,  0.5, 0.5, 0.5, 0.5]

# 現実的な境界条件（すべての相互作用パラメータを正の領域に幽閉）
lower_bounds = [0.01] * 12
upper_bounds = [20.0, 20.0, 10.0, 10.0,  5.0, 5.0, 5.0, 5.0,  2.0, 2.0, 2.0, 2.0]
bounds = (lower_bounds, upper_bounds)

init_conditions = [x1_data[0], x2_data[0], y1_data[0], y2_data[0]]

print("--- ODE積分軌道による直接フィッティングを開始します ---")
res = least_squares(
    residuals_ode_integration,
    initial_guess,
    bounds=bounds,
    args=(t_real, x1_data, x2_data, y1_data, y2_data, init_conditions),
    verbose=1
)

final_params = res.x

# ==========================================
# 5. 得られたパラメータを用いた最終シミュレーション
# ==========================================
sol_verify = solve_ivp(
    full_system_ode,
    [t_real[0], t_real[-1]],
    init_conditions,
    t_eval=t_real,
    args=(final_params,),
    method='RK45'
)

# ==========================================
# 6. 推定結果の出力と可視化
# ==========================================
print("\n=== 4種実データ連動 パラメータ逆算完了 ===")
param_names = ["r_x1", "r_x2", "r_y1", "r_y2", "l11", "l12", "l21", "l22", "c1", "c2", "d1", "d2"]
for i, name in enumerate(param_names):
    print(f"{name:<6} : {final_params[i]:.4f}")

fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
species_names = ['スルメイカ (x1)', 'マイワシ (x2)', 'ブリ (y1)', 'マサバ (y2)']
real_data_list = [x1_data, x2_data, y1_data, y2_data]

for i in range(2):
    for j in range(2):
        idx = i * 2 + j
        ax = axes[i, j]
        
        # モデルによるシミュレーション線 (千トンに復元)
        ax.plot(t_real, sol_verify.y[idx] * scale_factor, 'b-', linewidth=2.5, label='4種連動モデルの軌道')
        # 各魚種の本物の実データ (千トン)
        ax.plot(t_real, real_data_list[idx] * scale_factor, 'ro', alpha=0.7, label='水産庁 資源評価データ')
        
        ax.set_title(species_names[idx], fontsize=12)
        ax.set_ylabel('資源量（千トン）')
        ax.grid(True, linestyle='--')
        ax.legend()

plt.tight_layout()
plt.show()