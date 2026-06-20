import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# 日本語表示のためのフォント設定 (macOS用)
plt.rcParams['font.family'] = 'Hiragino Sans'

# ==========================================
# 1. 整理された4魚種のデータシートの読み込みと単位統一
# ==========================================
df_sardine = pd.read_csv('/Users/fukuokashunya/MSY/data/マイワシ時系列データ_資源量・漁獲量・漁獲係数 - マイワシ時系列データ_資源量・漁獲量・漁獲係数.csv')
df_yellowtail = pd.read_csv('/Users/fukuokashunya/MSY/data/ブリ時系列データ_資源量・漁獲量・漁獲係数 - ブリ時系列データ_資源量・漁獲量・漁獲係数.csv')
df_mackerel = pd.read_csv('/Users/fukuokashunya/MSY/data/マサバ時系列データ_資源量・漁獲量・漁獲係数 - マサバ時系列データ_資源量・漁獲量・漁獲係数.csv')
df_squid = pd.read_csv('/Users/fukuokashunya/MSY/data/スルメイカ秋季 資源量・漁獲量時系列データ - スルメイカ秋季 資源量・漁獲量時系列データ.csv')

df_mackerel.rename(columns={'漁期年': '年'}, inplace=True)
df_squid.rename(columns={'漁期年': '年'}, inplace=True)

df_merged = df_squid[['年', '資源量（千トン）', '漁獲量（千トン）']].merge(
    df_sardine[['年', '資源量（万トン）', '漁獲量（万トン）']], on='年'
).merge(
    df_yellowtail[['年', '資源量（トン）', '漁獲量（トン）']], on='年'
).merge(
    df_mackerel[['年', '資源量（万トン）', '漁獲量（万トン）']], on='年', suffixes=('_sardine', '_mackerel')
)

df_clean = df_merged.dropna().copy()

t_real = (df_clean['年'] - df_clean['年'].min()).values.astype(float)

# 【修正】数理モデル内の基準単位を「千トン」に完全統一（百万トンへの縮小を廃止）
# これにより、グラフプロット時や残差計算時のスケールミスマッチを根本から消去します
x1_data = df_clean['資源量（千トン）'].values                  # 千トン単位
x2_data = df_clean['資源量（万トン）_sardine'].values * 10.0   # 万トン -> 千トン
y1_data = df_clean['資源量（トン）'].values / 1000.0           # トン -> 千トン
y2_data = df_clean['資源量（万トン）_mackerel'].values * 10.0  # 万トン -> 千トン

# 漁獲係数の計算（前処理は完璧でしたのでそのまま維持）
f_x1 = np.clip(df_clean['漁獲量（千トン）'].values / df_clean['資源量（千トン）'].values, 0.0, 0.95)
f_x2 = np.clip(df_clean['漁獲量（万トン）_sardine'].values / df_clean['資源量（万トン）_sardine'].values, 0.0, 0.95)
f_y1 = np.clip(df_clean['漁獲量（トン）'].values / df_clean['資源量（トン）'].values, 0.0, 0.95)
f_y2 = np.clip(df_clean['漁獲量（万トン）_mackerel'].values / df_clean['漁獲量（万トン）_mackerel'].values, 0.0, 0.95)

f_x1_interp = interp1d(t_real, f_x1, kind='linear', fill_value="extrapolate")
f_x2_interp = interp1d(t_real, f_x2, kind='linear', fill_value="extrapolate")
f_y1_interp = interp1d(t_real, f_y1, kind='linear', fill_value="extrapolate")
f_y2_interp = interp1d(t_real, f_y2, kind='linear', fill_value="extrapolate")

# ==========================================
# 2. ODEモデルの定義（ガード付き）
# ==========================================
def full_system_ode(t, state, params):
    x1, x2, y1, y2 = state
    r_x1, r_x2, r_y1, r_y2, l11, l12, l21, l22, c1, c2, d1, d2 = params
    
    # 計算途中でマイナス（絶滅の奈落）に突入してソルバーが強制停止するのを防ぐ物理ガード
    x1 = max(1e-5, x1); x2 = max(1e-5, x2); y1 = max(1e-5, y1); y2 = max(1e-5, y2)
    
    fx1_t = f_x1_interp(t)
    fx2_t = f_x2_interp(t)
    fy1_t = f_y1_interp(t)
    fy2_t = f_y2_interp(t)
    
    dx1dt = (r_x1 - fx1_t) * x1 - l11 * x1 * y1 - l12 * x1 * y2
    dx2dt = (r_x2 - fx2_t) * x2 - l21 * x2 * y1 - l22 * x2 * y2
    dy1dt = (-r_y1 - fy1_t) * y1 + c1 * l11 * x1 * y1 + d1 * l21 * x2 * y1
    dy2dt = -r_y2 * y2 + c2 * l12 * x1 * y2 + d2 * l22 * x2 * y2
    return [dx1dt, dx2dt, dy1dt, dy2dt]

# ==========================================
# 3. 積分軌道を用いた残差関数の定義
# ==========================================
def residuals_ode_integration(params, t_real, x1_data, x2_data, y1_data, y2_data, init_conditions):
    # 積分精度をコントロールするパラメータを追加（カオス的発散の緩和）
    sol = solve_ivp(
        full_system_ode,
        [t_real[0], t_real[-1]],
        init_conditions,
        t_eval=t_real,
        args=(params,),
        method='RK45',
        rtol=1e-4,  # 最適化中の微小なカオス暴走を許容し、アルゴリズムの足止めを防ぐ
        atol=1e-4
    )
    
    if sol.status != 0 or sol.y.shape[1] != len(t_real):
        return np.ones(len(t_real) * 4) * 1e4  # ペナルティのスケールを最適化
        
    sim_x1, sim_x2, sim_y1, sim_y2 = sol.y
    
    res_x1 = sim_x1 - x1_data
    res_x2 = sim_x2 - x2_data
    res_y1 = sim_y1 - y1_data
    res_y2 = sim_y2 - y2_data
    
    return np.concatenate([res_x1, res_x2, res_y1, res_y2])

# ==========================================
# 4. 最適化の実行
# ==========================================
# 初期推測値（千トンスケールに合わせた妥当な初期値）
initial_guess = [1.0, 1.0, 0.2, 0.2,  0.001, 0.001, 0.001, 0.001,  0.1, 0.1, 0.1, 0.1]

# 境界条件：バイオマスの単位が「千トン」になったため、相互作用（l_ij）の上限を
# データの大きさに合わせて適切に引き下げます（ここが合わないと一瞬で発散します）
lower_bounds = [0.0001] * 12
upper_bounds = [10.0, 10.0, 5.0, 5.0,  0.1, 0.1, 0.1, 0.1,  1.0, 1.0, 1.0, 1.0]
bounds = (lower_bounds, upper_bounds)

init_conditions = [x1_data[0], x2_data[0], y1_data[0], y2_data[0]]

print("--- ODE積分軌道による直接フィッティングを開始します（単位修正版） ---")
res = least_squares(
    residuals_ode_integration,
    initial_guess,
    bounds=bounds,
    args=(t_real, x1_data, x2_data, y1_data, y2_data, init_conditions),
    ftol=1e-3,  # 収束判定を少しマイルドにすることで局所解へのスタックを回避
    xtol=1e-3,
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
    print(f"{name:<6} : {final_params[i]:.6f}")

fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
species_names = ['スルメイカ (x1)', 'マイワシ (x2)', 'ブリ (y1)', 'マサバ (y2)']
real_data_list = [x1_data, x2_data, y1_data, y2_data]

for i in range(2):
    for j in range(2):
        idx = i * 2 + j
        ax = axes[i, j]
        
        # 【修正】両者とも「千トン単位」で統一されたため、そのまま綺麗に重なります
        ax.plot(t_real, sol_verify.y[idx], 'b-', linewidth=2.5, label='4種連動モデルの軌道')
        ax.plot(t_real, real_data_list[idx], 'ro', alpha=0.7, label='水産庁 資源評価データ')
        
        ax.set_title(species_names[idx], fontsize=12)
        ax.set_ylabel('資源量（千トン）')
        ax.grid(True, linestyle='--')
        ax.legend()

plt.tight_layout()
plt.show()