import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os

# 日本語表示のためのフォント設定
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Hiragino Sans', 'Heiti TC']

# ==========================================
# 1. データの読み込みと「千トン」単位への統一
# ==========================================
df_squid = pd.read_csv('data/スルメイカ秋季 資源量・漁獲量時系列データ - スルメイカ秋季 資源量・漁獲量時系列データ.csv')
df_yellowtail = pd.read_csv('data/ブリ時系列データ_資源量・漁獲量・漁獲係数 - ブリ時系列データ_資源量・漁獲量・漁獲係数.csv')
df_sardine = pd.read_csv('data/マイワシ時系列データ_資源量・漁獲量・漁獲係数 - マイワシ時系列データ_資源量・漁獲量・漁獲係数.csv')
df_mackerel = pd.read_csv('data/マサバ時系列データ_資源量・漁獲量・漁獲係数 - マサバ時系列データ_資源量・漁獲量・漁獲係数.csv')

df_mackerel.rename(columns={'漁期年': '年'}, inplace=True)
df_squid.rename(columns={'漁期年': '年'}, inplace=True)

df_merged = df_squid[['年', '資源量（千トン）', '漁獲量（千トン）', '黒潮大蛇行の有無']].merge(
    df_sardine[['年', '資源量（万トン）', '漁獲量（万トン）']], on='年'
).merge(
    df_yellowtail[['年', '資源量（トン）', '漁獲量（トン）']], on='年'
).merge(
    df_mackerel[['年', '資源量（万トン）', '漁獲量（万トン）']], on='年', suffixes=('_sardine', '_mackerel')
)

df_clean = df_merged.dropna().sort_values('年').reset_index(drop=True)

x1_all = df_clean['資源量（千トン）'].values
x2_all = df_clean['資源量（万トン）_sardine'].values * 10.0
y1_all = df_clean['資源量（トン）'].values / 1000.0
y2_all = df_clean['資源量（万トン）_mackerel'].values * 10.0

f_x1_all = np.clip(df_clean['漁獲量（千トン）'].values / x1_all, 0.0, 0.95)
f_x2_all = np.clip((df_clean['漁獲量（万トン）_sardine'].values * 10.0) / x2_all, 0.0, 0.95)
f_y1_all = np.clip((df_clean['漁獲量（トン）'].values / 1000.0) / y1_all, 0.0, 0.95)
f_y2_all = np.clip((df_clean['漁獲量（万トン）_mackerel'].values * 10.0) / y2_all, 0.0, 0.95)

idx_nlm = (df_clean['年'] >= 2006) & (df_clean['年'] <= 2016)
idx_lm = (df_clean['年'] >= 2017) & (df_clean['年'] <= 2024)

def estimate_no_capacity(years, x1, x2, y1, y2, fx1, fx2, fy1, fy2, regime_name):
    t_rel = (years - years.min()).astype(float)
    
    fx1_i = interp1d(t_rel, fx1, kind='linear', fill_value="extrapolate")
    fx2_i = interp1d(t_rel, fx2, kind='linear', fill_value="extrapolate")
    fy1_i = interp1d(t_rel, fy1, kind='linear', fill_value="extrapolate")
    fy2_i = interp1d(t_rel, fy2, kind='linear', fill_value="extrapolate")
    
    # 水産庁および生態学基準に準拠した固定値
    r_y1, r_y2 = 0.3, 0.4
    c1, d1, c2, d2 = 0.15, 0.15, 0.15, 0.15
    
    def ode_fun(t, state, params):
        x1_v, x2_v, y1_v, y2_v = state
        # alpha（種内競争）を排除した純粋なロトカ・ヴォルテラ 6変数
        r_x1, r_x2, l11, l12, l21, l22 = params
        
        x1_v = max(1e-5, x1_v); x2_v = max(1e-5, x2_v); y1_v = max(1e-5, y1_v); y2_v = max(1e-5, y2_v)
        
        # 環境収容力（alpha）を0に固定した微分方程式システム
        dx1dt = (r_x1 - fx1_i(t)) * x1_v - l11 * x1_v * y1_v - l12 * x1_v * y2_v
        dx2dt = (r_x2 - fx2_i(t)) * x2_v - l21 * x2_v * y1_v - l22 * x2_v * y2_v
        dy1dt = (-r_y1 - fy1_i(t)) * y1_v + c1 * l11 * x1_v * y1_v + d1 * l21 * x2_v * y1_v
        dy2dt = (-r_y2 - fy2_i(t)) * y2_v + c2 * l12 * x1_v * y2_v + d2 * l22 * x2_v * y2_v
        return [dx1dt, dx2dt, dy1dt, dy2dt]
        
    def res_fun(params):
        sol = solve_ivp(ode_fun, [t_rel[0], t_rel[-1]], [x1[0], x2[0], y1[0], y2[0]], t_eval=t_rel, args=(params,))
        if sol.status != 0 or sol.y.shape[1] != len(t_rel):
            return np.ones(len(t_rel) * 4) * 1e5
        return np.concatenate([sol.y[0] - x1, sol.y[1] - x2, sol.y[2] - y1, sol.y[3] - y2])
        
    guess = [1.5, 1.5, 0.001, 0.001, 0.001, 0.001]
    lower = [0.1, 0.1, 1e-6, 1e-6, 1e-6, 1e-6]
    upper = [5.0, 5.0, 0.05, 0.05, 0.05, 0.05]
    
    print(f"--- 環境収容力なしver 推定開始: {regime_name} ---")
    res = least_squares(res_fun, guess, bounds=(lower, upper), verbose=1)
    
    final_sol = solve_ivp(ode_fun, [t_rel[0], t_rel[-1]], [x1[0], x2[0], y1[0], y2[0]], t_eval=t_rel, args=(res.x,))
    return res.x, final_sol.y

# 推定の実行
params_nlm, traj_nlm = estimate_no_capacity(
    df_clean.loc[idx_nlm, '年'].values, x1_all[idx_nlm], x2_all[idx_nlm], y1_all[idx_nlm], y2_all[idx_nlm],
    f_x1_all[idx_nlm], f_x2_all[idx_nlm], f_y1_all[idx_nlm], f_y2_all[idx_nlm], "非大蛇行期 (NLM)"
)

params_lm, traj_lm = estimate_no_capacity(
    df_clean.loc[idx_lm, '年'].values, x1_all[idx_lm], x2_all[idx_lm], y1_all[idx_lm], y2_all[idx_lm],
    f_x1_all[idx_lm], f_x2_all[idx_lm], f_y1_all[idx_lm], f_y2_all[idx_lm], "黒潮大蛇行期 (LM)"
)

# ==========================================
# 4. 結果の出力とプロット
# ==========================================
print("\n" + "="*50)
print("【環境収容力なしver 推定結果比較】")
print("="*50)
p_names = [
    "r_x1 (スルメイカ内的増加率)", "r_x2 (マイワシ内的増加率)  ", 
    "l11  (ブリ->イカ捕食効率)  ", "l12  (サバ->イカ捕食効率)  ", 
    "l21  (ブリ->イワシ捕食効率)", "l22  (サバ->イワシ捕食効率)"
]
for name, p_n, p_l in zip(p_names, params_nlm, params_lm):
    print(f"{name} | NLM: {p_n:.6f} | LM: {p_l:.6f}")

fig, axes = plt.subplots(4, 2, figsize=(14, 16))
species_labels = ['スルメイカ (x1)', 'マイワシ (x2)', 'ブリ (y1)', 'マサバ (y2)']
years_nlm = df_clean.loc[idx_nlm, '年'].values
years_lm = df_clean.loc[idx_lm, '年'].values

for idx in range(4):
    axes[idx, 0].plot(years_nlm, traj_nlm[idx], 'b-', linewidth=2.5, label='収容力なしモデル軌道')
    axes[idx, 0].plot(years_nlm, [x1_all[idx_nlm], x2_all[idx_nlm], y1_all[idx_nlm], y2_all[idx_nlm]][idx], 'ro', alpha=0.7, label='実データ')
    axes[idx, 0].set_title(f"NLM期 (環境収容力なし): {species_labels[idx]}")
    axes[idx, 0].set_ylabel('資源量（千トン）')
    axes[idx, 0].grid(True, linestyle='--')
    axes[idx, 0].legend()
    
    axes[idx, 1].plot(years_lm, traj_lm[idx], 'g-', linewidth=2.5, label='収容力なしモデル軌道')
    axes[idx, 1].plot(years_lm, [x1_all[idx_lm], x2_all[idx_lm], y1_all[idx_lm], y2_all[idx_lm]][idx], 'ro', alpha=0.7, label='実データ')
    axes[idx, 1].set_title(f"LM期 (環境収容力なし): {species_labels[idx]}")
    axes[idx, 1].set_ylabel('資源量（千トン）')
    axes[idx, 1].grid(True, linestyle='--')
    axes[idx, 1].legend()

plt.tight_layout()
plt.savefig('kuroshio_no_capacity_fitting.png', dpi=150)
plt.close()
print("\n対比グラフが 'kuroshio_no_capacity_fitting.png' としてカレントディレクトリに保存されました。")