import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# 日本語表示のためのフォント設定
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Hiragino Sans', 'Heiti TC']

# ==========================================
# 1. データの読み込みと「千トン」単位への統一
# ==========================================
df_sardine = pd.read_csv('マイワシ時系列データ_資源量・漁獲量・漁獲係数 - マイワシ時系列データ_資源量・漁獲量・漁獲係数.csv')
df_yellowtail = pd.read_csv('ブリ時系列データ_資源量・漁獲量・漁獲係数 - ブリ時系列データ_資源量・漁獲量・漁獲係数.csv')
df_mackerel = pd.read_csv('マサバ時系列データ_資源量・漁獲量・漁獲係数 - マサバ時系列データ_資源量・漁獲量・漁獲係数.csv')
df_squid = pd.read_csv('スルメイカ秋季 資源量・漁獲量時系列データ - スルメイカ秋季 資源量・漁獲量時系列データ.csv')

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

# 資源量のスケール統一（すべて千トン単位）
x1_all = df_clean['資源量（千トン）'].values
x2_all = df_clean['資源量（万トン）_sardine'].values * 10.0
y1_all = df_clean['資源量（トン）'].values / 1000.0
y2_all = df_clean['資源量（万トン）_mackerel'].values * 10.0

# 漁獲係数の計算
f_x1_all = np.clip(df_clean['漁獲量（千トン）'].values / x1_all, 0.0, 0.95)
f_x2_all = np.clip((df_clean['漁獲量（万トン）_sardine'].values * 10.0) / x2_all, 0.0, 0.95)
f_y1_all = np.clip((df_clean['漁獲量（トン）'].values / 1000.0) / y1_all, 0.0, 0.95)
f_y2_all = np.clip((df_clean['漁獲量（万トン）_mackerel'].values * 10.0) / y2_all, 0.0, 0.95)

# ==========================================
# 2. 最長レジーム（NLM / LM）の自動切り出し
# ==========================================
idx_nlm = (df_clean['年'] >= 2006) & (df_clean['年'] <= 2016)
idx_lm = (df_clean['年'] >= 2017) & (df_clean['年'] <= 2024)

def estimate_regime_parameters(years, x1, x2, y1, y2, fx1, fx2, fy1, fy2, regime_name):
    t_rel = (years - years.min()).astype(float)
    
    # 漁獲圧の補間関数
    fx1_interp = interp1d(t_rel, fx1, kind='linear', fill_value="extrapolate")
    fx2_interp = interp1d(t_rel, fx2, kind='linear', fill_value="extrapolate")
    fy1_interp = interp1d(t_rel, fy1, kind='linear', fill_value="extrapolate")
    fy2_interp = interp1d(t_rel, fy2, kind='linear', fill_value="extrapolate")
    
    # 固定パラメータ（捕食者死亡率：水産庁基準値、変換効率：生態学標準値）
    r_y1, r_y2 = 0.3, 0.4
    c1, d1, c2, d2 = 0.15, 0.15, 0.15, 0.15
    
    def full_system_ode(t, state, params):
        x1_v, x2_v, y1_v, y2_v = state
        r_x1, r_x2, l11, l12, l21, l22 = params
        
        # 絶滅の奈落を防ぐ物理ガード
        x1_v = max(1e-5, x1_v); x2_v = max(1e-5, x2_v); y1_v = max(1e-5, y1_v); y2_v = max(1e-5, y2_v)
        
        # 被食者には単独死亡項を置かず、すべて相互作用項（l_ij）だけで減少を表現
        dx1dt = (r_x1 - fx1_interp(t)) * x1_v - l11 * x1_v * y1_v - l12 * x1_v * y2_v
        dx2dt = (r_x2 - fx2_interp(t)) * x2_v - l21 * x2_v * y1_v - l22 * x2_v * y2_v
        dy1dt = (-r_y1 - fy1_interp(t)) * y1_v + c1 * l11 * x1_v * y1_v + d1 * l21 * x2_v * y1_v
        dy2dt = (-r_y2 - fy2_interp(t)) * y2_v + c2 * l12 * x1_v * y2_v + d2 * l22 * x2_v * y2_v
        return [dx1dt, dx2dt, dy1dt, dy2dt]
        
    def residuals_function(params):
        sol = solve_ivp(
            full_system_ode, 
            [t_rel[0], t_rel[-1]], 
            [x1[0], x2[0], y1[0], y2[0]], 
            t_eval=t_rel, 
            args=(params,),
            method='RK45'
        )
        if sol.status != 0 or sol.y.shape[1] != len(t_rel):
            return np.ones(len(t_rel) * 4) * 1e5
        return np.concatenate([sol.y[0] - x1, sol.y[1] - x2, sol.y[2] - y1, sol.y[3] - y2])
        
    # 初期推測値: [r_x1, r_x2, l11, l12, l21, l22]
    initial_guess = [1.5, 1.5, 0.001, 0.001, 0.001, 0.001]
    lower_bounds = [0.1, 0.1, 1e-6, 1e-6, 1e-6, 1e-6]
    upper_bounds = [5.0, 5.0, 0.05, 0.05, 0.05, 0.05]
    
    print(f"--- 最得化実行中: {regime_name} レジーム ---")
    res = least_squares(residuals_function, initial_guess, bounds=(lower_bounds, upper_bounds), verbose=1)
    
    # 最適化されたパラメータによる最終軌道の再計算
    final_sol = solve_ivp(
        full_system_ode, 
        [t_rel[0], t_rel[-1]], 
        [x1[0], x2[0], y1[0], y2[0]], 
        t_eval=t_rel, 
        args=(res.x,),
        method='RK45'
    )
    return res.x, final_sol.y

# 各期間での実行
params_nlm, trajectory_nlm = estimate_regime_parameters(
    df_clean.loc[idx_nlm, '年'].values, x1_all[idx_nlm], x2_all[idx_nlm], y1_all[idx_nlm], y2_all[idx_nlm],
    f_x1_all[idx_nlm], f_x2_all[idx_nlm], f_y1_all[idx_nlm], f_y2_all[idx_nlm], "非大蛇行期 (NLM)"
)

params_lm, trajectory_lm = estimate_regime_parameters(
    df_clean.loc[idx_lm, '年'].values, x1_all[idx_lm], x2_all[idx_lm], y1_all[idx_lm], y2_all[idx_lm],
    f_x1_all[idx_lm], f_x2_all[idx_lm], f_y1_all[idx_lm], f_y2_all[idx_lm], "黒潮大蛇行期 (LM)"
)

# ==========================================
# 3. 結果の出力とプロットの保存
# ==========================================
print("\n" + "="*40)
print("【推定結果の比較】")
print("="*40)
p_names = ["r_x1 (スルメイカ増加率)", "r_x2 (マイワシ増加率)  ", "l11  (ブリ->イカ捕食)  ", "l12  (サバ->イカ捕食)  ", "l21  (ブリ->イワシ捕食)", "l22  (サバ->イワシ捕食)"]
for name, p_nlm, p_lm in zip(p_names, params_nlm, params_lm):
    print(f"{name} | NLM: {p_nlm:.6f} | LM: {p_lm:.6f}")

# グラフ作成
fig, axes = plt.subplots(4, 2, figsize=(14, 16), sharex=False)
species_labels = ['スルメイカ (x1)', 'マイワシ (x2)', 'ブリ (y1)', 'マサバ (y2)']
years_nlm = df_clean.loc[idx_nlm, '年'].values
years_lm = df_clean.loc[idx_lm, '年'].values

for idx in range(4):
    # 左列：NLM
    ax_n = axes[idx, 0]
    ax_n.plot(years_nlm, trajectory_nlm[idx], 'b-', linewidth=2.5, label='モデル推定軌道')
    ax_n.plot(years_nlm, [x1_all[idx_nlm], x2_all[idx_nlm], y1_all[idx_nlm], y2_all[idx_nlm]][idx], 'ro', alpha=0.7, label='資源量実データ')
    ax_n.set_title(f"非大蛇行期 (NLM): {species_labels[idx]}")
    ax_n.set_ylabel('資源量（千トン）')
    ax_n.grid(True, linestyle='--')
    ax_n.legend()
    
    # 右列：LM
    ax_l = axes[idx, 1]
    ax_l.plot(years_lm, trajectory_lm[idx], 'g-', linewidth=2.5, label='モデル推定軌道')
    ax_l.plot(years_lm, [x1_all[idx_lm], x2_all[idx_lm], y1_all[idx_lm], y2_all[idx_lm]][idx], 'ro', alpha=0.7, label='資源量実データ')
    ax_l.set_title(f"黒潮大蛇行期 (LM): {species_labels[idx]}")
    ax_l.set_ylabel('資源量（千トン）')
    ax_l.grid(True, linestyle='--')
    ax_l.legend()

plt.tight_layout()
plt.savefig('kuroshio_regime_fitting.png', dpi=150)
plt.close()
print("\nフィッティンググラフを 'kuroshio_regime_fitting.png' として保存しました。")