import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os
import platform

# OSに応じた日本語フォントとマイナス記号の設定
if platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'Hiragino Sans'
elif platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Meiryo'
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. データの読み込みと「千トン」単位への統一
# ==========================================
def load_csv(filename):
    possible_paths = [
        filename,
        os.path.join('data', filename),
        os.path.join('..', 'data', filename),
        os.path.join(os.path.dirname(__file__), filename),
        os.path.join(os.path.dirname(__file__), 'data', filename),
        os.path.join(os.path.dirname(__file__), '..', 'data', filename),
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return pd.read_csv(path)
    raise FileNotFoundError(f"データファイルが見つかりません: {filename}")

df_sardine = load_csv('マイワシ時系列データ_資源量・漁獲量・漁獲係数 - マイワシ時系列データ_資源量・漁獲量・漁獲係数.csv')
df_anchovy = load_csv('カタクチイワシ太平洋時系列データ - カタクチイワシ太平洋時系列データ.csv')
df_yariika = load_csv('ヤリイカ太平洋時系列データ - ヤリイカ太平洋時系列データ.csv')
df_squid = load_csv('スルメイカ秋季 資源量・漁獲量時系列データ - スルメイカ秋季 資源量・漁獲量時系列データ.csv')

df_squid.rename(columns={'漁期年': '年'}, inplace=True)

df_merged = df_sardine[['年', '資源量（万トン）', '漁獲量（万トン）']].merge(
    df_anchovy[['年', '資源量（千トン）', '漁獲量（トン）']].rename(
        columns={'資源量（千トン）': '資源量_anchovy', '漁獲量（トン）': '漁獲量_anchovy'}
    ), on='年'
).merge(
    df_yariika[['年', '資源量（トン）', '漁獲量（トン）']].rename(
        columns={'資源量（トン）': '資源量_yariika', '漁獲量（トン）': '漁獲量_yariika'}
    ), on='年'
).merge(
    df_squid[['年', '資源量（千トン）', '漁獲量（千トン）']].rename(
        columns={'資源量（千トン）': '資源量_squid', '漁獲量（千トン）': '漁獲量_squid'}
    ), on='年'
)

df_clean = df_merged.dropna().sort_values('年').reset_index(drop=True)

# 資源量のスケール統一（すべて千トン単位）
x1_all = df_clean['資源量（万トン）'].values * 10.0
x2_all = df_clean['資源量_anchovy'].values
y1_all = df_clean['資源量_yariika'].values / 1000.0
y2_all = df_clean['資源量_squid'].values

# 漁獲量のスケール統一（すべて千トン単位）
c_x1_all = df_clean['漁獲量（万トン）'].values * 10.0
c_x2_all = df_clean['漁獲量_anchovy'].values / 1000.0
c_y1_all = df_clean['漁獲量_yariika'].values / 1000.0
c_y2_all = df_clean['漁獲量_squid'].values

# 漁獲係数の計算
f_x1_all = np.clip(c_x1_all / x1_all, 0.0, 0.95)
f_x2_all = np.clip(c_x2_all / x2_all, 0.0, 0.95)
f_y1_all = np.clip(c_y1_all / y1_all, 0.0, 0.95)
f_y2_all = np.clip(c_y2_all / y2_all, 0.0, 0.95)

# レジーム期間の定義（データからの最長期間：1997年〜2024年の範囲内）
idx_nlm = (df_clean['年'] >= 2006) & (df_clean['年'] <= 2016)
idx_lm = (df_clean['年'] >= 2017) & (df_clean['年'] <= 2024)

# 資源量の平均値（正規化用、全期間平均）
mean_x1 = np.mean(x1_all)
mean_x2 = np.mean(x2_all)
mean_y1 = np.mean(y1_all)
mean_y2 = np.mean(y2_all)

# 正規化された資源量データ (平均値が1.0になる)
x1_norm = x1_all / mean_x1
x2_norm = x2_all / mean_x2
y1_norm = y1_all / mean_y1
y2_norm = y2_all / mean_y2

def estimate_with_capacity(years, x1_n, x2_n, y1_n, y2_n, fx1, fx2, fy1, fy2, mean_x1, mean_x2, mean_y1, mean_y2, regime_name):
    t_rel = (years - years.min()).astype(float)
    
    # 漁獲圧の補間
    fx1_i = interp1d(t_rel, fx1, kind='linear', fill_value="extrapolate")
    fx2_i = interp1d(t_rel, fx2, kind='linear', fill_value="extrapolate")
    fy1_i = interp1d(t_rel, fy1, kind='linear', fill_value="extrapolate")
    fy2_i = interp1d(t_rel, fy2, kind='linear', fill_value="extrapolate")
    
    # 固定の自然死亡率
    r_y1, r_y2 = 0.3, 0.4
    
    def ode_fun(t, state, params):
        x1_v, x2_v, y1_v, y2_v = state
        # 10個の正規化パラメータを受け取る (増加率2個 + 捕食効率4個 + 変換効率4個)
        r_x1, r_x2, L11, L12, L21, L22, C1, D1, C2, D2 = params
        
        x1_v = max(1e-5, x1_v); x2_v = max(1e-5, x2_v); y1_v = max(1e-5, y1_v); y2_v = max(1e-5, y2_v)
        
        # 正規化された状態方程式
        dx1dt = (r_x1 - fx1_i(t)) * x1_v - L11 * x1_v * y1_v - L12 * x1_v * y2_v
        dx2dt = (r_x2 - fx2_i(t)) * x2_v - L21 * x2_v * y1_v - L22 * x2_v * y2_v
        dy1dt = (-r_y1 - fy1_i(t)) * y1_v + C1 * L11 * x1_v * y1_v + D1 * L21 * x2_v * y1_v
        dy2dt = (-r_y2 - fy2_i(t)) * y2_v + C2 * L12 * x1_v * y2_v + D2 * L22 * x2_v * y2_v
        return [dx1dt, dx2dt, dy1dt, dy2dt]
        
    def res_fun(params):
        sol = solve_ivp(ode_fun, [t_rel[0], t_rel[-1]], [x1_n[0], x2_n[0], y1_n[0], y2_n[0]], t_eval=t_rel, args=(params,))
        if sol.status != 0 or sol.y.shape[1] != len(t_rel):
            return np.ones(len(t_rel) * 4) * 1e5
        # 対数差を返すことで相対誤差評価にする
        log_sol = np.log(np.clip(sol.y, 1e-5, None))
        return np.concatenate([
            log_sol[0] - np.log(x1_n),
            log_sol[1] - np.log(x2_n),
            log_sol[2] - np.log(y1_n),
            log_sol[3] - np.log(y2_n)
        ])
        
    # 初期推測値 (正規化空間): [r_x1, r_x2, L11, L12, L21, L22, C1, D1, C2, D2]
    guess = [1.5, 1.5, 0.1, 0.1, 0.1, 0.1, 0.15, 0.15, 0.15, 0.15]
    
    # 境界条件の設定 (正規化パラメータ用)
    lower = [0.1,  0.1,  1e-4, 1e-4, 1e-4, 1e-4, 0.001, 0.001, 0.001, 0.001]
    upper = [5.0,  5.0,  5.0,  5.0,  5.0,  5.0,  10.0,  10.0,  10.0,  10.0]
    
    print(f"--- 収容力なしモデル最適化開始 (正規化): {regime_name} ---")
    res = least_squares(res_fun, guess, bounds=(lower, upper), verbose=1)
    
    # 正規化空間での最終軌道
    final_sol = solve_ivp(ode_fun, [t_rel[0], t_rel[-1]], [x1_n[0], x2_n[0], y1_n[0], y2_n[0]], t_eval=t_rel, args=(res.x,))
    
    # 元の物理スケール（千トン）に逆変換
    traj_absolute = np.zeros_like(final_sol.y)
    traj_absolute[0] = final_sol.y[0] * mean_x1
    traj_absolute[1] = final_sol.y[1] * mean_x2
    traj_absolute[2] = final_sol.y[2] * mean_y1
    traj_absolute[3] = final_sol.y[3] * mean_y2
    
    # パラメータを元のスケールに逆変換して戻す
    r_x1_est, r_x2_est, L11_est, L12_est, L21_est, L22_est, C1_est, D1_est, C2_est, D2_est = res.x
    
    l11_est = L11_est / mean_y1
    l12_est = L12_est / mean_y2
    l21_est = L21_est / mean_y1
    l22_est = L22_est / mean_y2
    
    c1_est = C1_est * mean_y1 / mean_x1
    d1_est = D1_est * mean_y1 / mean_x2
    c2_est = C2_est * mean_y2 / mean_x1
    d2_est = D2_est * mean_y2 / mean_x2
    
    params_absolute = [
        r_x1_est, r_x2_est,
        l11_est, l12_est, l21_est, l22_est,
        c1_est, d1_est, c2_est, d2_est
    ]
    
    return params_absolute, traj_absolute

# NLM・LMそれぞれの実行 (正規化されたデータを使用)
params_nlm, traj_nlm = estimate_with_capacity(
    df_clean.loc[idx_nlm, '年'].values,
    x1_norm[idx_nlm], x2_norm[idx_nlm], y1_norm[idx_nlm], y2_norm[idx_nlm],
    f_x1_all[idx_nlm], f_x2_all[idx_nlm], f_y1_all[idx_nlm], f_y2_all[idx_nlm],
    mean_x1, mean_x2, mean_y1, mean_y2, "非大蛇行期 (NLM)"
)

params_lm, traj_lm = estimate_with_capacity(
    df_clean.loc[idx_lm, '年'].values,
    x1_norm[idx_lm], x2_norm[idx_lm], y1_norm[idx_lm], y2_norm[idx_lm],
    f_x1_all[idx_lm], f_x2_all[idx_lm], f_y1_all[idx_lm], f_y2_all[idx_lm],
    mean_x1, mean_x2, mean_y1, mean_y2, "黒潮大蛇行期 (LM)"
)

# ==========================================
# 4. 結果の出力とプロット
# ==========================================
print("\n" + "="*50)
print("【収容力なし・c,d推定モデル 推定結果比較（元スケール換算値）】")
print("="*50)
p_names = [
    "r_x1 (マイワシ内的増加率)          ", "r_x2 (カタクチイワシ内的増加率)    ", 
    "l11  (ヤリイカによるマイワシ捕食効率)", "l12  (スルメイカによるマイワシ捕食効率)", 
    "l21  (ヤリイカによるカタクチ捕食効率)", "l22  (スルメイカによるカタクチ捕食効率)",
    "c1   (ヤリイカのマイワシ変換効率)    ", "d1   (ヤリイカのカタクチ変換効率)    ",
    "c2   (スルメイカのマイワシ変換効率)  ", "d2   (スルメイカのカタクチ変換効率)  "
]
for name, p_n, p_l in zip(p_names, params_nlm, params_lm):
    print(f"{name} | NLM: {p_n:.6f} | LM: {p_l:.6f}")

# 2列×4行 of graph generation
fig, axes = plt.subplots(4, 2, figsize=(14, 16))
species_labels = ['マイワシ (x1)', 'カタクチイワシ (x2)', 'ヤリイカ (y1)', 'スルメイカ (y2)']
years_nlm = df_clean.loc[idx_nlm, '年'].values
years_lm = df_clean.loc[idx_lm, '年'].values

for idx in range(4):
    # NLMプロット
    axes[idx, 0].plot(years_nlm, traj_nlm[idx], 'b-', linewidth=2.5, label='収容力なしモデル軌道')
    axes[idx, 0].plot(years_nlm, [x1_all[idx_nlm], x2_all[idx_nlm], y1_all[idx_nlm], y2_all[idx_nlm]][idx], 'ro', alpha=0.7, label='実データ')
    axes[idx, 0].set_title(f"NLM期 (収容力なし・c,d変数): {species_labels[idx]}")
    axes[idx, 0].set_ylabel('資源量（千トン）')
    axes[idx, 0].grid(True, linestyle='--')
    axes[idx, 0].legend()
    
    # LMプロット
    axes[idx, 1].plot(years_lm, traj_lm[idx], 'g-', linewidth=2.5, label='収容力なしモデル軌道')
    axes[idx, 1].plot(years_lm, [x1_all[idx_lm], x2_all[idx_lm], y1_all[idx_lm], y2_all[idx_lm]][idx], 'ro', alpha=0.7, label='実データ')
    axes[idx, 1].set_title(f"LM期 (収容力なし・c,d変数): {species_labels[idx]}")
    axes[idx, 1].set_ylabel('資源量（千トン）')
    axes[idx, 1].grid(True, linestyle='--')
    axes[idx, 1].legend()

plt.tight_layout()
output_img_path = os.path.join(os.path.dirname(__file__), 'kuroshio_capacity_fitting.png')
plt.savefig(output_img_path, dpi=150)
plt.close()
print(f"\nフィッティング画像が '{output_img_path}' として保存されました。")
