import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import itertools

# 日本語表示のためのフォント設定 (macOS用)
plt.rcParams['font.family'] = 'Hiragino Sans'

# ==========================================
# 1. CSVデータ読み込みと前処理
# ==========================================
csv_path = '/Users/fukuokashunya/MSY/スルメイカ秋季 資源量・漁獲量時系列データ - スルメイカ秋季 資源量・漁獲量時系列データ.csv'
df = pd.read_csv(csv_path)
df_clean = df.dropna(subset=['資源量（千トン）']).copy()

t_real = (df_clean['漁期年'] - df_clean['漁期年'].min()).values.astype(float)
scale_factor = 1000.0
x1_data = df_clean['資源量（千トン）'].values / scale_factor

raw_fishing_rates = df_clean['漁獲量（千トン）'].values / df_clean['資源量（千トン）'].values
fishing_rates = np.clip(raw_fishing_rates, 0.0, 0.95)
f_x1_interp = interp1d(t_real, fishing_rates, kind='linear', fill_value="extrapolate")

# ==========================================
# 2. 純粋な2部グラフ型4種系ODEモデル（制約・改変一切なし）
# ==========================================
def full_bi_partite_lv_ode(t, state, params, f_x1_func):
    x1, x2, y1, y2 = state
    r_x1, r_x2, r_y1, r_y2, l11, l12, l21, l22, c1, c2, d1, d2 = params
    
    f_x1 = f_x1_func(t)
    dx1dt = (r_x1 - f_x1) * x1 - l11 * x1 * y1 - l12 * x1 * y2
    dx2dt = r_x2 * x2 - l21 * x2 * y1 - l22 * x2 * y2
    dy1dt = -r_y1 * y1 + c1 * l11 * x1 * y1 + d1 * l21 * x2 * y1
    dy2dt = -r_y2 * y2 + c2 * l12 * x1 * y2 + d2 * l22 * x2 * y2
    return [dx1dt, dx2dt, dy1dt, dy2dt]

# ==========================================
# 3. モンテカルロ・全空間爆撃シミュレーション
# ==========================================
print("--- 局所最適解を粉砕する全空間全探索を開始します ---")

# 試行回数の設定（Cursorのパワーをフルに使ってまずは50000パターン爆撃します）
num_trials = 50000

best_score = np.inf
best_params = None
best_inits = None
best_sol = None

# 再現性のためのシード固定
np.random.seed(42)

# カオス周期を生み出すための過激な探索宇宙の定義（ランダムサンプリングの範囲）
# 福岡さんの仮説「必ずどこかにある」を検証するため、広い空間から一斉にサンプリングします
for trial in range(num_trials):
    if trial % 5000 == 0:
        print(f"探索進捗: {trial}/{num_trials} パターン検証中...")
        
    # 12個のパラメータをランダム生成
    r_x1 = np.random.uniform(2.0, 15.0)
    r_x2 = np.random.uniform(2.0, 15.0)
    r_y1 = np.random.uniform(1.0, 10.0)
    r_y2 = np.random.uniform(1.0, 10.0)
    
    l11 = np.random.uniform(0.05, 3.0)
    l12 = np.random.uniform(0.05, 3.0)
    l21 = np.random.uniform(0.05, 3.0)
    l22 = np.random.uniform(0.05, 3.0)
    
    c1 = np.random.uniform(0.1, 1.0)
    c2 = np.random.uniform(0.1, 1.0)
    d1 = np.random.uniform(0.1, 1.0)
    d2 = np.random.uniform(0.1, 1.0)
    
    params_candidate = [r_x1, r_x2, r_y1, r_y2, l11, l12, l21, l22, c1, c2, d1, d2]
    
    # 3つの隠れた初期値をランダム生成（0.15〜3.0の健全な範囲）
    x2_0 = np.random.uniform(0.15, 3.0)
    y1_0 = np.random.uniform(0.15, 3.0)
    y2_0 = np.random.uniform(0.15, 3.0)
    inits_candidate = [x1_data[0], x2_0, y1_0, y2_0]
    
    # シミュレーション実行
    sol = solve_ivp(
        full_bi_partite_lv_ode,
        [t_real[0], t_real[-1]],
        inits_candidate,
        t_eval=t_real,
        args=(params_candidate, f_x1_interp),
        method='RK45'
    )
    
    # ソルバーが途中でクラッシュした、またはマイナスに突入した場合は即却下
    if sol.status != 0 or sol.y.shape[1] != len(t_real) or np.any(sol.y < 0):
        continue
        
    x1_sol, x2_sol, y1_sol, y2_sol = sol.y
    
    # -------------------------------------------------------------
    # 【核心】福岡さんの生存制約フィルター
    # -------------------------------------------------------------
    # 15年間の全期間において、他3種のいずれか1つでも「0.05（絶滅ライン）」を下回ったら
    # そのパラメータの組み合わせは、どれだけイカの波が合っていても『強制ゴミ箱行き』にする
    if np.any(x2_sol < 0.05) or np.any(y1_sol < 0.05) or np.any(y2_sol < 0.05):
        continue
        
    # すべての種が生き残った場合のみ、スルメイカのエラー（残差平方和）を計算
    score = np.sum((x1_sol - x1_data) ** 2)
    
    # 暫定ベストの更新
    if score < best_score:
        best_score = score
        best_params = params_candidate
        best_inits = inits_candidate
        best_sol = sol
        print(f"-> 【神解候補を発見！】 暫定最小エラー RSS: {best_score:.6f} (他3種生存クリア)")

# ==========================================
# 4. 探索結果の検証とプロット
# ==========================================
if best_sol is None:
    print("\n[警告] 設定された探索範囲内では、他種が一度も絶滅せずに生存できる組み合わせが見つかりませんでした。")
    print("探索回数 `num_trials` を増やすか、パラメータの探索範囲をさらに広げる必要があります。")
else:
    print("\n=== 全空間爆撃による『真の共存軌道』の抽出に成功しました！ ===")
    param_names = ["r_x1", "r_x2", "r_y1", "r_y2", "l11", "l12", "l21", "l22", "c1", "c2", "d1", "d2"]
    for name, val in zip(param_names, best_params):
        print(f"{name:<6} : {val:.4f}")
    print(f"逆算された初期値 x2(0): {best_inits[1]:.4f}, y1(0): {best_inits[2]:.4f}, y2(0): {best_inits[3]:.4f}")
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    species_labels = [['x1 (スルメイカ資源量)', 'x2 (競合種 - 潜在変数)'], 
                      ['y1 (捕食者1 - 潜在変数)', 'y2 (捕食者2 - 潜在変数)']]

    for i in range(2):
        for j in range(2):
            idx = i * 2 + j
            ax = axes[i, j]
            
            ax.plot(t_real, best_sol.y[idx] * scale_factor, 'b-', linewidth=2, label='全探索で見つけ出した真の軌道')
            if idx == 0:
                ax.plot(t_real, df_clean['資源量（千トン）'].values, 'ro', alpha=0.6, label='実際の観測データ (csv)')
                
            ax.set_title(species_labels[i][j])
            ax.set_ylabel('資源量（千トン）' if idx < 2 else '個体数/資源量')
            ax.grid(True)
            ax.legend()

    plt.tight_layout()
    plt.show()