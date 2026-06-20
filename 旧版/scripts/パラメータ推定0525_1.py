import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor  # 並列処理ライブラリ

# 日本語表示のためのフォント設定 (macOS用)
plt.rcParams['font.family'] = 'Hiragino Sans'

# ==========================================
# 1. データ前処理（共通）
# ==========================================
csv_path = '/Users/fukuokashunya/MSY/スルメイカ秋季 資源量・漁獲量時系列データ - スルメイカ秋季 資源量・漁獲量時系列データ.csv'
df = pd.read_csv(csv_path)
df_clean = df.dropna(subset=['資源量（千トン）']).copy()

t_real = (df_clean['漁期年'] - df_clean['漁期年'].min()).values.astype(float)
scale_factor = 1000.0
x1_data = df_clean['資源量（千トン）'].values / scale_factor

raw_fishing_rates = df_clean['漁獲量（千トン）'].values / df_clean['資源量（千トン）'].values
fishing_rates = np.clip(raw_fishing_rates, 0.0, 0.95)

# ==========================================
# 2. 微分方程式の定義
# ==========================================
def full_bi_partite_lv_ode(t, state, params, fishing_rates_fixed, t_real_fixed):
    x1, x2, y1, y2 = state
    r_x1, r_x2, r_y1, r_y2, l11, l12, l21, l22, c1, c2, d1, d2 = params
    
    # 補間関数を内部で毎回作ると重いため、高速なnp.interpに差し替え
    f_x1 = np.interp(t, t_real_fixed, fishing_rates_fixed)
    
    dx1dt = (r_x1 - f_x1) * x1 - l11 * x1 * y1 - l12 * x1 * y2
    dx2dt = r_x2 * x2 - l21 * x2 * y1 - l22 * x2 * y2
    dy1dt = -r_y1 * y1 + c1 * l11 * x1 * y1 + d1 * l21 * x2 * y1
    dy2dt = -r_y2 * y2 + c2 * l12 * x1 * y2 + d2 * l22 * x2 * y2
    return [dx1dt, dx2dt, dy1dt, dy2dt]

# ==========================================
# 3. 1回分のシミュレーションを評価する関数（並列化用）
# ==========================================
def evaluate_one_pattern(seed):
    # 各プロセスで異なるシードを設定
    np.random.seed(seed)
    
    # パラメータのランダム生成
    r_x1 = np.random.uniform(2.0, 15.0)
    r_x2 = np.random.uniform(2.0, 15.0)
    r_y1 = np.random.uniform(1.0, 10.0)
    r_y2 = np.random.uniform(1.0, 10.0)
    l11, l12, l21, l22 = np.random.uniform(0.05, 3.0, 4)
    c1, c2, d1, d2 = np.random.uniform(0.1, 1.0, 4)
    params = [r_x1, r_x2, r_y1, r_y2, l11, l12, l21, l22, c1, c2, d1, d2]
    
    # 初期値のランダム生成
    inits = [x1_data[0], np.random.uniform(0.15, 3.0), np.random.uniform(0.15, 3.0), np.random.uniform(0.15, 3.0)]
    
    sol = solve_ivp(
        full_bi_partite_lv_ode, [t_real[0], t_real[-1]], inits, 
        t_eval=t_real, args=(params, fishing_rates, t_real), method='RK45'
    )
    
    if sol.status != 0 or sol.y.shape[1] != len(t_real) or np.any(sol.y < 0):
        return None
    
    # 生存フィルター（他3種が0.05を下回ったら即却下）
    if np.any(sol.y[1] < 0.05) or np.any(sol.y[2] < 0.05) or np.any(sol.y[3] < 0.05):
        return None
        
    score = np.sum((sol.y[0] - x1_data) ** 2)
    return score, params, inits

# ==========================================
# 4. メインルーチン（並列実行の管理）
# ==========================================
if __name__ == '__main__':
    num_trials = 50000
    print(f"--- Macの全コアを解放して並列全探索を開始します（総試行: {num_trials}） ---")
    
    best_score = np.inf
    best_params = None
    best_inits = None
    
    # MacのCPUコアをフルに使って同時爆撃
    with ProcessPoolExecutor() as executor:
        seeds = range(num_trials)
        # map関数で全コアにタスクを均等に分配
        results = executor.map(evaluate_one_pattern, seeds)
        
        for i, res in enumerate(results):
            if i % 5000 == 0 and i > 0:
                print(f"解析進捗: {i}/{num_trials} パターン突破...")
            if res is not None:
                score, params, inits = res
                if score < best_score:
                    best_score = score
                    best_params = params
                    best_inits = inits
                    print(f"-> 【神解候補を発見！】 暫定最小エラー RSS: {best_score:.6f}")

    # (以下、最適解が得られた場合のプロット処理は前述のコードと同様のため省略)
    if best_params is not None:
        # 最終プロット用の再シミュレーション
        sol_verify = solve_ivp(full_bi_partite_lv_ode, [t_real[0], t_real[-1]], best_inits, t_eval=t_real, args=(best_params, fishing_rates, t_real))
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        species_labels = [['x1 (対象種資源量)', 'x2 (競合種)'], ['y1 (捕食者1)', 'y2 (捕食者2)']]
        for i in range(2):
            for j in range(2):
                idx = i * 2 + j
                ax = axes[i, j]
                ax.plot(t_real, sol_verify.y[idx] * scale_factor, 'b-', linewidth=2)
                if idx == 0: ax.plot(t_real, df_clean['資源量（千トン）'].values, 'ro', alpha=0.6)
                ax.set_title(species_labels[i][j])
                ax.grid(True)
        plt.tight_layout()
        plt.show()  