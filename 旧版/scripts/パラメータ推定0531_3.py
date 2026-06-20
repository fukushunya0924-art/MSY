import os
# JAXのCPU特徴量チェック（AVX命令の有無）を強制的にスルーさせる安全弁
os.environ["JAX_DISABLE_CPU_FEATURE_GUARD"] = "1"

# ここから通常のインポートを開始
import numpy as np
import pandas as pd
import pymc as pm
# macOSのリンカーエラー(-ld64)を完全に回避するため、Cコンパイルを無効化
os.environ["PYTENSOR_FLAGS"] = "cxx="

# 既存のキャッシュが原因でエラーがループするのを防ぐため、古いコンプファイルを強制削除
cache_dir = os.path.expanduser("~/.pytensor")

if os.path.exists(cache_dir):
    import shutil
    try:
        shutil.rmtree(cache_dir)
    except Exception:
        pass
import numpy as np
import pandas as pd
import pymc as pm
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import arviz as az

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
T = len(t_real)

# 基準単位を「千トン」に統一
x1_real = df_clean['資源量（千トン）'].values
x2_real = df_clean['資源量（万トン）_sardine'].values * 10.0
y1_real = df_clean['資源量（トン）'].values / 1000.0
y2_real = df_clean['資源量（万トン）_mackerel'].values * 10.0

# (15, 4) 形状のデータ行列を作成
data_matrix = np.column_stack([x1_real, x2_real, y1_real, y2_real])
log_data = np.log(data_matrix)

# 4種それぞれの漁獲係数 f(t) の計算
f_x1 = np.clip(df_clean['漁獲量（千トン）'].values / df_clean['資源量（千トン）'].values, 0.0, 0.95)
f_x2 = np.clip(df_clean['漁獲量（万トン）_sardine'].values / df_clean['資源量（万トン）_sardine'].values, 0.0, 0.95)
f_y1 = np.clip(df_clean['漁獲量（トン）'].values / df_clean['資源量（トン）'].values, 0.0, 0.95)
f_y2 = np.clip(df_clean['漁獲量（万トン）_mackerel'].values / df_clean['漁獲量（万トン）_mackerel'].values, 0.0, 0.95)

# ==========================================
# 2. PyMC状態空間モデルの定義（漁獲圧完全連動版）
# ==========================================
with pm.Model() as lv_model:
    
    # --- 事前分布の設定 ---
    # 千トンスケールに適合するよう強めの事前分布で緊縛
    r = pm.HalfNormal("r", sigma=0.1, shape=2) + 0.4  # 被食者増加率
    d = pm.HalfNormal("d", sigma=0.1, shape=2) + 0.1  # 捕食者死亡率
    
    # 相互作用係数 (千トンスケールのため、微小な値を想定)
    beta = pm.HalfNormal("beta", sigma=0.01, shape=(2, 2))   # 被食者2種 × 捕食者2種
    gamma = pm.HalfNormal("gamma", sigma=0.01, shape=(2, 2)) # 捕食者2種 × 被食者2種
    
    # システムノイズ（環境変動による不確実性）
    sigma_proc = pm.HalfNormal("sigma_proc", sigma=0.1, shape=4)
    
    # 観測誤差（データの信頼性を5%程度に束縛）
    sigma_obs = 0.05 

    # --- 状態空間の構築 (時間発展) ---
    log_X = []
    
    # 1ステップ目の初期状態
    log_X.append(pm.Normal("log_X_0", mu=log_data[0], sigma=sigma_obs, shape=4))
    
    for t in range(T - 1):
        # 現在の状態を指数変換
        X_t = pm.math.exp(log_X[t])
        
        # 決定論的な次のステップの計算（インデックスは明示的に整数型にする）
        mu_next_x1 = log_X[t][0] + (r[0] - f_x1[t]) - beta[0,0]*X_t[2] - beta[0,1]*X_t[3]
        mu_next_x2 = log_X[t][1] + (r[1] - f_x2[t]) - beta[1,0]*X_t[2] - beta[1,1]*X_t[3]
        mu_next_y1 = log_X[t][2] + (-d[0] - f_y1[t]) + gamma[0,0]*X_t[0] + gamma[0,1]*X_t[1]
        mu_next_y2 = log_X[t][3] + (-d[1] - f_y2[t]) + gamma[1,0]*X_t[0] + gamma[1,1]*X_t[1]
        
        mu_next = pm.math.stack([mu_next_x1, mu_next_x2, mu_next_y1, mu_next_y2])
        
        log_X_next = pm.Normal(f"log_X_{t+1}", mu=mu_next, sigma=sigma_proc, shape=4)
        log_X.append(log_X_next)
        
    log_X_stacked = pm.math.stack(log_X)
    
    # --- 観測モデル ---
    observed = pm.Normal("observed", mu=log_X_stacked, sigma=sigma_obs, observed=log_data)

# ==========================================
# 3. MCMCサンプリングの実行
# ==========================================
print("--- PyMC状態空間モデルによるMCMCサンプリングを開始します ---")
with lv_model:
    # pipで入れたnumpyroをサンプラーに指定することで、PyTensorのC++コンパイルバグを完全スルー
    trace = pm.sample(
        draws=1000, 
        tune=1000, 
        target_accept=0.95, 
        chains=4, 
        random_seed=42, 
        nuts_sampler="numpyro"
    )

# ==========================================
# 4. 推定結果の抽出と可視化
# ==========================================
print("\n=== パラメータ推定完了（ベイズ事後平均） ===")
summary = az.summary(trace, var_names=["r", "d", "beta", "gamma"])
print(summary[["mean", "sd", "hdi_3%", "hdi_97%"]])

# サンプリングされた「真の状態（隠れた動態の平均値）」の抽出
post_log_X = trace.posterior["log_X_0"].model.rvs_to_values # 通常の手順で各時間ステップの平均を抽出
estimated_states = np.zeros((T, 4))
for t in range(T):
    estimated_states[t] = np.mean(np.exp(trace.posterior[f"log_X_{t}"].values), axis=(0, 1))

# 4魚種のプロット
fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
species_names = ['スルメイカ (x1)', 'マイワシ (x2)', 'ブリ (y1)', 'マサバ (y2)']

for i in range(2):
    for j in range(2):
        idx = i * 2 + j
        ax = axes[i, j]
        
        # MCMCが導き出した、観測ノイズを取り除いた「真のシステム軌道」
        ax.plot(t_real, estimated_states[:, idx], 'b-', linewidth=2.5, label='状態空間モデルによる推定真値')
        # 水産庁の生の観測データ点
        ax.plot(t_real, data_matrix[:, idx], 'ro', alpha=0.7, label='水産庁 資源評価データ')
        
        ax.set_title(species_names[idx], fontsize=12)
        ax.set_ylabel('資源量（千トン）')
        ax.grid(True, linestyle='--')
        ax.legend()

plt.tight_layout()
plt.show()