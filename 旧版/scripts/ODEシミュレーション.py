import numpy as np
import matplotlib.pyplot as plt

# 日本語表示のためのフォント設定 (macOS用)
plt.rcParams['font.family'] = 'Hiragino Sans'

# 1. パラメータとシミュレーション条件の設定
T = 10
dt = 0.01
N_steps = int(T / dt)
N_sims = 1000
target_yield = 0.15

# 基本となる決定論的パラメータ
a = 1.0
c = 0.5
d = 0.5
f_x = 0.3

success_count = 0
all_yields = []

# 2. 確率シミュレーション（初期値・パラメータのサンプリング）
np.random.seed(42)
for sim in range(N_sims):
    # 案Bの核心：毎シミュレーションの開始時に、パラメータを分布からランダムに選ぶ
    # 例：内的増加率 r が 0.8 〜 1.2 の間で毎回ランダムに決まる（一様分布）
    r_sim = np.random.uniform(0.8, 1.2)
    x_init = np.random.uniform(0.9, 1.1) # 初期値も±10%ブレさせる
    
    x = np.zeros(N_steps)
    y = np.zeros(N_steps)
    x[0] = x_init
    y[0] = 0.5
    
    # 決定論的な時間発展 (通常の常微分方程式: ODE)
    for t in range(N_steps - 1):
        # 毎ステップのランダムノイズは入らない（軌道自体は滑らか）
        x[t+1] = x[t] + x[t] * (r_sim - a * y[t] - f_x) * dt
        y[t+1] = y[t] + y[t] * (d * a * x[t] - c) * dt
        
        if x[t+1] < 0: x[t+1] = 0
        if y[t+1] < 0: y[t+1] = 0

    # 3. 時間平均漁獲量の計算
    mean_yield = (1/T) * np.sum(f_x * x) * dt
    all_yields.append(mean_yield)
    
    if mean_yield >= target_yield:
        success_count += 1

# 4. 確率（％）の算出と判定
probability = (success_count / N_sims) * 100
print(f"【案B (ODE並列化)】目標達成確率: {probability:.2f}%")
print(f"採用判定（95%基準）: {'採用' if probability >= 95 else '不採用'}")

plt.figure(figsize=(10, 4))
plt.hist(all_yields, bins=30, alpha=0.7, color='green', edgecolor='black')
plt.axvline(target_yield, color='red', linestyle='--', label='Target Yield')
plt.title('案B: パラメータを振ったODEモデルにおける10年平均漁獲量の分布')
plt.xlabel('Time-averaged Yield')
plt.ylabel('Frequency')
plt.legend()
plt.show()