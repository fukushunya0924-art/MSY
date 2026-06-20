import numpy as np
import matplotlib.pyplot as plt

# 日本語表示のためのフォント設定 (macOS用)
plt.rcParams['font.family'] = 'Hiragino Sans'

# 1. パラメータとシミュレーション条件の設定
T = 10          # 10年間
dt = 0.001       # 刻み幅
N_steps = int(T / dt)
N_sims = 1000   # モンテカルロ試行回数
target_yield = 0.15  # 目標とする時間平均漁獲量 (一定量)

# 決定論的な生物パラメータ (2種ロトカ・ヴォルテラ: 被食者x, 捕食者y)
r = 1.0       # 被食者の内的増加率
a = 1.0       # 捕食効率
c = 0.5       # 捕食者の死亡率
d = 0.5       # 効率変換
f_x = 0.3     # 人間が操作する管理規則（被食者への漁獲圧）

# 案Aの核心：環境ノイズの強度 (sigma)
sigma_x = 0.2 
sigma_y = 0.1

success_count = 0
all_yields = []

# 2. 確率シミュレーション（モンテカルロ）の実行
np.random.seed(42) # 再現性のため
for sim in range(N_sims):
    # 初期値の設定
    x = np.zeros(N_steps)
    y = np.zeros(N_steps)
    x[0] = 1.0
    y[0] = 0.5
    
    # タイムステップごとの動的計算 (オイラー・丸山法)
    for t in range(N_steps - 1):
        # 独立した標準正規乱数（ホワイトノイズの源）を生成
        dW_x = np.random.normal(0, np.sqrt(dt))
        dW_y = np.random.normal(0, np.sqrt(dt))
        
        # 確率微分方程式 (SDE) の更新式
        x[t+1] = x[t] + x[t] * (r - a * y[t] - f_x) * dt + sigma_x * x[t] * dW_x
        y[t+1] = y[t] + y[t] * (d * a * x[t] - c) * dt + sigma_y * y[t] * dW_y
        
        # 資源量の非負制約（絶滅したら0に張り付く）
        if x[t+1] < 0: x[t+1] = 0
        if y[t+1] < 0: y[t+1] = 0

    # 3. あなたの定義した「時間平均漁獲量」の計算
    # 積分 (1/T) ∫ f_x * x(t) dt を台形公式等で近似
    mean_yield = (1/T) * np.sum(f_x * x) * dt
    all_yields.append(mean_yield)
    
    if mean_yield >= target_yield:
        success_count += 1

# 4. 確率（％）の算出と判定
probability = (success_count / N_sims) * 100
print(f"【案A (SDE)】目標達成確率: {probability:.2f}%")
print(f"採用判定（95%基準）: {'採用' if probability >= 95 else '不採用'}")

# 上位5本の軌跡のみ可視化
plt.figure(figsize=(10, 4))
plt.hist(all_yields, bins=30, alpha=0.7, color='blue', edgecolor='black')
plt.axvline(target_yield, color='red', linestyle='--', label='Target Yield')
plt.title('案A: SDEモデルにおける10年平均漁獲量の分布')
plt.xlabel('Time-averaged Yield')
plt.ylabel('Frequency')
plt.legend()
plt.show()