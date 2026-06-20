# --- 元データの可視化 ---
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# データの読み込み
csv_path = '/Users/fukuokashunya/MSY/スルメイカ秋季 資源量・漁獲量時系列データ - スルメイカ秋季 資源量・漁獲量時系列データ.csv'
df = pd.read_csv(csv_path)

# 日本語フォントの設定 (macOS環境を想定)
plt.rcParams['font.family'] = 'Hiragino Sans'

plt.figure(figsize=(12, 6))

# 資源量と漁獲量をプロット
plt.plot(df['漁期年'], df['資源量（千トン）'], marker='o', label='資源量（千トン）', color='tab:blue', linewidth=2)
plt.plot(df['漁期年'], df['漁獲量（千トン）'], marker='s', label='漁獲量（千トン）', color='tab:red', linewidth=2)

plt.title('スルメイカ秋季：資源量と漁獲量の時系列推移', fontsize=14)
plt.xlabel('漁期年', fontsize=12)
plt.ylabel('重量（千トン）', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
