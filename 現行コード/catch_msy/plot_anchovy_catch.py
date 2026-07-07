"""カタクチイワシ漁獲量の推移プロット（太平洋12県版, 1956-2023）。"""
import matplotlib.pyplot as plt

from catch_data_loader import get_catch_series, SPECIES_LABELS

KEY = "anchovy"
years, catch = get_catch_series(KEY)

plt.rcParams["font.family"] = "Hiragino Sans"
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(years, catch, marker="o", markersize=3, linewidth=1.2, color="tab:blue")
ax.set_xlabel("年")
ax.set_ylabel("漁獲量（千トン）")
ax.set_title(f"{SPECIES_LABELS[KEY]} 漁獲量の推移（太平洋12県, {years[0]}-{years[-1]}）")
ax.grid(alpha=0.3)
fig.tight_layout()

out_path = "anchovy_catch_timeseries.png"
fig.savefig(out_path, dpi=150)
print(f"保存: {out_path}")
