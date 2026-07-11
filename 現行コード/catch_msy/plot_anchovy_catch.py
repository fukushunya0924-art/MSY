"""カタクチイワシ漁獲量の推移プロット（太平洋12県版, 1956-2023）。

参考用スクリプト: 主対象4種（MAIN_KEYS）からはカタクチはウルメイワシに
置換済み（Phase 7d, 2026-07-07）だが、置換前の検討資料としてカタクチ単独の
時系列を確認できるよう残す。
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

# PNG出力先: catch_msy/outputs/legacy/（置換済み種の参考資料のため隔離）
_out_dir = os.path.join(_here, "outputs", "legacy")
os.makedirs(_out_dir, exist_ok=True)

from catch_data_loader import get_catch_series, SPECIES_LABELS, setup_japanese_plot_style

KEY = "anchovy"

plt = setup_japanese_plot_style()
years, catch = get_catch_series(KEY)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(years, catch, marker="o", markersize=3, linewidth=1.2, color="tab:blue")
ax.set_xlabel("年")
ax.set_ylabel("漁獲量（千トン）")
ax.set_title(f"{SPECIES_LABELS[KEY]} 漁獲量の推移（太平洋12県, {years[0]}-{years[-1]}）")
ax.grid(alpha=0.3)
fig.tight_layout()

out_path = os.path.join(_out_dir, "カタクチイワシ_漁獲量推移_参考_置換前レガシー.png")
fig.savefig(out_path, dpi=150)
print(f"保存: {out_path}")
