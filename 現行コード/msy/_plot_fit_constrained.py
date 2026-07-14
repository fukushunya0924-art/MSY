import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data_loader import load_clean_dataframe, get_series, regime_masks, slice_series
from estimate_cache import load_estimates_constrained
from plot_fit_smooth import smooth_trajectory, KEYS, SPECIES_LABELS, _out_dir

df = load_clean_dataframe(); series = get_series(df)
nlm_mask, lm_mask = regime_masks(series)
regimes = {"NLM": slice_series(series, nlm_mask), "LM": slice_series(series, lm_mask)}
cached = load_estimates_constrained()
assert cached is not None, "制約キャッシュが無い"
res_by = {}
for name, sl in regimes.items():
    res = cached[name]
    yrs, traj = smooth_trajectory(sl, res)
    res_by[name] = (sl, res, yrs, traj)

fig, axes = plt.subplots(4, 2, figsize=(14, 16))
for col, name in enumerate(["NLM","LM"]):
    sl, res, yrs, traj = res_by[name]
    years = sl["years"]
    for row in range(4):
        ax = axes[row, col]
        ax.plot(years, sl[KEYS[row]], "ko", ms=7, label="実データ", zorder=5)
        ax.plot(yrs, traj[row], "r-", lw=2.2,
                label=f"制約推定 R²={res['metrics'][KEYS[row]]['R2']:.2f} NRMSE={res['metrics'][KEYS[row]]['NRMSE']:.2f}")
        ax.set_title(f"{name}: {SPECIES_LABELS[row]}")
        ax.set_ylabel("資源量（千トン）"); ax.grid(True, ls="--", alpha=0.5); ax.legend(fontsize=8)
fig.suptitle("【制約推定】マアジ+ウルメ / ブリ+サワラ — Catch-MSY確定値固定(r_x1,r_x2,c1+d1,c2+d2)", fontsize=14, y=1.003)
plt.tight_layout()
out = os.path.join(_out_dir, "fit_制約_マアジ_ウルメイワシ_ブリ_サワラ_capacity_ry_constrained.png")
plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
print("saved:", out)
