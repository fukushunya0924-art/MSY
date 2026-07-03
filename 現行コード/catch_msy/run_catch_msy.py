"""
連続時間 Catch-MSY のメインスクリプト（単一種4本）。

対象: マイワシ(sardine)・カタクチイワシ(anchovy)・ブリ(buri)・サワラ(sawara)
出力:
  - コンソール: 各種の r 幾何平均・分位点、K、MSY=rK/4、生存ペア数
  - PNG: catch_msy_overview.png（4種の漁獲量時系列＋r/MSY推定の一覧）

教授提案（Phase 7）に基づく位置づけ:
  被食者 sardine/anchovy の r は r_x1, r_x2 の外挿値。
  捕食者 buri/sawara の r は c1+d1, c2+d2 の実効推定値（解釈A）。

使い方:
  python3 run_catch_msy.py                 # 4種すべて・既定終端レンジ(0.2,0.6)
  python3 run_catch_msy.py sardine buri    # 魚種を指定
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

from catch_data_loader import get_catch_series, SPECIES_LABELS, MAIN_KEYS
from catch_msy_core import run_catch_msy, SPECIES_RESILIENCE, DEFAULT_FINAL_RANGE

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Hiragino Sans", "DejaVu Sans", "Arial", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False


def run_all(keys, final_range=DEFAULT_FINAL_RANGE, seed=0):
    results = {}
    for k in keys:
        years, catch = get_catch_series(k)
        res = run_catch_msy(years, catch, SPECIES_RESILIENCE[k],
                            final_range=final_range, seed=seed)
        res["years"] = years
        res["catch"] = catch
        results[k] = res
    return results


def print_table(results):
    print()
    print("=" * 88)
    print(f"連続時間 Catch-MSY 結果  （終端枯渇度レンジ {DEFAULT_FINAL_RANGE}）")
    print("=" * 88)
    hdr = (f"{'魚種':<8} {'resil':<7} {'生存/試行':>12} "
           f"{'r geomean':>10} {'r[25-75%]':>16} {'K':>9} {'MSY':>9}")
    print(hdr)
    print("-" * 88)
    for k, r in results.items():
        label = SPECIES_LABELS[k]
        viable = f"{r['n_viable']}/{r['n_samples']}"
        rrange = f"[{r['r_lo']:.3f},{r['r_hi']:.3f}]"
        print(f"{label:<8} {SPECIES_RESILIENCE[k]:<7} {viable:>12} "
              f"{r['r_geomean']:>10.3f} {rrange:>16} "
              f"{r['k_geomean']:>9.0f} {r['msy_geomean']:>9.0f}")
    print("-" * 88)
    print("K・MSY の単位は千トン/年。r は 1/年。")
    print("被食者(マイワシ,カタクチ)の r → r_x1, r_x2 の外挿値。")
    print("捕食者(ブリ,サワラ)の r → c1+d1, c2+d2 の実効推定値（解釈A）。")
    print("=" * 88)


def plot_overview(results, out_path):
    keys = list(results.keys())
    n = len(keys)
    fig, axes = plt.subplots(n, 1, figsize=(9, 2.6 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, k in zip(axes, keys):
        r = results[k]
        ax.plot(r["years"], r["catch"], "o-", ms=3, color="#1f77b4",
                label="漁獲量")
        txt = (f"r={r['r_geomean']:.3f} "
               f"[{r['r_lo']:.3f},{r['r_hi']:.3f}]  "
               f"MSY={r['msy_geomean']:.0f} 千トン/年  "
               f"(生存 {r['n_viable']}/{r['n_samples']})")
        ax.set_title(f"{SPECIES_LABELS[k]}   {txt}", fontsize=10, loc="left")
        ax.axhline(r["msy_geomean"], ls="--", lw=1, color="#d62728",
                   label="MSY=rK/4")
        ax.set_ylabel("千トン")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("年")
    fig.suptitle("連続時間 Catch-MSY（e-stat 全国漁獲量 1956-2024）",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=130)
    print(f"[saved] {out_path}")


def main():
    args = [a for a in sys.argv[1:] if a in SPECIES_LABELS]
    keys = args if args else MAIN_KEYS
    results = run_all(keys)
    print_table(results)
    out = os.path.join(_here, "catch_msy_overview.png")
    plot_overview(results, out)


if __name__ == "__main__":
    main()
