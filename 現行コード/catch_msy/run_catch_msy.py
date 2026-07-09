"""
連続時間 Catch-MSY のメインスクリプト（単一種4本）。

対象: マイワシ(sardine)・ウルメイワシ(urume)・ブリ(buri)・サワラ(sawara)
　　　（2026-07-07確定: カタクチ→ウルメイワシ置換）
出力:
  - コンソール: 各種の r 幾何平均・分位点、K、MSY=rK/4、生存ペア数
  - PNG: catch_msy_overview.png（4種の漁獲量時系列＋r/MSY推定の一覧）

教授提案（Phase 7）に基づく位置づけ:
  被食者 sardine/urume の r は r_x1, r_x2 の外挿値。
  捕食者 buri/sawara の r は c1+d1, c2+d2 の実効推定値（解釈A）。

終端枯渇度レンジは魚種ごとに Froese Table1 の標準ルール（終端年 catch/max 比）
で個別に選ぶ（FINAL_RANGE_BY_SPECIES）。マイワシのみ標準ルールのレンジで
生存ペアが0（ブーム・バスト構造のため、Phase 5b/7c で既知）なので、
唯一解ける高レンジ[0.6,0.95]を暫定採用。

使い方:
  python3 run_catch_msy.py                 # 4種すべて・種別既定レンジ
  python3 run_catch_msy.py sardine buri    # 魚種を指定
"""
import os
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

from catch_data_loader import (get_catch_series, SPECIES_LABELS, MAIN_KEYS,
                                setup_japanese_plot_style, parse_species_args)
from catch_msy_core import run_catch_msy, SPECIES_RESILIENCE, DEFAULT_FINAL_RANGE

plt = setup_japanese_plot_style()


# 魚種ごとの終端枯渇度レンジ（Table1標準ルール。マイワシのみ例外）
FINAL_RANGE_BY_SPECIES = {
    "sardine": (0.6, 0.95),  # 標準ルール[0.01,0.4]はn=0（ブーム・バスト）→高レンジ暫定採用
    "urume":   (0.01, 0.4),  # 終端/max=0.42 <=0.5
    "buri":    (0.3, 0.7),   # 終端/max=0.55 >0.5
    "sawara":  (0.01, 0.4),  # 終端/max=0.46 <=0.5
}


def run_all(keys, final_range=None, seed=0):
    results = {}
    for k in keys:
        years, catch = get_catch_series(k)
        fr = final_range if final_range is not None else \
            FINAL_RANGE_BY_SPECIES.get(k, DEFAULT_FINAL_RANGE)
        res = run_catch_msy(years, catch, SPECIES_RESILIENCE[k],
                            final_range=fr, seed=seed)
        res["years"] = years
        res["catch"] = catch
        results[k] = res
    return results


def print_table(results):
    print()
    print("=" * 96)
    print("連続時間 Catch-MSY 結果  （終端枯渇度レンジは魚種ごとに個別設定）")
    print("=" * 96)
    hdr = (f"{'魚種':<8} {'resil':<7} {'終端レンジ':>12} {'生存/試行':>12} "
           f"{'r geomean':>10} {'r[25-75%]':>16} {'K':>9} {'MSY':>9}")
    print(hdr)
    print("-" * 96)
    for k, r in results.items():
        label = SPECIES_LABELS[k]
        viable = f"{r['n_viable']}/{r['n_samples']}"
        rrange = f"[{r['r_lo']:.3f},{r['r_hi']:.3f}]"
        frange = f"[{r['final_range'][0]},{r['final_range'][1]}]"
        print(f"{label:<8} {SPECIES_RESILIENCE[k]:<7} {frange:>12} {viable:>12} "
              f"{r['r_geomean']:>10.3f} {rrange:>16} "
              f"{r['k_geomean']:>9.0f} {r['msy_geomean']:>9.0f}")
    print("-" * 96)
    print("K・MSY の単位は千トン/年。r は 1/年。")
    print("被食者(マイワシ,ウルメイワシ)の r → r_x1, r_x2 の外挿値。")
    print("捕食者(ブリ,サワラ)の r → c1+d1, c2+d2 の実効推定値（解釈A）。")
    print("=" * 96)


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
               f"MSY={r['msy_geomean']:.1f} 千トン/年  "
               f"終端レンジ{r['final_range']}  "
               f"(生存 {r['n_viable']}/{r['n_samples']})")
        ax.set_title(f"{SPECIES_LABELS[k]}   {txt}", fontsize=10, loc="left")
        ax.axhline(r["msy_geomean"], ls="--", lw=1, color="#d62728",
                   label="MSY=rK/4")
        ax.set_ylabel("千トン")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("年")
    fig.suptitle("連続時間 Catch-MSY（e-stat 太平洋12県漁獲量 1956-2023）",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=130)
    print(f"[saved] {out_path}")


def main():
    keys = parse_species_args(sys.argv[1:], default_keys=MAIN_KEYS)
    results = run_all(keys)
    print_table(results)
    out = os.path.join(_here, "catch_msy_overview.png")
    plot_overview(results, out)


if __name__ == "__main__":
    main()
