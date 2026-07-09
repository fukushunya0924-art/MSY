"""
終端枯渇度レンジに対する r 推定の感度分析（箱ひげ図）。

Catch-MSY で資源量を使わない場合、結果を最も左右する仮定は
「終端年の枯渇度 B(T)/K をどのレンジに置くか」（Froese ヒューリスティック）。
複数レンジを振って、各魚種の生存 r 分布がどうシフトするかを箱ひげで並べる。

出力:
  catch_msy_sensitivity.png … 4魚種×終端レンジの箱ひげ（r の分布）

使い方:
  python3 sensitivity.py
"""
import os
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

from catch_data_loader import (get_catch_series, SPECIES_LABELS, MAIN_KEYS,
                                setup_japanese_plot_style, parse_species_args)
from catch_msy_core import run_catch_msy, SPECIES_RESILIENCE

plt = setup_japanese_plot_style()


# 振る終端枯渇度レンジ
# 低〜中: 減少・逼迫を仮定 / 高: 現在資源が潤沢（回復中）を仮定。
# マイワシ・カタクチのような現在高水準の浮魚は高レンジでのみ生存点が出る。
FINAL_RANGES = [
    (0.1, 0.4),
    (0.2, 0.6),
    (0.4, 0.7),
    (0.6, 0.95),
]
RANGE_LABELS = [f"[{lo},{hi}]" for lo, hi in FINAL_RANGES]


def run_sensitivity(keys, seed=0):
    """keys × FINAL_RANGES で run_catch_msy を回し、生存 r 配列を集める。"""
    data = {}  # key -> list（レンジごとの r_viable 配列）
    stats = {}  # key -> list（レンジごとの (geomean, n_viable)）
    for k in keys:
        years, catch = get_catch_series(k)
        rs, st = [], []
        for fr in FINAL_RANGES:
            res = run_catch_msy(years, catch, SPECIES_RESILIENCE[k],
                                final_range=fr, seed=seed)
            rs.append(res["r_viable"])
            st.append((res["r_geomean"], res["n_viable"]))
        data[k] = rs
        stats[k] = st
    return data, stats


def plot_sensitivity(data, stats, out_path):
    keys = list(data.keys())
    n = len(keys)
    ncol = 2
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 3.4 * nrow))
    axes = np.atleast_1d(axes).ravel()

    for ax, k in zip(axes, keys):
        rs = data[k]
        st = stats[k]
        # 空配列を箱ひげに渡すと落ちるので保護
        plot_data = [a if a.size else np.array([np.nan]) for a in rs]
        bp = ax.boxplot(plot_data, tick_labels=RANGE_LABELS, showfliers=False,
                        patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("#9ecae1")
            patch.set_alpha(0.7)
        # 幾何平均を赤点で重ねる
        for i, (gm, nv) in enumerate(st, start=1):
            ax.plot(i, gm, "D", color="#d62728", ms=6, zorder=5)
            ax.annotate(f"{gm:.3f}\n(n={nv})", (i, gm),
                        textcoords="offset points", xytext=(10, 0),
                        fontsize=7, va="center")
        ax.set_title(SPECIES_LABELS[k], fontsize=11)
        ax.set_ylabel("生存 r (1/年)")
        ax.set_xlabel("終端枯渇度 B(T)/K レンジ")
        ax.grid(alpha=0.3, axis="y")

    # 余ったパネルを消す
    for ax in axes[len(keys):]:
        ax.axis("off")

    fig.suptitle("終端枯渇度レンジに対する r 推定の感度（赤◆=幾何平均）",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=130)
    print(f"[saved] {out_path}")


def print_summary(stats):
    print()
    print("=" * 72)
    print("終端枯渇度レンジ別 r 幾何平均")
    print("=" * 72)
    header = f"{'魚種':<8}" + "".join(f"{lbl:>16}" for lbl in RANGE_LABELS)
    print(header)
    print("-" * 72)
    for k, st in stats.items():
        cells = "".join(f"{gm:>8.3f}(n={nv:<4})" for gm, nv in st)
        print(f"{SPECIES_LABELS[k]:<8}{cells}")
    print("=" * 72)


def main():
    keys = parse_species_args(sys.argv[1:], default_keys=MAIN_KEYS)
    data, stats = run_sensitivity(keys)
    print_summary(stats)
    out = os.path.join(_here, "catch_msy_sensitivity.png")
    plot_sensitivity(data, stats, out)


if __name__ == "__main__":
    main()
