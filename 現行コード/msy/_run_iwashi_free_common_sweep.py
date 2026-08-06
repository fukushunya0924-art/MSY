"""
マイワシ版（Phase 11以前の種構成）自由推定12変数の共通漁獲率スイープ図を再生成する。

run_sustainability_diagnostics_iwashi.py の reconstruct_iwashi_estimates()
（docs/research_log.md Phase 8 記載の絶対パラメータ値から params_norm を逆変換、
再推定なし・round-trip検証済み）をそのまま再利用し、msy_core.scan_common_rate() で
共通漁獲率スイープを計算、run_msy.py の plot_common_sweep() と同じ描画ロジック
（feasible/infeasible の色分けなし）で PNG を保存する。

outputs/マイワシ版/自由推定/msy_共通漁獲率スイープ_無制約_マイワシ_ウルメイワシ_ブリ_サワラ_capacity_ry.png
を上書きする。既存ファイルはグリッド解像度違い（N_COMMON, 旧版で作られたもの）の
可能性があるため、現行コードで作り直す。
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_here)
if _here not in sys.path:
    sys.path.insert(0, _here)
if _parent not in sys.path:
    sys.path.append(_parent)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loader_iwashi import (
    load_clean_dataframe, get_series, regime_masks, slice_series,
    get_regime_T, get_regime_X0_norm, SPECIES_LABELS,
)
from msy_core import scan_common_rate, N_COMMON
from run_sustainability_diagnostics_iwashi import reconstruct_iwashi_estimates

plt.rcParams["font.family"]     = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Hiragino Sans", "DejaVu Sans", "Arial", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

SPECIES_COLORS = ["#1b7837", "#762a83", "#e66101", "#4393c3"]
_out_dir = os.path.join(_here, "outputs", "マイワシ版", "自由推定")
os.makedirs(_out_dir, exist_ok=True)

_SPECIES_NAMES = [lbl.split(" (")[0] for lbl in SPECIES_LABELS]
SPECIES_TAG = "_".join(_SPECIES_NAMES)


def plot_common_sweep_plain(sweep_results, model_str):
    """feasible/infeasible の色分けをしない、素の共通漁獲率スイープ図。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    regime_names = ["NLM", "LM"]

    for col, (rname, sweep) in enumerate(zip(regime_names, sweep_results)):
        ax = axes[col]
        fc = sweep["f_common"]
        my = sweep["mean_yield"]
        ps = sweep["per_species"]

        ax.plot(fc, my, "k-", lw=2.5, label="合計")
        for i, lab in enumerate(SPECIES_LABELS):
            ax.plot(fc, ps[i], "--", color=SPECIES_COLORS[i], lw=1.5,
                    label=lab, alpha=0.85)

        if np.isfinite(sweep["best_f"]):
            ax.axvline(sweep["best_f"], color="gray", ls=":", lw=1.2,
                       label=f"最大 f={sweep['best_f']:.3f}")
            ax.axhline(sweep["best_yield"], color="gray", ls=":", lw=1.2)

        ax.set_title(f"{rname}: 共通漁獲率スイープ（{model_str}）")
        ax.set_xlabel("共通漁獲率 f")
        ax.set_ylabel("平均漁獲量（千トン/年）")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, ls="--", alpha=0.45)

    fig.suptitle("共通漁獲率 vs 平均漁獲量（NLM / LM, マイワシ版）", fontsize=13)
    plt.tight_layout()
    out = os.path.join(_out_dir, f"msy_共通漁獲率スイープ_無制約_{SPECIES_TAG}_{model_str}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")


def main():
    df = load_clean_dataframe()
    series = get_series(df)
    nlm_mask, lm_mask = regime_masks(series)
    regimes = [
        ("NLM", slice_series(series, nlm_mask)),
        ("LM", slice_series(series, lm_mask)),
    ]

    print(f"[Step 1] マイワシ版 自由推定12変数を Phase 8 記載値から再構成（再推定なし, round-trip検証込み）")
    est_results = reconstruct_iwashi_estimates(regimes)

    sweep_res_list = []
    for rname, sl in regimes:
        est = est_results[rname]
        pn, mn = est["params_norm"], est["means"]
        T = get_regime_T(sl)
        X0n = get_regime_X0_norm(sl, mn)
        print(f"[Step 2] {rname}: 共通漁獲率スイープ ({N_COMMON} 点) ...")
        sweep = scan_common_rate(pn, mn, T, X0n)
        sweep_res_list.append(sweep)
        print(f"  最大収量: {sweep['best_yield']:.3f} 千トン/年  at f_common={sweep['best_f']:.3f}")

    print("[Step 3] PNG 出力")
    plot_common_sweep_plain(sweep_res_list, "capacity_ry")


if __name__ == "__main__":
    main()
