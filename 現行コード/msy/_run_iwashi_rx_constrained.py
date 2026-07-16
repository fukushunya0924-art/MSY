"""
マイワシ版（旧種構成: マイワシ+ウルメイワシ / ブリ+サワラ）の
制約推定10変数版（r_x1,r_x2のみ固定, C1,D1,C2,D2自由推定）を実行する一回限りのスクリプト。

現行の主力構成（マアジ版, data_loader.ASSIGN）はこのスクリプト内でのみ
一時的に上書きする（プロセス内限定、他スクリプト・ファイルには影響しない）。

固定するr_x1はマイワシのCatch-MSY値 0.940（FRA catch 1975-2024, 終端レンジ[0.6,0.95]の
例外採用, 旧fixed_params.py commit 4d9b3ebより）。r_x2(ウルメ)はx1に依存しないため
現行値 0.739 を流用する。

使い方: cd 現行コード/msy && python3 _run_iwashi_rx_constrained.py
出力:
  現行コード/msy/estimates_マイワシ_capacity_ry_constrained.pkl（キャッシュ、マアジ版とは別ファイル）
  現行コード/msy/outputs/fit_制約10var_rxのみ固定_マイワシ_ウルメイワシ_ブリ_サワラ_capacity_ry_constrained.png
"""
import os
import sys
import pickle
import time

_here = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_here)
sys.path.insert(0, _here)
sys.path.append(_parent)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import data_loader
# マイワシ版へ一時的に切替（このプロセス内限定）
data_loader.ASSIGN["x1"] = "マイワシ"
SPECIES_LABELS = ["マイワシ (x1)", "ウルメイワシ (x2)", "ブリ (y1)", "サワラ (y2)"]

from data_loader import (
    load_clean_dataframe, get_series, KEYS,
    slice_series, regime_masks,
)
import model_constrained as mc
from estimate_cache import N_STARTS_C, N_SEEDS_C, REG_LAMBDA_C
from plot_fit_smooth import smooth_trajectory

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Hiragino Sans", "DejaVu Sans", "Arial", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

_out_dir = os.path.join(_here, "outputs")
os.makedirs(_out_dir, exist_ok=True)

CACHE_FILE = os.path.join(_here, "estimates_マイワシ_capacity_ry_constrained.pkl")

# マイワシのCatch-MSY確定値（旧fixed_params.py, commit 4d9b3ebより。r_x2はx1非依存で現行値を流用）
FIXED = {"r_x1": 0.940, "r_x2": 0.739}


def load_cache():
    if not os.path.exists(CACHE_FILE):
        return None
    with open(CACHE_FILE, "rb") as f:
        payload = pickle.load(f)
    sig = {"n_starts_c": N_STARTS_C, "n_seeds_c": N_SEEDS_C,
           "reg_lambda_c": REG_LAMBDA_C, "fixed": FIXED}
    if any(payload.get(k) != v for k, v in sig.items()):
        return None
    return payload["est_results"]


def save_cache(est_results):
    payload = {"est_results": est_results,
               "n_starts_c": N_STARTS_C, "n_seeds_c": N_SEEDS_C,
               "reg_lambda_c": REG_LAMBDA_C, "fixed": FIXED}
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(payload, f)
    return CACHE_FILE


def main():
    df = load_clean_dataframe()
    series = get_series(df)
    nlm_mask, lm_mask = regime_masks(series)
    regimes = [("NLM", slice_series(series, nlm_mask)),
               ("LM", slice_series(series, lm_mask))]

    est_results = load_cache()
    if est_results is not None:
        print("→ 有効なキャッシュを再利用")
    else:
        est_results = {}
        for rname, sl in regimes:
            n_y = len(sl["years"])
            print(f"推定中: {rname} ({n_y}年, {N_STARTS_C[rname]}x{N_SEEDS_C[rname]}, "
                  f"reg_lambda={REG_LAMBDA_C[rname]}) ...", flush=True)
            t0 = time.time()
            res = mc.estimate_constrained_robust(
                sl, n_starts=N_STARTS_C[rname], reg_lambda=REG_LAMBDA_C[rname],
                n_seeds=N_SEEDS_C[rname], seed0=0, fixed=FIXED)
            dt = time.time() - t0
            est_results[rname] = res
            m = res["metrics"]["overall"]
            print(f"  完了 ({dt:.0f}s)  平均R²={m['mean_R2']:+.3f}  平均NRMSE={m['mean_NRMSE']:.3f}")
            if res["at_bounds"]:
                print(f"  ⚠ 境界張り付き: {', '.join(res['at_bounds'])}")
        path = save_cache(est_results)
        print(f"→ キャッシュ保存: {path}")

    print("\n" + "=" * 64)
    for rname, sl in regimes:
        res = est_results[rname]
        m = res["metrics"]
        ov = m["overall"]
        ap = res["params_abs"]
        pf = res["params_free"]
        print(f"■ {rname}")
        print(f"  cost      = {res['cost']:.4f}")
        print(f"  平均NRMSE = {ov['mean_NRMSE']:.4f}   平均R² = {ov['mean_R2']:+.4f}")
        print("  魚種別NRMSE: " + "  ".join(
            f"{SPECIES_LABELS[i]}={m[KEYS[i]]['NRMSE']:.4f}" for i in range(4)))
        print("  魚種別R²   : " + "  ".join(
            f"{SPECIES_LABELS[i]}={m[KEYS[i]]['R2']:+.3f}" for i in range(4)))
        print(f"  自由10変数: r_y1={pf[0]:.4f} r_y2={pf[1]:.4f} "
              f"L11={pf[2]:.4f} L12={pf[3]:.4f} L21={pf[4]:.4f} L22={pf[5]:.4f} "
              f"C1={pf[6]:.4f} D1={pf[7]:.4f} C2={pf[8]:.4f} D2={pf[9]:.4f}")
        print(f"  物理c/d(abs): c1={ap['c1']:.4f} d1={ap['d1']:.4f} "
              f"c2={ap['c2']:.4f} d2={ap['d2']:.4f}")
        print(f"  固定 r_x  : r_x1={res['fixed']['r_x1']:.3f} r_x2={res['fixed']['r_x2']:.3f}")
        if res["at_bounds"]:
            print(f"  ⚠ 境界張り付き: {', '.join(res['at_bounds'])}")
        print("-" * 64)

    # fit図
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    for col, (rname, sl) in enumerate(regimes):
        res = est_results[rname]
        years_fine, traj_fine = smooth_trajectory(sl, res)
        years = sl["years"]
        for row in range(4):
            ax = axes[row, col]
            ax.plot(years, sl[KEYS[row]], "ko", ms=7, label="実データ", zorder=5)
            ax.plot(years_fine, traj_fine[row], "r-", lw=2.2,
                    label=f"推定 R²={res['metrics'][KEYS[row]]['R2']:.2f} "
                          f"NRMSE={res['metrics'][KEYS[row]]['NRMSE']:.2f}")
            ax.set_title(f"{rname}: {SPECIES_LABELS[row]}")
            ax.set_ylabel("資源量（千トン）")
            ax.grid(True, ls="--", alpha=0.5)
            ax.legend(fontsize=8)
    fig.suptitle("【制約推定10変数, r_xのみ固定】マイワシ+ウルメイワシ / ブリ+サワラ — "
                 "capacity_ry_constrained（積分結果の滑らか軌道）", fontsize=13, y=1.003)
    plt.tight_layout()
    out = os.path.join(_out_dir,
        "fit_制約10var_rxのみ固定_マイワシ_ウルメイワシ_ブリ_サワラ_capacity_ry_constrained.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n→ fit図保存: {out}")


if __name__ == "__main__":
    main()
