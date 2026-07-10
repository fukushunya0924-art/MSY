"""
マイワシ終盤挙動の診断スクリプト。

f_x1 をスイープし、マイワシ軌道と終端比 B(T)/B(0)・B(T)/B(T-1) を可視化する。

出力:
  - PNG: diagnose_iwashi.png（同フォルダ）
  - コンソール: 各レジームの f_x1 スイープ結果
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------
# パス設定
# -----------------------------------------------------------------------
_here   = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_here)
sys.path.insert(0, _here)
sys.path.append(_parent)

from data_loader import (
    load_clean_dataframe, get_series,
    slice_series, regime_masks, get_regime_T, get_regime_X0_norm,
)
from estimate_cache import REG_LAMBDA, load_estimates, estimate_regime
from msy_core import average_yield, N_EVAL_TRAJ

# -----------------------------------------------------------------------
# matplotlib 日本語フォント
# -----------------------------------------------------------------------
plt.rcParams["font.family"]        = "sans-serif"
plt.rcParams["font.sans-serif"]    = ["Hiragino Sans", "DejaVu Sans", "Arial", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

# -----------------------------------------------------------------------
# 定数
# -----------------------------------------------------------------------
# NLM_YEARS / LM_YEARS は data_loader.py で定義（run_msy.py 等と一貫させるため）

# 探索設定（N_STARTS/N_SEEDS/REG_LAMBDA）・推定・キャッシュは estimate_cache.py に集約。
MODEL_STR  = "capacity_ry"

# f_x1 スイープ
N_SWEEP    = 20
F_SWEEP    = np.linspace(0.0, 0.95, N_SWEEP)

# 他種の「現在の制約 MSY f*」（run_msy.py の制約版グリッド探索結果からの手動転記）。
# ⚠ この値は run_msy.py フルラン時点のスナップショットであり自動追従しない。
#   estimate_robust 移行（Phase 7d, commit a7597fc）やウルメイワシ載せ替え
#   （Phase 7d, commit 7ef0b36）で推定パラメータが変わっているため、
#   run_msy.py を再実行した場合は下記の値を最新の制約 f* に更新すること。
# NLM: (f_x1, f_x2, f_y1, f_y2) = (0.407, 0.543, 0.950, 0.950)
# LM:  (f_x1, f_x2, f_y1, f_y2) = (0.950, 0.543, 0.950, 0.407)
F_STAR_FIXED = {
    "NLM": np.array([0.407, 0.543, 0.950, 0.950]),
    "LM":  np.array([0.950, 0.543, 0.950, 0.407]),
}

# 軌道プロット用の代表 f_x1 インデックス（0.1, 0.5, 0.95 に最も近い点）
TRAJ_F_X1 = [0.1, 0.5, 0.95]
TRAJ_COLORS = ["#2166ac", "#f4a582", "#d6604d"]

REGIME_COLORS = {"NLM": "#2166ac", "LM": "#d6604d"}


# -----------------------------------------------------------------------
# f_x1 スイープ
# -----------------------------------------------------------------------
def sweep_fx1(rname, est, sl, T):
    """
    f_x1 を変化させてマイワシ軌道と終端比を計算する。

    Returns
    -------
    dict with keys:
      f_sweep, B_start, B_end, B_T_minus_1,
      ratio_vs_start, ratio_vs_prev, mean_yield_iwashi, trajs
    """
    pn     = est["params_norm"]
    mn     = est["means"]
    X0_norm = get_regime_X0_norm(sl, mn)
    f_base = F_STAR_FIXED[rname].copy()

    n      = len(F_SWEEP)
    B_start       = np.full(n, np.nan)
    B_end         = np.full(n, np.nan)
    B_T_minus_1   = np.full(n, np.nan)
    ratio_vs_start = np.full(n, np.nan)
    ratio_vs_prev  = np.full(n, np.nan)
    mean_yield_iwashi = np.full(n, np.nan)
    trajs = [None] * n  # 全軌道を格納

    for i, fx1 in enumerate(F_SWEEP):
        f_vec = f_base.copy()
        f_vec[0] = fx1
        res = average_yield(f_vec, pn, mn, T, X0_norm, n_eval=N_EVAL_TRAJ)
        if not res["success"]:
            continue

        traj = res["traj_abs"]  # shape (4, N_EVAL_TRAJ)
        trajs[i] = traj

        b_start = traj[0, 0]
        b_end   = traj[0, -1]
        b_prev  = traj[0, -2]  # 末尾から2番目

        B_start[i]         = b_start
        B_end[i]           = b_end
        B_T_minus_1[i]     = b_prev
        ratio_vs_start[i]  = b_end / b_start if b_start > 0 else np.nan
        ratio_vs_prev[i]   = b_end / b_prev  if b_prev > 0 else np.nan
        mean_yield_iwashi[i] = float(res["per_species_yield"][0])

    return {
        "f_sweep":           F_SWEEP,
        "B_start":           B_start,
        "B_end":             B_end,
        "B_T_minus_1":       B_T_minus_1,
        "ratio_vs_start":    ratio_vs_start,
        "ratio_vs_prev":     ratio_vs_prev,
        "mean_yield_iwashi": mean_yield_iwashi,
        "trajs":             trajs,
        "T":                 T,
        "X0_norm":           X0_norm,
        "pn":                pn,
        "mn":                mn,
    }


# -----------------------------------------------------------------------
# コンソール出力
# -----------------------------------------------------------------------
def print_sweep(rname, sw):
    print(f"\n{rname} f_x1スイープ結果:")
    print(f"  {'f_x1':>6}  {'B(T)/B(0)':>10}  {'B(T)/B(T-1)':>12}  {'収量(千トン/年)':>16}")
    for i, fx1 in enumerate(sw["f_sweep"]):
        rvs  = sw["ratio_vs_start"][i]
        rvp  = sw["ratio_vs_prev"][i]
        yld  = sw["mean_yield_iwashi"][i]
        rvs_str  = f"{rvs:.3f}"  if np.isfinite(rvs) else "   nan"
        rvp_str  = f"{rvp:.3f}"  if np.isfinite(rvp) else "   nan"
        yld_str  = f"{yld:.1f}"  if np.isfinite(yld) else "   nan"
        print(f"  f_x1={fx1:.2f}: B(T)/B(0)={rvs_str:>7}  B(T)/B(T-1)={rvp_str:>7}  収量={yld_str:>8}千トン")


def find_threshold(sw, key="ratio_vs_prev", threshold=0.9):
    """ratio が threshold を下回る最初の f_x1 を返す。なければ None。"""
    for i, fx1 in enumerate(sw["f_sweep"]):
        val = sw[key][i]
        if np.isfinite(val) and val < threshold:
            return fx1
    return None


# -----------------------------------------------------------------------
# プロット
# -----------------------------------------------------------------------
def make_plot(sweep_nlm, sweep_lm, est_nlm, est_lm, sl_nlm, sl_lm):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    regimes = [
        ("NLM", sweep_nlm, REGIME_COLORS["NLM"]),
        ("LM",  sweep_lm,  REGIME_COLORS["LM"]),
    ]

    for row, (rname, sw, rcol) in enumerate(regimes):
        T       = sw["T"]
        pn      = sw["pn"]
        mn      = sw["mn"]
        X0_norm = sw["X0_norm"]
        f_base  = F_STAR_FIXED[rname].copy()
        t_eval  = np.linspace(0.0, T, N_EVAL_TRAJ)

        # ── 列1: マイワシ軌道 ──
        ax0 = axes[row, 0]
        # 代表 3 点を色分け
        for fi_target, col in zip(TRAJ_F_X1, TRAJ_COLORS):
            idx = int(np.argmin(np.abs(F_SWEEP - fi_target)))
            traj = sw["trajs"][idx]
            fx1_actual = F_SWEEP[idx]
            if traj is not None:
                ax0.plot(t_eval, traj[0, :], color=col, lw=2.0,
                         label=f"f_x1={fx1_actual:.2f}")
        # 水平点線: B_start（有効な最初の値を使う）
        b0 = sw["B_start"]
        valid_b0 = b0[np.isfinite(b0)]
        if len(valid_b0) > 0:
            ax0.axhline(valid_b0[0], color="gray", ls="--", lw=1.2,
                        label=f"B(0)={valid_b0[0]:.1f}")
        ax0.set_title(f"{rname}: マイワシ軌道（他種f*固定）")
        ax0.set_xlabel("時間（年）")
        ax0.set_ylabel("マイワシ資源量（千トン）")
        ax0.legend(fontsize=8)
        ax0.grid(True, ls="--", alpha=0.4)

        # ── 列2: 終端比 vs f_x1 ──
        ax1 = axes[row, 1]
        valid = np.isfinite(sw["ratio_vs_start"]) & np.isfinite(sw["ratio_vs_prev"])
        fx1_v = sw["f_sweep"][valid]
        rvs   = sw["ratio_vs_start"][valid]
        rvp   = sw["ratio_vs_prev"][valid]

        ax1.plot(fx1_v, rvs, "o-", color=rcol,         lw=2.0, ms=5, label="B(T)/B(0)")
        ax1.plot(fx1_v, rvp, "s--", color="#4dac26",   lw=2.0, ms=5, label="B(T)/B(T−1)")
        ax1.axhline(0.9, color="red",  ls=":",  lw=1.5, label="y=0.9（制約ライン）")
        ax1.axhline(1.0, color="gray", ls="--", lw=1.2, label="y=1.0（現状維持）")
        ax1.set_title(f"{rname}: 終端比 vs f_x1")
        ax1.set_xlabel("f_x1")
        ax1.set_ylabel("比率")
        ax1.legend(fontsize=8)
        ax1.grid(True, ls="--", alpha=0.4)

        # ── 列3: マイワシ平均漁獲量 vs f_x1 ──
        ax2 = axes[row, 2]
        valid2 = np.isfinite(sw["mean_yield_iwashi"])
        ax2.plot(sw["f_sweep"][valid2], sw["mean_yield_iwashi"][valid2],
                 "o-", color=rcol, lw=2.0, ms=5)
        ax2.set_title(f"{rname}: マイワシ平均漁獲量 vs f_x1")
        ax2.set_xlabel("f_x1")
        ax2.set_ylabel("平均漁獲量（千トン/年）")
        ax2.grid(True, ls="--", alpha=0.4)

    fig.suptitle(f"マイワシ終盤挙動診断（モデル: {MODEL_STR}）", fontsize=14)
    plt.tight_layout()
    out = os.path.join(_here, "diagnose_iwashi.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  → {out}")


# -----------------------------------------------------------------------
# メイン
# -----------------------------------------------------------------------
def main():
    print("=" * 60)
    print(f"マイワシ終盤挙動診断  モデル: {MODEL_STR}")
    print("=" * 60)

    # データ読み込み
    df     = load_clean_dataframe()
    series = get_series(df)
    nlm_mask, lm_mask = regime_masks(series)
    sl_nlm = slice_series(series, nlm_mask)
    sl_lm  = slice_series(series, lm_mask)

    # ODE パラメータ推定（run_msy.py が保存したキャッシュがあれば再利用）
    print("\n[Step 1] ODE パラメータ推定")
    regimes_data = [("NLM", sl_nlm), ("LM", sl_lm)]
    cached = load_estimates()
    if cached is not None:
        print("  推定結果のキャッシュを再利用（run_msy.py が保存したもの）")
    est_results = {}
    for rname, sl in regimes_data:
        if cached is not None:
            res = cached[rname]
        else:
            print(f"  {rname} (reg_lambda={REG_LAMBDA[rname]}) ...", flush=True)
            res = estimate_regime(sl, rname)
        est_results[rname] = res
        m = res["metrics"]["overall"]
        print(f"    平均R²={m['mean_R2']:+.3f}  平均NRMSE={m['mean_NRMSE']:.3f}")

    # f_x1 スイープ
    print("\n[Step 2] f_x1 スイープ")
    T_nlm = get_regime_T(sl_nlm)
    T_lm  = get_regime_T(sl_lm)
    print(f"  NLM T={T_nlm:.1f}年  LM T={T_lm:.1f}年")

    print("  NLM スイープ中 ...", flush=True)
    sw_nlm = sweep_fx1("NLM", est_results["NLM"], sl_nlm, T_nlm)
    print("  LM  スイープ中 ...", flush=True)
    sw_lm  = sweep_fx1("LM",  est_results["LM"],  sl_lm,  T_lm)

    # コンソール出力
    print_sweep("NLM", sw_nlm)
    print_sweep("LM",  sw_lm)

    # 閾値サマリ
    print("\n[閾値サマリ]")
    for rname, sw in [("NLM", sw_nlm), ("LM", sw_lm)]:
        th_start = find_threshold(sw, "ratio_vs_start", 0.9)
        th_prev  = find_threshold(sw, "ratio_vs_prev",  0.9)
        print(f"  {rname}:")
        print(f"    B(T)/B(0)   < 0.9 になる最初の f_x1: "
              f"{ f'{th_start:.2f}' if th_start is not None else 'なし（全域で>=0.9）'}")
        print(f"    B(T)/B(T-1) < 0.9 になる最初の f_x1: "
              f"{ f'{th_prev:.2f}' if th_prev is not None else 'なし（全域で>=0.9）'}")

        # 収量の山型 vs 単調増加の判断
        yld = sw["mean_yield_iwashi"]
        valid_yld = yld[np.isfinite(yld)]
        if len(valid_yld) > 2:
            peak_idx = int(np.argmax(valid_yld))
            valid_fx1 = sw["f_sweep"][np.isfinite(yld)]
            if peak_idx == len(valid_yld) - 1:
                print(f"    マイワシ収量: 単調増加（内点最大なし、f_x1=0.95 で最大）")
            else:
                print(f"    マイワシ収量: 山型（ピーク f_x1≈{valid_fx1[peak_idx]:.2f}）")

    # PNG 出力
    print("\n[Step 3] PNG 出力")
    make_plot(sw_nlm, sw_lm, est_results["NLM"], est_results["LM"], sl_nlm, sl_lm)
    print("\n完了。")


if __name__ == "__main__":
    main()
