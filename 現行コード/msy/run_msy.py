"""
MSY（最大持続生産量）計算のメインスクリプト。

NLM(2006-2016) と LM(2017-2024) それぞれについて:
  (A) レジーム全期間の戦略的 MSY  … T = (最終年 - 初年)
  (B) 1 年ごとの戦術的 MSY        … T = 1 × 各年

各条件で 3 種のスイープを実施:
  1. 共通漁獲率スイープ    : 全種に同じ f を与えた収量曲線
  2. 4 次元粗グリッド探索  : f* と MSY 値の特定（6^4=1296 評価）
  3. 種別感度スイープ      : f* 基準で 1 種ずつ変化させた 4 本の曲線

出力:
  - コンソール: レジーム・範囲ごとの MSY 値・最適 f*・種別収量内訳
  - PNG: 収量曲線 (msy_common_sweep.png, msy_grid_scatter.png,
         msy_sensitivity.png, msy_tactical.png)
    → 保存先は このスクリプトと同じフォルダ（現行コード/msy/）

モデル: capacity_ry（12 変数, 主力モデル）または capacity（10 変数）
使い方:
  python3 run_msy.py               # capacity_ry（デフォルト）
  python3 run_msy.py capacity      # capacity
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------
# パス設定: 親フォルダ（現行コード/）の model.py・data_loader.py を再利用
# rank1_squid_largefish/run_rank1.py と同じパターン
# -----------------------------------------------------------------------
_here   = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_here)
sys.path.insert(0, _here)      # msy_core.py を先頭に
sys.path.append(_parent)       # 親フォルダの model.py・data_loader.py を後方追加

from data_loader import load_clean_dataframe, get_series, SPECIES_LABELS, KEYS
from model import estimate
from msy_core import (
    normalize_X0,
    scan_common_rate,
    grid_search_msy,
    species_sensitivity,
    tactical_msy_per_year,
    N_GRID, N_COMMON, N_SENS,
)

# -----------------------------------------------------------------------
# matplotlib 日本語フォント（run_estimation.py からコピー）
# -----------------------------------------------------------------------
plt.rcParams["font.family"]     = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Hiragino Sans", "DejaVu Sans", "Arial", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

# -----------------------------------------------------------------------
# 定数
# -----------------------------------------------------------------------
NLM_YEARS = (2006, 2016)
LM_YEARS  = (2017, 2024)

# 推定パラメータ
N_STARTS   = 40
# レジーム別正則化強度:
#   NLM は 11 点・12 変数で識別性が保てるため正則化不要（Phase4 で λ>0 だと当てはまり悪化が判明）
#   LM  は  8 点・12 変数で識別性が弱いため安定化
REG_LAMBDA = {"NLM": 0.0, "LM": 0.005}

# 絵のカラー
REGIME_COLORS = {"NLM": "#2166ac", "LM": "#d6604d"}
SPECIES_COLORS = ["#1b7837", "#762a83", "#e66101", "#4393c3"]


# =============================================================================
# データ準備ユーティリティ
# =============================================================================

def slice_series(series, mask):
    """mask で時系列を切り出す（run_estimation.py と同じ）。"""
    return {k: v[mask] for k, v in series.items()}


def regime_masks(series):
    """NLM / LM マスクを返す（run_estimation.py と同じ）。"""
    y = series["years"]
    return ((y >= NLM_YEARS[0]) & (y <= NLM_YEARS[1]),
            (y >= LM_YEARS[0])  & (y <= LM_YEARS[1]))


def get_regime_T(series_slice):
    """
    レジームの全期間 T = 最終年 - 初年（年単位）を返す。
    NLM(11 年) → T=10, LM(8 年) → T=7。
    """
    years = series_slice["years"].astype(float)
    return float(years[-1] - years[0])


def get_regime_X0_norm(series_slice, means):
    """
    レジームの初年観測資源量から正規化初期値を作成する。
    """
    obs_abs_t0 = np.array([
        float(series_slice["x1"][0]),
        float(series_slice["x2"][0]),
        float(series_slice["y1"][0]),
        float(series_slice["y2"][0]),
    ])
    return normalize_X0(obs_abs_t0, means)


# =============================================================================
# コンソール出力ユーティリティ
# =============================================================================

def _sep(char="=", n=72):
    return char * n


def print_strategic_result(regime_name, model_str, T, grid_res):
    """戦略的 MSY のグリッド探索結果をコンソールに整形出力する。"""
    print(_sep())
    print(f"[戦略的 MSY]  レジーム: {regime_name}  モデル: {model_str}  T={T:.1f} 年")
    print(_sep("-"))
    f_star = grid_res["f_star"]
    msy    = grid_res["msy"]
    per_sp = grid_res["per_species_at_msy"]
    print(f"  MSY        = {msy:.3f} 千トン/年")
    print(f"  最適 f*    : f_x1={f_star[0]:.3f}  f_x2={f_star[1]:.3f}  "
          f"f_y1={f_star[2]:.3f}  f_y2={f_star[3]:.3f}")
    print("  種別収量内訳（千トン/年）:")
    for i, lab in enumerate(SPECIES_LABELS):
        print(f"    {lab:22s}: {per_sp[i]:.3f}")
    print(f"  (評価点数: {grid_res['n_evaluated']:5d}  成功: {grid_res['n_success']:5d})")


def print_tactical_summary(regime_name, model_str, tac_list):
    """戦術的 MSY の年別結果をコンソールに一覧出力する。"""
    print(_sep())
    print(f"[戦術的 MSY]  レジーム: {regime_name}  モデル: {model_str}  T=1 年")
    print(_sep("-"))
    header = f"  {'年':>4}  {'MSY (千トン/年)':>15}  " \
             f"f_x1   f_x2   f_y1   f_y2"
    print(header)
    for r in tac_list:
        f = r["f_star"]
        print(f"  {r['year']:>4d}  {r['msy']:>15.3f}  "
              f"{f[0]:.3f}  {f[1]:.3f}  {f[2]:.3f}  {f[3]:.3f}")


# =============================================================================
# プロット関数群
# =============================================================================

def plot_common_sweep(sweep_results, model_str):
    """
    図 1: 共通漁獲率スイープの収量曲線（NLM vs LM）。
    subplot(1,2,*): 左=NLM, 右=LM。各パネルに合計と種別内訳を描く。
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    regime_names = ["NLM", "LM"]

    for col, (rname, sweep) in enumerate(zip(regime_names, sweep_results)):
        ax = axes[col]
        fc = sweep["f_common"]
        my = sweep["mean_yield"]
        ps = sweep["per_species"]   # shape (4, n_common)

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

    fig.suptitle("共通漁獲率 vs 平均漁獲量（NLM / LM）", fontsize=13)
    plt.tight_layout()
    out = os.path.join(_here, f"msy_common_sweep_{model_str}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")


def plot_grid_scatter(grid_results_nlm, grid_results_lm, model_str):
    """
    図 2: グリッド全評価の散布図（f の合計 vs 平均漁獲量, NLM vs LM 比較）。
    MSY 点を星マークで強調する。
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, grid_res, rname, color in zip(
        axes,
        [grid_results_nlm, grid_results_lm],
        ["NLM", "LM"],
        [REGIME_COLORS["NLM"], REGIME_COLORS["LM"]],
    ):
        valid = np.isfinite(grid_res["all_yield"])
        f_sum = grid_res["all_f"][valid].sum(axis=1)
        y_val = grid_res["all_yield"][valid]

        ax.scatter(f_sum, y_val, c=color, alpha=0.25, s=15, rasterized=True)

        # MSY 点を星マークで強調
        if np.isfinite(grid_res["msy"]):
            ax.scatter(
                grid_res["f_star"].sum(), grid_res["msy"],
                marker="*", s=200, color="gold", edgecolors="black",
                linewidths=0.8, zorder=10,
                label=f"MSY={grid_res['msy']:.2f}\n"
                      f"f*={grid_res['f_star'].round(3)}",
            )

        ax.set_title(f"{rname}: グリッド探索（{N_GRID}^4={N_GRID**4} 評価）")
        ax.set_xlabel("漁獲率の合計 Σfᵢ")
        ax.set_ylabel("平均漁獲量（千トン/年）")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, ls="--", alpha=0.35)

    fig.suptitle(f"グリッド全評価散布図 — {model_str}", fontsize=13)
    plt.tight_layout()
    out = os.path.join(_here, f"msy_grid_scatter_{model_str}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")


def plot_sensitivity(sens_results_nlm, sens_results_lm, model_str):
    """
    図 3: 種別感度スイープ（4 × 2 のグリッド, 行=種, 列=NLM/LM）。
    """
    fig, axes = plt.subplots(4, 2, figsize=(12, 16))
    regime_names = ["NLM", "LM"]
    sens_list    = [sens_results_nlm, sens_results_lm]

    for col, (rname, sens) in enumerate(zip(regime_names, sens_list)):
        for row, s in enumerate(sens):
            ax = axes[row, col]
            ax.plot(s["f_sweep"], s["mean_yield"], "k-", lw=2.0, label="合計")
            for i, lab in enumerate(SPECIES_LABELS):
                ax.plot(s["f_sweep"], s["per_species"][i], "--",
                        color=SPECIES_COLORS[i], lw=1.3, label=lab, alpha=0.8)
            ax.set_title(f"{rname}: {SPECIES_LABELS[row]} の f を変化（他は f* 固定）",
                         fontsize=9)
            ax.set_xlabel(f"f_{['x1','x2','y1','y2'][row]}", fontsize=8)
            ax.set_ylabel("平均漁獲量（千トン/年）", fontsize=8)
            ax.legend(fontsize=7)
            ax.grid(True, ls="--", alpha=0.4)

    fig.suptitle(f"種別感度スイープ（f* 基準 1 次元変化） — {model_str}", fontsize=13)
    plt.tight_layout()
    out = os.path.join(_here, f"msy_sensitivity_{model_str}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")


def plot_tactical(tac_nlm, tac_lm, model_str):
    """
    図 4: 戦術的 MSY の年別推移（NLM/LM を同一グラフ上に描く）。
    上段: MSY 値, 下段: 最適 f* の 4 種別。
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 9))

    # ── 上段: MSY 値 ──
    ax0 = axes[0]
    for tac_list, rname in [(tac_nlm, "NLM"), (tac_lm, "LM")]:
        years = [r["year"] for r in tac_list]
        msys  = [r["msy"]  for r in tac_list]
        color = REGIME_COLORS[rname]
        ax0.plot(years, msys, "o-", color=color, lw=2.0, ms=7, label=rname)
    ax0.set_title(f"戦術的 MSY（T=1 年ごと） — {model_str}")
    ax0.set_ylabel("最大平均漁獲量（千トン/年）")
    ax0.legend()
    ax0.grid(True, ls="--", alpha=0.45)

    # ── 下段: 最適 f* の年別推移 ──
    ax1 = axes[1]
    f_labels = ["f*_x1", "f*_x2", "f*_y1", "f*_y2"]
    for col, (tac_list, rname) in enumerate([(tac_nlm, "NLM"), (tac_lm, "LM")]):
        years = [r["year"] for r in tac_list]
        for i, flabel in enumerate(f_labels):
            vals = [r["f_star"][i] for r in tac_list]
            ls   = "-" if col == 0 else "--"
            alpha = 1.0 if col == 0 else 0.75
            ax1.plot(years, vals, ls, color=SPECIES_COLORS[i], lw=1.5,
                     label=f"{rname} {flabel}" if col == 0 else None,
                     alpha=alpha)
    ax1.set_ylabel("最適漁獲率 f*")
    ax1.set_xlabel("年")
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(True, ls="--", alpha=0.4)

    plt.tight_layout()
    out = os.path.join(_here, f"msy_tactical_{model_str}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")


def plot_nlm_lm_comparison(grid_nlm, grid_lm, model_str):
    """
    図 5: NLM vs LM の MSY 値と種別収量の棒グラフ比較。
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── 左: MSY 値の比較 ──
    ax = axes[0]
    rnames = ["NLM", "LM"]
    msys   = [grid_nlm["msy"], grid_lm["msy"]]
    colors = [REGIME_COLORS["NLM"], REGIME_COLORS["LM"]]
    bars = ax.bar(rnames, msys, color=colors, width=0.5, edgecolor="black")
    for bar, val in zip(bars, msys):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.2f}", ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("最大平均漁獲量（千トン/年）")
    ax.set_title(f"戦略的 MSY — NLM vs LM（{model_str}）")
    ax.grid(axis="y", ls="--", alpha=0.5)

    # ── 右: 種別収量内訳の積み上げ棒グラフ ──
    ax2 = axes[1]
    x = np.arange(2)
    width = 0.5
    bottom = np.zeros(2)
    per_sp_data = np.vstack([
        grid_nlm["per_species_at_msy"],
        grid_lm["per_species_at_msy"],
    ]).T   # shape (4, 2)

    for i, lab in enumerate(SPECIES_LABELS):
        vals = per_sp_data[i]
        vals_clipped = np.clip(vals, 0, None)   # 描画上は 0 以上にクリップ
        ax2.bar(x, vals_clipped, width, bottom=bottom,
                color=SPECIES_COLORS[i], label=lab, edgecolor="white", lw=0.5)
        bottom += vals_clipped

    ax2.set_xticks(x)
    ax2.set_xticklabels(["NLM", "LM"])
    ax2.set_ylabel("種別平均漁獲量（千トン/年）")
    ax2.set_title(f"種別収量内訳（f* での構成） — {model_str}")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(axis="y", ls="--", alpha=0.5)

    plt.tight_layout()
    out = os.path.join(_here, f"msy_nlm_lm_comparison_{model_str}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")


# =============================================================================
# メイン
# =============================================================================

def main():
    model_str = sys.argv[1] if len(sys.argv) > 1 else "capacity_ry"
    if model_str not in ("capacity", "capacity_ry"):
        print(f"[ERROR] model は capacity または capacity_ry を指定してください（指定: {model_str}）")
        sys.exit(1)

    print(_sep())
    print(f"MSY 計算  モデル: {model_str}  "
          f"正則化: NLM={REG_LAMBDA['NLM']}  LM={REG_LAMBDA['LM']}  "
          f"グリッド: {N_GRID}^4={N_GRID**4} 評価")
    print(_sep())

    # ------------------------------------------------------------------
    # データ読み込みとレジーム分割
    # ------------------------------------------------------------------
    df = load_clean_dataframe()
    series = get_series(df)
    nlm_mask, lm_mask = regime_masks(series)
    sl_nlm = slice_series(series, nlm_mask)
    sl_lm  = slice_series(series, lm_mask)
    regimes = [
        ("NLM", sl_nlm, NLM_YEARS),
        ("LM",  sl_lm,  LM_YEARS),
    ]

    # ------------------------------------------------------------------
    # Step 1: 各レジームの ODE パラメータ推定
    # ------------------------------------------------------------------
    print(f"\n[Step 1] ODE パラメータ推定 (n_starts={N_STARTS}, reg_lambda: NLM={REG_LAMBDA['NLM']}  LM={REG_LAMBDA['LM']})")
    est_results = {}
    for rname, sl, _ in regimes:
        n_y = len(sl["years"])
        reg = REG_LAMBDA[rname]  # レジーム別に正則化強度を切り替える
        print(f"  推定中: {rname} ({n_y} 年, reg_lambda={reg}) ...", flush=True)
        res = estimate(sl, model=model_str, n_starts=N_STARTS, reg_lambda=reg, seed=0)
        est_results[rname] = res
        m = res["metrics"]["overall"]
        print(f"    平均R²={m['mean_R2']:+.3f}  平均NRMSE={m['mean_NRMSE']:.3f}")
        if res["at_bounds"]:
            print(f"    ⚠ 境界張り付き: {', '.join(res['at_bounds'])}")

    # ------------------------------------------------------------------
    # Step 2: 戦略的 MSY（レジーム全期間）
    # ------------------------------------------------------------------
    print(f"\n[Step 2] 戦略的 MSY（レジーム全期間）")
    strategic = {}
    sweep_res_list = []
    grid_res_list  = []
    sens_res_list  = []

    for rname, sl, _ in regimes:
        est  = est_results[rname]
        pn   = est["params_norm"]
        mn   = est["means"]
        T    = get_regime_T(sl)
        X0n  = get_regime_X0_norm(sl, mn)

        print(f"\n  ── {rname}  T={T:.1f} 年 ──")

        # スイープ 1: 共通漁獲率
        print(f"  1. 共通漁獲率スイープ ({N_COMMON} 点) ...", flush=True)
        sweep = scan_common_rate(pn, mn, model_str, T, X0n)
        sweep_res_list.append(sweep)
        print(f"     最大収量: {sweep['best_yield']:.3f} 千トン/年  "
              f"at f_common={sweep['best_f']:.3f}")

        # スイープ 2: 4 次元グリッド探索
        print(f"  2. グリッド探索 ({N_GRID}^4={N_GRID**4} 評価) ...", flush=True)
        grid = grid_search_msy(pn, mn, model_str, T, X0n)
        grid_res_list.append(grid)
        print_strategic_result(rname, model_str, T, grid)

        # スイープ 3: 種別感度
        print(f"  3. 種別感度スイープ ({N_SENS} 点 × 4 種) ...", flush=True)
        sens = species_sensitivity(grid["f_star"], pn, mn, model_str, T, X0n)
        sens_res_list.append(sens)
        print(f"     完了")

        strategic[rname] = {
            "sweep": sweep, "grid": grid, "sens": sens,
            "T": T, "X0_norm": X0n, "means": mn, "params_norm": pn,
        }

    # ------------------------------------------------------------------
    # Step 3: 戦術的 MSY（1 年ごと）
    # ------------------------------------------------------------------
    print(f"\n[Step 3] 戦術的 MSY（T=1 年ごと, {N_GRID}^4={N_GRID**4} 評価 × 年数）")
    tactical = {}
    for rname, sl, _ in regimes:
        est  = est_results[rname]
        n_y  = len(sl["years"])
        print(f"  {rname} ({n_y} 年 × {N_GRID**4} 評価 = {n_y * N_GRID**4} 評価) ...", flush=True)
        tac = tactical_msy_per_year(sl, est["params_norm"], est["means"], model_str)
        tactical[rname] = tac
        print_tactical_summary(rname, model_str, tac)

    # ------------------------------------------------------------------
    # Step 4: PNG 出力
    # ------------------------------------------------------------------
    print(f"\n[Step 4] PNG 出力")
    plot_common_sweep(sweep_res_list, model_str)
    plot_grid_scatter(grid_res_list[0], grid_res_list[1], model_str)
    plot_sensitivity(sens_res_list[0], sens_res_list[1], model_str)
    plot_tactical(tactical["NLM"], tactical["LM"], model_str)
    plot_nlm_lm_comparison(grid_res_list[0], grid_res_list[1], model_str)

    # ------------------------------------------------------------------
    # Step 5: 全体サマリ
    # ------------------------------------------------------------------
    print("\n" + _sep())
    print("[全体サマリ]")
    print(_sep("-"))
    print(f"{'レジーム':>6}  {'戦略的MSY(千トン/年)':>22}  最適 f*（x1, x2, y1, y2）")
    for rname, data in strategic.items():
        g = data["grid"]
        f = g["f_star"]
        print(f"  {rname:>4}  {g['msy']:>22.3f}  "
              f"{f[0]:.3f}  {f[1]:.3f}  {f[2]:.3f}  {f[3]:.3f}")
    print(_sep())
    print("MSY 計算完了。")


if __name__ == "__main__":
    main()
