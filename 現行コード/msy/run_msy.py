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

モデル: capacity_ry（12 変数, 唯一の現行モデル。capacity/full は廃止済み）
使い方:
  python3 run_msy.py               # 自由推定（12変数）→ MSY（従来どおり）
  python3 run_msy.py --constrained # 制約推定（8変数, Catch-MSY確定値 r_x1/r_x2/
                                   #   c1+d1/c2+d2 を固定）→ MSY
    → Step 1 の推定器だけ estimate_regime_constrained に差し替わり、Step 2〜5
      （MSY計算・作図）は無改修で共用。出力は *_capacity_ry_constrained.png に分離。
      固定値は fixed_params.py（Catch-MSY 太平洋12県の確定値）が単一の真実の源。
      初回は約11分（診断予算 16×8, Phase 8実測）、以降は署名一致でキャッシュ再利用。
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

# PNG出力先: msy/outputs/（種構成＋実装＋制約種別を明記したファイル名で保存）
_out_dir = os.path.join(_here, "outputs")
os.makedirs(_out_dir, exist_ok=True)

from data_loader import (
    load_clean_dataframe, get_series, SPECIES_LABELS, KEYS,
    NLM_YEARS, LM_YEARS, slice_series, regime_masks,
    get_regime_T, get_regime_X0_norm,
)
from estimate_cache import (
    N_STARTS, N_SEEDS, REG_LAMBDA, estimate_regime, save_estimates,
    N_STARTS_C, N_SEEDS_C, REG_LAMBDA_C,
    estimate_regime_constrained, save_estimates_constrained,
    load_estimates_constrained,
)
import fixed_params
from msy_core import (
    average_yield,
    scan_common_rate,
    grid_search_msy,
    species_sensitivity,
    tactical_msy_per_year,
    check_sustainability,
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
# NLM_YEARS / LM_YEARS は data_loader.py で定義（regime_masks 等と一貫させるため）

# 推定パラメータ（N_STARTS / N_SEEDS / REG_LAMBDA）は estimate_cache.py に集約。
# 推定結果は同モジュールの save_estimates でキャッシュし、plot_fit_smooth.py 等が再利用する。

# -----------------------------------------------------------------------
# 持続性制約設定（戦略的 MSY にのみ適用）
# -----------------------------------------------------------------------
SUSTAIN_CFG = {"scope": "all", "mode": "endpoint", "tol": 0.1}
# 戦略的 MSY 専用グリッド解像度（8^4 = 4096 評価）
# 戦術的 MSY は既存の N_GRID=6 のまま維持する
N_GRID_STRATEGIC = 8

# 絵のカラー
REGIME_COLORS = {"NLM": "#2166ac", "LM": "#d6604d"}
SPECIES_COLORS = ["#1b7837", "#762a83", "#e66101", "#4393c3"]


# =============================================================================
# コンソール出力ユーティリティ
# =============================================================================

def _sep(char="=", n=72):
    return char * n


def print_strategic_result(regime_name, model_str, T, grid_res,
                           sustain_cfg=None, sustain_margins=None):
    """
    戦略的 MSY のグリッド探索結果をコンソールに整形出力する。

    Parameters
    ----------
    grid_res : dict
        grid_search_msy() の返り値。
    sustain_cfg : dict or None
        SUSTAIN_CFG（持続性制約設定）。制約版出力に使う。
    sustain_margins : ndarray shape (4,) or None
        制約 MSY 点での期末増減率。avg_yield の返り値から check_sustainability で取得済みのもの。
    """
    print(_sep())
    print(f"[戦略的 MSY]  レジーム: {regime_name}  モデル: {model_str}  T={T:.1f} 年")
    print(_sep("-"))

    # ── 無制約版（比較用・従来どおり） ──
    f_star = grid_res["f_star"]
    msy    = grid_res["msy"]
    per_sp = grid_res["per_species_at_msy"]
    n_eval = grid_res["n_evaluated"]
    n_ok   = grid_res["n_success"]
    print(f"  [無制約] MSY = {msy:.3f} 千トン/年")
    print(f"           f*  : f_x1={f_star[0]:.3f}  f_x2={f_star[1]:.3f}  "
          f"f_y1={f_star[2]:.3f}  f_y2={f_star[3]:.3f}")
    print("           種別収量内訳（千トン/年）:")
    for i, lab in enumerate(SPECIES_LABELS):
        print(f"             {lab:22s}: {per_sp[i]:.3f}")
    print(f"  (評価点数: {n_eval:5d}  ODE成功: {n_ok:5d})")

    # ── 制約版（sustain 指定時） ──
    if sustain_cfg is not None:
        n_feas = grid_res["n_feasible"]
        f_con  = grid_res["f_star_constrained"]
        msy_c  = grid_res["msy_constrained"]
        per_c  = grid_res["per_species_at_msy_constrained"]
        print(_sep("-"))
        tol_pct = int(sustain_cfg.get("tol", 0.1) * 100)
        print(f"  [制約版]  持続可能点: {n_feas:5d} / {n_eval:5d}")
        print(f"  制約 MSY = {msy_c:.3f} 千トン/年"
              f"  (全時点 ≥ 初期値×{100 - tol_pct}%)")
        if np.isfinite(msy_c):
            print(f"  制約 f*  : f_x1={f_con[0]:.3f}  f_x2={f_con[1]:.3f}  "
                  f"f_y1={f_con[2]:.3f}  f_y2={f_con[3]:.3f}")
            print("  制約 f* での種別収量（千トン/年）:")
            for i, lab in enumerate(SPECIES_LABELS):
                print(f"             {lab:22s}: {per_c[i]:.3f}")
            if sustain_margins is not None:
                print("  制約 f* 点での期末増減（margins = B_end/B0 - 1）:")
                for i, lab in enumerate(SPECIES_LABELS):
                    m_val = sustain_margins[i]
                    if np.isfinite(m_val):
                        print(f"             {lab:22s}: {m_val:+.1%}")
                    else:
                        print(f"             {lab:22s}: N/A")
        else:
            print("  ※ 持続可能点なし（feasible点が0件）")


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
    out = os.path.join(_out_dir, f"msy_共通漁獲率スイープ_無制約_マイワシ_ウルメイワシ_ブリ_サワラ_{model_str}.png")
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
    out = os.path.join(_out_dir, f"msy_グリッド散布_無制約_マイワシ_ウルメイワシ_ブリ_サワラ_{model_str}.png")
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
    out = os.path.join(_out_dir, f"msy_種別感度スイープ_マイワシ_ウルメイワシ_ブリ_サワラ_{model_str}.png")
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
    out = os.path.join(_out_dir, f"msy_戦術的MSY_年次_マイワシ_ウルメイワシ_ブリ_サワラ_{model_str}.png")
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
    out = os.path.join(_out_dir, f"msy_NLM_LM比較_無制約_マイワシ_ウルメイワシ_ブリ_サワラ_{model_str}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")


# =============================================================================
# 制約版プロット関数群（ファイル名: *_constrained_<model>.png）
# =============================================================================

def plot_grid_scatter_constrained(grid_results_nlm, grid_results_lm, model_str):
    """
    図 C1: グリッド全評価散布図（制約版）。
    feasible 点と infeasible 点を色分けし、制約 MSY 点を星で強調する。
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, grid_res, rname in zip(
        axes,
        [grid_results_nlm, grid_results_lm],
        ["NLM", "LM"],
    ):
        valid    = np.isfinite(grid_res["all_yield"])
        feasible = grid_res["all_feasible"]

        f_sum_inf  = grid_res["all_f"][valid & ~feasible].sum(axis=1)
        y_inf      = grid_res["all_yield"][valid & ~feasible]
        f_sum_feas = grid_res["all_f"][valid & feasible].sum(axis=1)
        y_feas     = grid_res["all_yield"][valid & feasible]

        # infeasible: 薄いグレー
        ax.scatter(f_sum_inf, y_inf, c="lightgray", alpha=0.3, s=12,
                   rasterized=True, label="infeasible")
        # feasible: レジームカラー
        color = REGIME_COLORS[rname]
        ax.scatter(f_sum_feas, y_feas, c=color, alpha=0.5, s=15,
                   rasterized=True, label="feasible")

        # 制約 MSY 点を星マーク
        msy_c = grid_res["msy_constrained"]
        f_c   = grid_res["f_star_constrained"]
        if np.isfinite(msy_c):
            ax.scatter(
                f_c.sum(), msy_c,
                marker="*", s=250, color="gold", edgecolors="black",
                linewidths=0.8, zorder=10,
                label=f"制約MSY={msy_c:.2f}\nf*={f_c.round(3)}",
            )

        n_feas = grid_res["n_feasible"]
        n_eval = grid_res["n_evaluated"]
        ax.set_title(f"{rname}: 制約グリッド探索（持続可能点: {n_feas}/{n_eval}）")
        ax.set_xlabel("漁獲率の合計 Σfᵢ")
        ax.set_ylabel("平均漁獲量（千トン/年）")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, ls="--", alpha=0.35)

    fig.suptitle(f"制約グリッド散布図（feasible/infeasible 色分け） — {model_str}", fontsize=13)
    plt.tight_layout()
    out = os.path.join(_out_dir, f"msy_グリッド散布_持続可能制約_マイワシ_ウルメイワシ_ブリ_サワラ_{model_str}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")


def plot_common_sweep_constrained(sweep_results, model_str):
    """
    図 C2: 共通漁獲率スイープ（制約版）。
    feasible 域を緑網掛けで示し、制約 MSY 点を縦線で強調する。
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    regime_names = ["NLM", "LM"]

    for col, (rname, sweep) in enumerate(zip(regime_names, sweep_results)):
        ax = axes[col]
        fc = sweep["f_common"]
        my = sweep["mean_yield"]
        feasible_mask = sweep["feasible_mask"]

        ax.plot(fc, my, "k-", lw=2.5, label="合計（無制約）")

        # feasible 域を薄い緑で塗りつぶす
        for i in range(len(fc) - 1):
            if feasible_mask[i] and feasible_mask[i + 1]:
                ax.axvspan(fc[i], fc[i + 1], color="green", alpha=0.12)

        # 個別点を色分けして描く
        ax.scatter(
            fc[feasible_mask], my[feasible_mask],
            c="green", s=20, zorder=5, label="feasible", alpha=0.8,
        )
        ax.scatter(
            fc[~feasible_mask], my[~feasible_mask],
            c="lightgray", s=15, zorder=4, label="infeasible", alpha=0.5,
        )

        # 制約最大点
        best_f_c = sweep["best_f_constrained"]
        best_y_c = sweep["best_yield_constrained"]
        if np.isfinite(best_f_c):
            ax.axvline(best_f_c, color="green", ls="--", lw=1.5,
                       label=f"制約最大 f={best_f_c:.3f}")
            ax.axhline(best_y_c, color="green", ls=":", lw=1.2)

        # 無制約最大点
        if np.isfinite(sweep["best_f"]):
            ax.axvline(sweep["best_f"], color="gray", ls=":", lw=1.2,
                       label=f"無制約最大 f={sweep['best_f']:.3f}")

        ax.set_title(f"{rname}: 共通漁獲率スイープ（制約版・{model_str}）")
        ax.set_xlabel("共通漁獲率 f")
        ax.set_ylabel("平均漁獲量（千トン/年）")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, ls="--", alpha=0.45)

    fig.suptitle("共通漁獲率スイープ（制約版 feasible 域を緑で表示）", fontsize=13)
    plt.tight_layout()
    out = os.path.join(_out_dir, f"msy_共通漁獲率スイープ_持続可能制約_マイワシ_ウルメイワシ_ブリ_サワラ_{model_str}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")


def plot_nlm_lm_comparison_constrained(grid_nlm, grid_lm, model_str):
    """
    図 C3: NLM vs LM の制約 MSY 値と種別収量の棒グラフ比較。
    無制約 MSY も同一グラフに薄く重ねて比較する。
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── 左: MSY 値の比較（無制約 vs 制約） ──
    ax = axes[0]
    rnames    = ["NLM", "LM"]
    msys_unc  = [grid_nlm["msy"],             grid_lm["msy"]]
    msys_con  = [grid_nlm["msy_constrained"], grid_lm["msy_constrained"]]
    x_pos     = np.arange(len(rnames))
    width     = 0.35
    colors    = [REGIME_COLORS["NLM"], REGIME_COLORS["LM"]]

    bars_unc = ax.bar(x_pos - width / 2, msys_unc, width,
                      color=colors, alpha=0.4, edgecolor="black",
                      label="無制約 MSY", hatch="//")
    bars_con = ax.bar(x_pos + width / 2, msys_con, width,
                      color=colors, alpha=0.85, edgecolor="black",
                      label="制約 MSY")
    for bar, val in zip(bars_unc, msys_unc):
        if np.isfinite(val):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9, color="gray")
    for bar, val in zip(bars_con, msys_con):
        if np.isfinite(val):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(rnames)
    ax.set_ylabel("最大平均漁獲量（千トン/年）")
    ax.set_title(f"戦略的 MSY 比較（無制約 vs 制約） — {model_str}")
    ax.legend(fontsize=9)
    ax.grid(axis="y", ls="--", alpha=0.5)

    # ── 右: 制約 f* での種別収量内訳 ──
    ax2 = axes[1]
    x  = np.arange(2)
    bottom = np.zeros(2)
    per_sp_data = np.vstack([
        grid_nlm["per_species_at_msy_constrained"],
        grid_lm["per_species_at_msy_constrained"],
    ]).T   # shape (4, 2)

    for i, lab in enumerate(SPECIES_LABELS):
        vals = per_sp_data[i]
        vals_clipped = np.where(np.isfinite(vals), np.clip(vals, 0, None), 0.0)
        ax2.bar(x, vals_clipped, width, bottom=bottom,
                color=SPECIES_COLORS[i], label=lab, edgecolor="white", lw=0.5)
        bottom += vals_clipped

    ax2.set_xticks(x)
    ax2.set_xticklabels(["NLM", "LM"])
    ax2.set_ylabel("種別平均漁獲量（千トン/年）")
    ax2.set_title(f"制約 f* での種別収量内訳 — {model_str}")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(axis="y", ls="--", alpha=0.5)

    plt.tight_layout()
    out = os.path.join(_out_dir, f"msy_NLM_LM比較_持続可能制約_マイワシ_ウルメイワシ_ブリ_サワラ_{model_str}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")


# =============================================================================
# メイン
# =============================================================================

def main():
    # model_str は現在 capacity_ry 固定（model.py の MODELS には capacity_ry のみ
    # 定義されている。capacity/full は2026-06-30に廃止済み、CLAUDE.md参照）。
    # 出力ファイル名・コンソール表示のラベルとしてのみ使う。
    # 引数解析: 位置引数 model（capacity_ry のみ）と --constrained フラグ。
    _args = sys.argv[1:]
    constrained = "--constrained" in _args
    _positional = [a for a in _args if not a.startswith("--")]
    model_str = _positional[0] if _positional else "capacity_ry"
    if model_str != "capacity_ry":
        print(f"[ERROR] model は capacity_ry のみ指定可能です（指定: {model_str}）。"
              f"capacity/full モデルは廃止されました。")
        sys.exit(1)
    # 制約モードでは出力ファイル名・図タイトル・コンソールのラベルを分離する。
    # Step 2〜5 は model_str を「ラベル」としてしか使わないため、これで無改修共用できる。
    if constrained:
        model_str = "capacity_ry_constrained"

    print(_sep())
    _mode = "制約推定(8変数, Catch-MSY確定値固定)" if constrained else "自由推定(12変数)"
    print(f"MSY 計算  モデル: {model_str}  推定: {_mode}  "
          f"戦略グリッド: {N_GRID_STRATEGIC}^4={N_GRID_STRATEGIC**4} 評価  "
          f"戦術グリッド: {N_GRID}^4={N_GRID**4} 評価")
    print(f"持続性制約: scope={SUSTAIN_CFG['scope']}  mode={SUSTAIN_CFG['mode']}  "
          f"tol={SUSTAIN_CFG['tol']}")
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
    #   constrained=False: 12自由変数（従来）
    #   constrained=True : 8自由変数（Catch-MSY確定値 r_x1/r_x2/c1+d1/c2+d2 を固定）
    # 以降の Step 2〜5 は est_results の params_norm/means のみ使うので、どちらでも共用。
    # ------------------------------------------------------------------
    if constrained:
        fx = fixed_params.get_point()
        print(f"\n[Step 1] 制約付き ODE 推定（8自由変数, Catch-MSY確定値を固定）")
        print(f"    固定値: r_x1={fx['r_x1']:.3f}  r_x2={fx['r_x2']:.3f}  "
              f"S1(c1+d1)={fx['S1']:.3f}  S2(c2+d2)={fx['S2']:.3f}")
        print(f"    予算  : NLM {N_STARTS_C['NLM']}×{N_SEEDS_C['NLM']}  "
              f"LM {N_STARTS_C['LM']}×{N_SEEDS_C['LM']}  "
              f"reg_lambda NLM={REG_LAMBDA_C['NLM']} LM={REG_LAMBDA_C['LM']}")
        est_results = load_estimates_constrained()
        if est_results is not None:
            print("  → 有効なキャッシュを再利用（固定値・予算が一致）")
            for rname, _, _ in regimes:
                res = est_results[rname]
                m = res["metrics"]["overall"]
                th = res["params_free"]
                bnd = f"  ⚠境界: {', '.join(res['at_bounds'])}" if res["at_bounds"] else ""
                print(f"    {rname}: 平均NRMSE={m['mean_NRMSE']:.3f}  "
                      f"θ1={th[6]:.3f} θ2={th[7]:.3f}{bnd}")
        else:
            est_results = {}
            for rname, sl, _ in regimes:
                n_y = len(sl["years"])
                print(f"  推定中: {rname} ({n_y} 年, {N_STARTS_C[rname]}×{N_SEEDS_C[rname]}, "
                      f"reg_lambda={REG_LAMBDA_C[rname]}) ...", flush=True)
                res = estimate_regime_constrained(sl, rname)
                est_results[rname] = res
                m = res["metrics"]["overall"]
                th = res["params_free"]
                print(f"    平均R²={m['mean_R2']:+.3f}  平均NRMSE={m['mean_NRMSE']:.3f}  "
                      f"θ1={th[6]:.3f} θ2={th[7]:.3f}（S の c/d 配分比）")
                if res["at_bounds"]:
                    print(f"    ⚠ 境界張り付き: {', '.join(res['at_bounds'])}")
            print(f"  → 推定結果を保存: {save_estimates_constrained(est_results)}")
    else:
        print(f"\n[Step 1] ODE パラメータ推定 （レジーム別設定）"
              f"\n    NLM: n_starts={N_STARTS['NLM']} n_seeds={N_SEEDS['NLM']} reg_lambda={REG_LAMBDA['NLM']}"
              f"\n    LM : n_starts={N_STARTS['LM']} n_seeds={N_SEEDS['LM']} reg_lambda={REG_LAMBDA['LM']}")
        est_results = {}
        for rname, sl, _ in regimes:
            n_y = len(sl["years"])
            print(f"  推定中: {rname} ({n_y} 年, n_starts={N_STARTS[rname]}, "
                  f"n_seeds={N_SEEDS[rname]}, reg_lambda={REG_LAMBDA[rname]}) ...", flush=True)
            res = estimate_regime(sl, rname)
            est_results[rname] = res
            m = res["metrics"]["overall"]
            print(f"    平均R²={m['mean_R2']:+.3f}  平均NRMSE={m['mean_NRMSE']:.3f}")
            if res["at_bounds"]:
                print(f"    ⚠ 境界張り付き: {', '.join(res['at_bounds'])}")

        print(f"  → 推定結果を保存: {save_estimates(est_results)}")

    # ------------------------------------------------------------------
    # Step 2: 戦略的 MSY（レジーム全期間）
    # ------------------------------------------------------------------
    print(f"\n[Step 2] 戦略的 MSY（レジーム全期間）")
    strategic = {}
    sweep_res_list = []
    grid_res_list  = []
    sens_res_list  = []
    # 制約 MSY 点での margins（コンソール表示用）
    sustain_margins_dict = {}

    for rname, sl, _ in regimes:
        est  = est_results[rname]
        pn   = est["params_norm"]
        mn   = est["means"]
        T    = get_regime_T(sl)
        X0n  = get_regime_X0_norm(sl, mn)

        print(f"\n  ── {rname}  T={T:.1f} 年 ──")

        # スイープ 1: 共通漁獲率（制約付き）
        print(f"  1. 共通漁獲率スイープ ({N_COMMON} 点, 制約付き) ...", flush=True)
        sweep = scan_common_rate(pn, mn, T, X0n, sustain=SUSTAIN_CFG)
        sweep_res_list.append(sweep)
        print(f"     最大収量（無制約）: {sweep['best_yield']:.3f} 千トン/年  "
              f"at f_common={sweep['best_f']:.3f}")
        print(f"     最大収量（制約）  : {sweep['best_yield_constrained']:.3f} 千トン/年  "
              f"at f_common={sweep['best_f_constrained']:.3f}")

        # スイープ 2: 4 次元グリッド探索（制約付き、N_GRID_STRATEGIC=8）
        print(f"  2. グリッド探索 ({N_GRID_STRATEGIC}^4={N_GRID_STRATEGIC**4} 評価, 制約付き) ...",
              flush=True)
        grid = grid_search_msy(pn, mn, T, X0n,
                                n_grid=N_GRID_STRATEGIC, sustain=SUSTAIN_CFG)
        grid_res_list.append(grid)

        # 制約 MSY 点での margins を取得（コンソール表示・average_yield を 1回追加実行）
        margins_con = None
        f_con = grid["f_star_constrained"]
        if np.isfinite(grid["msy_constrained"]):
            _res_con = average_yield(f_con, pn, mn, T, X0n)
            if _res_con["success"]:
                sc = check_sustainability(_res_con["traj_abs"], **SUSTAIN_CFG)
                margins_con = sc["margins"]
        sustain_margins_dict[rname] = margins_con

        print_strategic_result(rname, model_str, T, grid,
                               sustain_cfg=SUSTAIN_CFG,
                               sustain_margins=margins_con)

        # スイープ 3: 種別感度（制約 f* を基準点にする）
        print(f"  3. 種別感度スイープ ({N_SENS} 点 × 4 種, 基準: 制約 f*) ...", flush=True)
        f_base = f_con if np.isfinite(grid["msy_constrained"]) else grid["f_star"]
        sens = species_sensitivity(f_base, pn, mn, T, X0n)
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
        tac = tactical_msy_per_year(sl, est["params_norm"], est["means"])
        tactical[rname] = tac
        print_tactical_summary(rname, model_str, tac)

    # ------------------------------------------------------------------
    # Step 4: PNG 出力
    # ------------------------------------------------------------------
    print(f"\n[Step 4] PNG 出力")
    plot_sensitivity(sens_res_list[0], sens_res_list[1], model_str)
    plot_tactical(tactical["NLM"], tactical["LM"], model_str)
    plot_grid_scatter_constrained(grid_res_list[0], grid_res_list[1], model_str)
    plot_common_sweep_constrained(sweep_res_list, model_str)
    plot_nlm_lm_comparison_constrained(grid_res_list[0], grid_res_list[1], model_str)

    # ------------------------------------------------------------------
    # Step 5: 全体サマリ
    # ------------------------------------------------------------------
    print("\n" + _sep())
    print("[全体サマリ]")
    print(_sep("-"))
    print(f"{'レジーム':>6}  {'無制約MSY':>12}  {'制約MSY':>10}  "
          f"持続可能点/評価点  制約 f*（x1, x2, y1, y2）")
    for rname, data in strategic.items():
        g       = data["grid"]
        f_unc   = g["f_star"]
        f_con   = g["f_star_constrained"]
        msy_unc = g["msy"]
        msy_con = g["msy_constrained"]
        n_feas  = g["n_feasible"]
        n_eval  = g["n_evaluated"]
        f_con_str = (f"{f_con[0]:.3f}  {f_con[1]:.3f}  {f_con[2]:.3f}  {f_con[3]:.3f}"
                     if np.isfinite(msy_con) else "N/A")
        print(f"  {rname:>4}  {msy_unc:>12.3f}  {msy_con:>10.3f}  "
              f"{n_feas:>7d} / {n_eval:<6d}  {f_con_str}")
    print(_sep())
    print("MSY 計算完了。")


if __name__ == "__main__":
    main()
