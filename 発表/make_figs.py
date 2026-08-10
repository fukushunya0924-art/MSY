"""発表用の図を作る。

方針（2026-08-08 変更）:
  **画像に焼き込むのは数値だけ。** 見出し・軸の名前・凡例の言葉・注記など
  日本語のテキストは一切描かず、位置だけを labels.json に書き出す。
  build_deck.py がそれを読んで PowerPoint のテキストボックスとして置くので、
  あとから文言を直せる。

  位置は「図の左下を(0,0)、右上を(1,1)とした割合」で記録する。
  bbox_inches は使わない（使うと切り取りで割合がずれるため）。

実行: cd 発表 && python3 make_figs.py
出力: figs/*.png と figs/labels.json
"""
import json
import os
import sys
import itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from scipy.interpolate import interp1d

MSY_DIR = "/Users/fukuokashunya/Desktop/MSY/現行コード/msy"
sys.path.insert(0, MSY_DIR)
sys.path.append(os.path.dirname(MSY_DIR))

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(OUT, exist_ok=True)

import pickle
import sustainability as su
from data_loader import (load_clean_dataframe, get_series, KEYS,
                         slice_series, regime_masks)
from model import make_ode, simulate

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["figure.facecolor"] = "white"

NAVY = "#1F4E79"
ORANGE = "#C0561A"
GRAY = "#6E6E6E"
LGRAY = "#BFBFBF"
BAND_A = "#E7EEF6"
BAND_B = "#F7EDE2"
INK = "#222222"

JP = ["マアジ", "ウルメイワシ", "ブリ", "サワラ"]

LABELS = {}     # 図の名前 -> テキストの一覧
SIZES = {}      # 図の名前 -> (幅, 高さ) インチ


def put(name, x, y, text, size=13, color=INK, bold=False, ha="c", va="c",
        w=2.0, lines=1):
    """図の割合座標 (x, y) にテキストを置くよう記録する（描画はしない）。"""
    LABELS.setdefault(name, []).append(
        dict(x=float(x), y=float(y), text=text, size=size,
             color=color.lstrip("#"), bold=bold, ha=ha, va=va, w=w,
             lines=lines))


def ax_put(name, ax, ax_x, ax_y, text, **kw):
    """軸の割合座標で置く（0,0=軸の左下）。"""
    p = ax.get_position()
    put(name, p.x0 + ax_x * p.width, p.y0 + ax_y * p.height, text, **kw)


def title_of(name, ax, text, dy=0.022, size=13, **kw):
    p = ax.get_position()
    put(name, (p.x0 + p.x1) / 2, p.y1 + dy, text, size=size, ha="c", va="b", **kw)


def xlabel_of(name, ax, text, dy=0.10, size=12, **kw):
    p = ax.get_position()
    put(name, (p.x0 + p.x1) / 2, p.y0 - dy, text, size=size, ha="c", va="t", **kw)


def save(fig, name, w, h):
    SIZES[name] = (w, h)
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=200)
    plt.close(fig)
    print("saved:", name)


# ---------------------------------------------------------------- データ
df = load_clean_dataframe()
series = get_series(df)
nlm_mask, lm_mask = regime_masks(series)
SL = {"NLM": slice_series(series, nlm_mask), "LM": slice_series(series, lm_mask)}
def _est(fname):
    return pickle.load(open(os.path.join(MSY_DIR, fname), "rb"))["est_results"]


EST = _est("estimates_capacity_ry.pkl")

# 自由度を落とした2版（当てはまり図で重ねる）
# (見出し, 推定結果, 色, 線の種類, 太さ)
VERSIONS = [
    ("12個（自由）", EST, NAVY, (0, ()), 2.4),
    ("10個", _est("estimates_capacity_ry_constrained_v2_rx_only_10free.pkl"),
     "#5E8FB5", (0, (5, 2.4)), 2.0),
    ("8個", _est("estimates_マアジ_capacity_ry_constrained_v1_8var.pkl"),
     "#7E8B96", (0, (1.2, 2.0)), 2.1),
]


def smooth_traj(sl, res, n=300):
    years = sl["years"]
    t_rel = (years - years.min()).astype(float)
    means = res["means"]
    init = [sl[KEYS[i]][0] / means[i] for i in range(4)]
    f_interp = [interp1d(t_rel, sl["f" + KEYS[i]], kind="linear",
                         fill_value="extrapolate") for i in range(4)]
    ode = make_ode(*f_interp)
    t_fine = np.linspace(t_rel[0], t_rel[-1], n)
    yn = simulate(res["params_norm"], ode, t_fine, init)
    return years.min() + t_fine, np.vstack([yn[i] * means[i] for i in range(4)])


# ============================================================ 01 資源量の推移
def fig_timeseries():
    name, W, H = "01_資源量の移り変わり", 12.09, 3.05
    fig, axes = plt.subplots(1, 4, figsize=(W, H))
    fig.subplots_adjust(left=0.040, right=0.995, top=0.845, bottom=0.215,
                        wspace=0.24)
    yrs = series["years"]
    for i, ax in enumerate(axes):
        ax.axvspan(2006, 2016, color=BAND_A, zorder=0)
        ax.axvspan(2017, 2024, color=BAND_B, zorder=0)
        ax.plot(yrs, series[KEYS[i]], "-o", color=NAVY, ms=3.0, lw=1.5)
        ax.set_xlim(1994, 2025)
        ax.set_xticks([2000, 2010, 2020])
        ax.tick_params(labelsize=10)
        ax.grid(True, ls="--", alpha=0.35)
        title_of(name, ax, JP[i], size=13.5)
    ax_put(name, axes[0], 0.42, 0.90, "大蛇行なし", size=10.5, color=GRAY, w=1.1)
    ax_put(name, axes[0], 0.86, 0.90, "大蛇行", size=10.5, color=ORANGE, w=0.8)
    put(name, 0.5, 0.035,
        "縦軸＝資源量（千トン）　／　青い帯＝大蛇行なし（2006〜2016年）　／　橙の帯＝大蛇行（2017〜2024年）",
        size=11.5, color=GRAY, ha="c", va="c", w=8.5)
    save(fig, name, W, H)


# ============================================================ 02 山の図
def fig_hill():
    name, W, H = "02_山ができる", 5.7, 3.35
    fig, ax = plt.subplots(figsize=(W, H))
    fig.subplots_adjust(left=0.115, right=0.975, top=0.885, bottom=0.205)
    r, K = 0.6, 1000.0
    f = np.linspace(0, r, 300)
    ax.plot(f, K * f - (K / r) * f ** 2, color=NAVY, lw=3)
    ax.plot([r / 2], [K * r / 4], "o", color=ORANGE, ms=11, zorder=5)
    ax.annotate("", xy=(r / 2, K * r / 4), xytext=(r * 0.70, K * r / 4 * 0.66),
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.8))
    ax.set_ylim(0, 178)
    ax.tick_params(labelsize=10)
    ax.grid(True, ls="--", alpha=0.35)
    title_of(name, ax, "1種だけのモデル：山ができる", size=13.5, color=NAVY)
    xlabel_of(name, ax, "漁獲率 f（1年に資源量の何割をとるか）", dy=0.115, size=12)
    ax_put(name, ax, 0.03, 0.93, "1年にとれる量", size=11.5, color=GRAY,
           ha="l", w=1.5)
    ax_put(name, ax, 0.78, 0.55, "ここが最大\n（MSY）", size=12.5, color=ORANGE,
           ha="c", w=1.3, lines=2)
    save(fig, name, W, H)


# ============================================================ 05 当てはまり
def fig_fit():
    name, W, H = "05_当てはまり", 11.0, 4.55
    fig, axes = plt.subplots(2, 4, figsize=(W, H))
    fig.subplots_adjust(left=0.105, right=0.995, top=0.845, bottom=0.135,
                        wspace=0.30, hspace=0.42)
    for row, reg in enumerate(["NLM", "LM"]):
        sl = SL[reg]
        band = BAND_A if reg == "NLM" else BAND_B
        traj = [smooth_traj(sl, est[reg]) for _, est, _, _, _ in VERSIONS]
        for i in range(4):
            ax = axes[row, i]
            ax.set_facecolor(band)
            # 固定が多い版から先に描いて、自由推定を一番上に重ねる
            for k in range(len(VERSIONS) - 1, -1, -1):
                _, _, col, dash, lw = VERSIONS[k]
                yf, tf = traj[k]
                ax.plot(yf, tf[i], color=col, lw=lw, ls=dash, zorder=3 + k,
                        solid_capstyle="round", dash_capstyle="round")
            ax.plot(sl["years"], sl[KEYS[i]], "o", color=INK, ms=5.0, zorder=8)
            ax.set_xticks([2007, 2011, 2015] if reg == "NLM" else [2018, 2021, 2024])
            ax.tick_params(labelsize=9)
            ax.grid(True, ls="--", alpha=0.3, color="white")
            for s in ax.spines.values():
                s.set_color("#CCCCCC")
            # 数字を置くぶんの余白を上に取る
            top = max([sl[KEYS[i]].max()] + [t[i].max() for _, t in traj])
            ax.set_ylim(0, top * 1.34)
            title_of(name, ax, JP[i], dy=0.012, size=12)
            for k, (_, est, col, _, _) in enumerate(VERSIONS):
                nr = est[reg]["metrics"][KEYS[i]]["NRMSE"]
                ax_put(name, ax, 0.05 + k * 0.33, 0.925, f"{nr:.3f}",
                       size=9.5, color=col, ha="l", w=0.44)
    # 凡例（印と線は図に描き、言葉はテキストボックス）
    fig.add_artist(Line2D([0.318], [0.966], marker="o", color=INK, ms=6,
                          lw=0, transform=fig.transFigure))
    put(name, 0.331, 0.966, "実データ", size=11.5, ha="l", w=1.0)
    for x0, (lab, _, col, dash, lw) in zip([0.432, 0.600, 0.712], VERSIONS):
        fig.add_artist(Line2D([x0, x0 + 0.030], [0.966, 0.966], color=col,
                              lw=lw + 0.3, ls=dash,
                              transform=fig.transFigure))
        put(name, x0 + 0.037, 0.966, lab, size=11.5, color=col, ha="l", w=1.3)
    put(name, 0.004, 0.978, "縦軸：資源量（千トン）", size=11.5, color=GRAY,
        ha="l", va="t", w=2.2)
    put(name, 0.006, 0.680, "大蛇行\nなし", size=12, color=GRAY, ha="l", va="c",
        w=0.7, lines=2)
    put(name, 0.006, 0.240, "大蛇行", size=12, color=ORANGE, ha="l", va="c", w=0.7)
    put(name, 0.5, 0.028,
        "「ずれ」＝平均で割った誤差（小さいほど良い）。枠の中の数字は左から 12個・10個・8個。",
        size=11.5, color=GRAY, ha="c", w=7.6)
    save(fig, name, W, H)


# ============================================================ 06 ずれの階段
def fig_ladder():
    name, W, H = "06_固定を増やすと悪化", 6.9, 3.35
    fig, ax = plt.subplots(figsize=(W, H))
    fig.subplots_adjust(left=0.105, right=0.985, top=0.865, bottom=0.245)
    nlm, lm = [0.099, 0.293, 0.452], [0.065, 0.170, 0.338]
    x = np.arange(3); w = 0.36
    ax.bar(x - w / 2, nlm, w, color=NAVY)
    ax.bar(x + w / 2, lm, w, color=ORANGE)
    for xi, v in zip(x - w / 2, nlm):
        ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=11, color=NAVY)
    for xi, v in zip(x + w / 2, lm):
        ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=11, color=ORANGE)
    ax.set_xticks(x); ax.set_xticklabels([""] * 3)
    ax.set_ylim(0, 0.66)
    ax.tick_params(labelsize=10)
    ax.grid(True, axis="y", ls="--", alpha=0.35)
    ax.annotate("", xy=(2.32, 0.505), xytext=(0.05, 0.16),
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=2.0, ls="--"))
    ax.add_patch(Rectangle((-0.44, 0.605), 0.16, 0.035, color=NAVY))
    ax.add_patch(Rectangle((-0.44, 0.535), 0.16, 0.035, color=ORANGE))
    ax_put(name, ax, 0.045, 0.933, "大蛇行なし", size=11, ha="l", w=1.1)
    ax_put(name, ax, 0.045, 0.827, "大蛇行", size=11, ha="l", w=1.1)
    for xi, t in zip(x, ["自由に決める\n（12個）", "増える速さを\n固定（10個）",
                         "さらに変換の\n割合も固定（8個）"]):
        ax_put(name, ax, (xi + 0.5) / 3, -0.115, t, size=11.5, ha="c", va="t",
               w=1.85, lines=2)
    put(name, 0.012, 0.985, "縦軸：当てはまりのずれ（小さいほど良い）",
        size=11.5, color=GRAY, ha="l", va="t", w=3.4)
    ax_put(name, ax, 0.52, 0.83, "固定する係数を増やすほど、ずれが大きくなる",
           size=11.5, color=GRAY, ha="c", w=4.0)
    save(fig, name, W, H)


# ============================================================ 07 r と f
def fig_r_vs_f():
    name, W, H = "07_増える速さと漁獲率", 6.75, 3.15
    fig, ax = plt.subplots(figsize=(W, H))
    fig.subplots_adjust(left=0.105, right=0.985, top=0.955, bottom=0.155)
    r_fix = [0.228, 0.739]
    f_obs = [float(series["fx1"].mean()), float(series["fx2"].mean())]
    x = np.arange(2); w = 0.36
    ax.bar(x - w / 2, r_fix, w, color=NAVY)
    ax.bar(x + w / 2, f_obs, w, color=ORANGE)
    for xi, v in zip(x - w / 2, r_fix):
        ax.text(xi, v + 0.02, f"{v:.3f}", ha="center", fontsize=11.5, color=NAVY)
    for xi, v in zip(x + w / 2, f_obs):
        ax.text(xi, v + 0.02, f"{v:.3f}", ha="center", fontsize=11.5, color=ORANGE)
    ax.set_xticks(x); ax.set_xticklabels([""] * 2)
    ax.set_ylim(0, 1.30)
    ax.tick_params(labelsize=10)
    ax.grid(True, axis="y", ls="--", alpha=0.35)
    ax.add_patch(Rectangle((-0.45, 0.0), 0.9, 0.52, facecolor="none",
                           edgecolor=ORANGE, lw=2.0, ls="--"))
    ax.add_patch(Rectangle((0.38, 1.19), 0.07, 0.055, color=NAVY))
    ax.add_patch(Rectangle((0.38, 1.06), 0.07, 0.055, color=ORANGE))
    ax_put(name, ax, 0.44, 0.935, "1種ずつの解析から借りた「増える速さ」",
           size=10.5, ha="l", w=3.3)
    ax_put(name, ax, 0.44, 0.835, "実際にとっている割合（1994〜2024年）",
           size=10.5, ha="l", w=3.3)
    for xi, t in zip(x, ["マアジ", "ウルメイワシ"]):
        ax_put(name, ax, (xi + 0.5) / 2, -0.055, t, size=12.5, ha="c", va="t", w=1.6)
    ax_put(name, ax, 0.25, 0.50, "マアジだけ逆転\n（とる量のほうが多い）",
           size=12, color=ORANGE, ha="c", w=2.1, lines=2)
    put(name, 0.012, 0.985, "縦軸：1年あたりの割合", size=11.5, color=GRAY,
        ha="l", va="t", w=2.2)
    save(fig, name, W, H)


# ============================================================ 08 時期の比較
def fig_regime():
    name, W, H = "08_時期ごとの比較", 11.4, 3.30
    fig, axes = plt.subplots(1, 2, figsize=(W, H))
    fig.subplots_adjust(left=0.095, right=0.985, top=0.865, bottom=0.175,
                        wspace=0.30)
    ax = axes[0]
    nlm = [64.4, 243.5, 289.7, 4.82]
    lm = [46.3, 203.6, 365.0, 7.13]
    chg = [(b - a) / a * 100 for a, b in zip(nlm, lm)]
    cols = [NAVY, NAVY, ORANGE, ORANGE]
    ax.barh(np.arange(4), chg, color=cols, height=0.55)
    for i, v in enumerate(chg):
        ax.text(v + (3 if v > 0 else -3), i, f"{v:+.0f}%", va="center",
                ha="left" if v > 0 else "right", fontsize=12, color=cols[i])
    ax.set_yticks(np.arange(4)); ax.set_yticklabels([""] * 4)
    ax.invert_yaxis()
    ax.axvline(0, color=INK, lw=1.2)
    ax.set_xlim(-45, 65)
    ax.tick_params(labelsize=10)
    ax.grid(True, axis="x", ls="--", alpha=0.35)
    title_of(name, ax, "餌の魚は減り、食べる魚は増えた", size=13)
    xlabel_of(name, ax, "大蛇行なし → 大蛇行 の変化（期間平均の資源量）",
              dy=0.085, size=11)
    for i, t in enumerate(JP):
        ax_put(name, ax, -0.02, 1 - (i + 0.5) / 4, t, size=12, ha="r", w=1.6)

    ax = axes[1]
    a = [0.305, 0.272, 0.271, 0.169]
    b = [0.230, 0.120, 0.686, 0.0025]
    x = np.arange(4); w = 0.36
    ax.bar(x - w / 2, a, w, color=LGRAY)
    ax.bar(x + w / 2, b, w, color=ORANGE)
    ax.set_xticks(x); ax.set_xticklabels([""] * 4)
    ax.set_ylim(0, 0.83)
    ax.tick_params(labelsize=10)
    ax.grid(True, axis="y", ls="--", alpha=0.35)
    ax.add_patch(Rectangle((2.55, 0.780), 0.13, 0.032, color=LGRAY))
    ax.add_patch(Rectangle((2.55, 0.700), 0.13, 0.032, color=ORANGE))
    title_of(name, ax, "サワラの取り分がウルメからマアジへ移った", size=13)
    for i, t in enumerate(["マアジ→ブリ", "ウルメ→ブリ", "マアジ→サワラ",
                           "ウルメ→サワラ"]):
        ax_put(name, ax, (i + 0.5) / 4, -0.055, t, size=10.5, ha="c", va="t", w=1.5)
    ax_put(name, ax, 0.685, 0.952, "大蛇行なし", size=10.5, ha="l", w=1.1)
    ax_put(name, ax, 0.685, 0.855, "大蛇行", size=10.5, ha="l", w=1.1)
    ax_put(name, ax, 0.905, 0.045, "ほぼ0", size=10.5, color=ORANGE, ha="c", w=0.8)
    put(name, ax.get_position().x1, 0.030, "縦軸：食べて増える強さ", size=11,
        color=GRAY, ha="r", va="c", w=2.3)
    save(fig, name, W, H)


# ============================================================ 09 上限に張り付く
def fig_upper():
    name, W, H = "09_上限に張り付く", 12.09, 3.55
    fig, axes = plt.subplots(1, 2, figsize=(W, H))
    fig.subplots_adjust(left=0.070, right=0.985, top=0.865, bottom=0.175,
                        wspace=0.24)
    ub = [0.25, 0.50, 0.75, 0.95, 1.25]
    nlm = [206.7, 380.6, 607.2, 801.4, 1094.0]
    lm = [140.6, 203.6, 246.0, 282.2, 345.5]
    ax = axes[0]
    ax.plot(ub, nlm, "-o", color=NAVY, lw=2.6, ms=8)
    ax.plot(ub, lm, "-s", color=ORANGE, lw=2.6, ms=8)
    for u, v in zip(ub, nlm):
        ax.text(u, v + 48, f"{v:.0f}", ha="center", fontsize=11, color=NAVY)
    ax.set_ylim(0, 1290)
    ax.tick_params(labelsize=10)
    ax.grid(True, ls="--", alpha=0.35)
    ax.add_patch(Rectangle((0.30, 1188), 0.055, 45, color=NAVY, clip_on=False))
    ax.add_patch(Rectangle((0.30, 1058), 0.055, 45, color=ORANGE, clip_on=False))
    title_of(name, ax, "上限を上げるほど、答えも上がり続ける", size=13.5)
    xlabel_of(name, ax, "探索で許した漁獲率の上限", dy=0.085, size=12)
    put(name, 0.004, 0.030, "縦軸：とれる量（千トン）", size=11,
        color=GRAY, ha="l", va="c", w=2.0)
    ax_put(name, ax, 0.11, 0.930, "大蛇行なし", size=11, ha="l", w=1.1)
    ax_put(name, ax, 0.11, 0.828, "大蛇行", size=11, ha="l", w=1.1)

    ax = axes[1]
    f_opt = [0.0, 0.0, 0.7125, 0.95]
    at_edge = [True, True, False, True]
    x = np.arange(4)
    ax.bar(x, np.maximum(f_opt, 0.006), 0.5,
           color=[ORANGE if e else NAVY for e in at_edge])
    ax.axhline(0.95, color=ORANGE, ls="--", lw=2.0)
    for xi, v, e in zip(x, f_opt, at_edge):
        ax.text(xi, v + 0.04, f"{v:.3f}", ha="center", fontsize=11.5,
                color=ORANGE if e else INK)
    ax.set_xticks(x); ax.set_xticklabels([""] * 4)
    ax.set_ylim(0, 1.18)
    ax.tick_params(labelsize=10)
    ax.grid(True, axis="y", ls="--", alpha=0.35)
    title_of(name, ax, "4種のうち3種が端（0か上限）に寄る", size=13.5)
    ax_put(name, ax, 0.02, 0.845, "上限 0.95", size=11.5, color=ORANGE, ha="l", w=1.3)
    for i, t in enumerate(JP):
        ax_put(name, ax, (i + 0.5) / 4, -0.055, t, size=12, ha="c", va="t", w=1.6)
    put(name, 0.996, 0.030, "縦軸：最適とされた漁獲率", size=11,
        color=GRAY, ha="r", va="c", w=2.4)
    save(fig, name, W, H)


# ============================================================ 11 釣り合いが負
def fig_equilibrium():
    name, W, H = "11_釣り合いの量が負", 9.2, 2.95
    fig, axes = plt.subplots(1, 2, figsize=(W, H))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.845, bottom=0.135,
                        wspace=0.26)
    heads = ["大蛇行なし", "大蛇行"]
    for ax, reg, head in zip(axes, ["NLM", "LM"], heads):
        p, m = EST[reg]["params_norm"], EST[reg]["means"]
        A, rho = su.build_A_rho(p, np.zeros(4))
        B = su.equilibrium_generalized_lv(A, rho)["B_eq"] * m
        cols = [ORANGE if v < 0 else NAVY for v in B]
        ax.bar(np.arange(4), B, 0.55, color=cols)
        for i, v in enumerate(B):
            ax.text(i, v + (30 if v > 0 else -30), f"{v:.0f}", ha="center",
                    va="bottom" if v > 0 else "top", fontsize=11, color=cols[i])
        ax.axhline(0, color=INK, lw=1.3)
        ax.set_xticks(np.arange(4)); ax.set_xticklabels([""] * 4)
        ax.tick_params(labelsize=10)
        ax.grid(True, axis="y", ls="--", alpha=0.35)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo - abs(lo) * 0.30 - 45, hi + abs(hi) * 0.20 + 45)
        title_of(name, ax, head, size=12.5)
        for i, t in enumerate(JP):
            ax_put(name, ax, (i + 0.5) / 4, -0.055, t, size=11, ha="c", va="t",
                   w=1.5)
    ax_put(name, axes[0], 0.03, 0.93, "マアジがマイナス", size=12, color=ORANGE,
           ha="l", w=2.0)
    ax_put(name, axes[1], 0.03, 0.93, "ウルメイワシがマイナス", size=12,
           color=ORANGE, ha="l", w=2.4)
    put(name, 0.004, 0.998, "縦軸：資源量（千トン）", size=11,
        color=GRAY, ha="l", va="t", w=2.0)
    save(fig, name, W, H)


# ============================================================ 12 角と内側
def fig_vertex():
    name, W, H = "12_角と内側", 8.1, 3.35
    fig, ax = plt.subplots(figsize=(W, H))
    fig.subplots_adjust(left=0.095, right=0.985, top=0.865, bottom=0.135)
    rng = np.random.default_rng(0)
    rows = []
    for reg in ["NLM", "LM"]:
        p, m = EST[reg]["params_norm"], EST[reg]["means"]

        def Yeq(f):
            A, rho = su.build_A_rho(p, np.asarray(f, float))
            B = su.equilibrium_generalized_lv(A, rho)["B_eq"] * m
            return float(np.dot(np.asarray(f, float), B))
        vt = max(Yeq(np.array(v) * 0.95) for v in itertools.product([0, 1], repeat=4))
        it = max(Yeq(q) for q in rng.uniform(0, 0.95, size=(20000, 4)))
        rows.append((vt, it))
    x = np.arange(2); w = 0.36
    ax.bar(x - w / 2, [r[1] for r in rows], w, color=LGRAY)
    ax.bar(x + w / 2, [r[0] for r in rows], w, color=NAVY)
    for i, r in enumerate(rows):
        ax.text(i - w / 2, r[1] + 25, f"{r[1]:.0f}", ha="center", fontsize=11.5,
                color=GRAY)
        ax.text(i + w / 2, r[0] + 25, f"{r[0]:.0f}", ha="center", fontsize=11.5,
                color=NAVY)
    ax.set_xticks(x); ax.set_xticklabels([""] * 2)
    ax.set_ylim(0, 2850)
    ax.tick_params(labelsize=10)
    ax.grid(True, axis="y", ls="--", alpha=0.35)
    ax.add_patch(Rectangle((0.30, 2655), 0.09, 90, color=LGRAY))
    ax.add_patch(Rectangle((0.30, 2400), 0.09, 90, color=NAVY))
    title_of(name, ax, "内側をいくら探しても、角を超えない", size=13.5)
    ax_put(name, ax, 0.57, 0.950, "内側を2万点さがした最大", size=11, ha="l", w=2.6)
    ax_put(name, ax, 0.57, 0.860, "角16点の最大", size=11, ha="l", w=2.6)
    for i, t in enumerate(["大蛇行なし", "大蛇行"]):
        ax_put(name, ax, (i + 0.5) / 2, -0.055, t, size=12.5, ha="c", va="t", w=1.8)
    put(name, 0.004, 0.985, "縦軸：とれる量（千トン）", size=11,
        color=GRAY, ha="l", va="t", w=2.2)
    save(fig, name, W, H)


if __name__ == "__main__":
    fig_timeseries()
    fig_hill()
    fig_fit()
    fig_ladder()
    fig_r_vs_f()
    fig_regime()
    fig_upper()
    fig_equilibrium()
    fig_vertex()
    with open(os.path.join(OUT, "labels.json"), "w") as fh:
        json.dump({"labels": LABELS, "sizes": SIZES}, fh,
                  ensure_ascii=False, indent=1)
    print("\nlabels.json を書き出しました")
