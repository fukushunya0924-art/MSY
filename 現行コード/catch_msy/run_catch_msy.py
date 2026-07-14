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
  python3 run_catch_msy.py --emit-fixed-params
      # 4種を再推定し、現行コード/fixed_params.py に貼り付け可能な _POINT/_CI
      # ブロックを印字（段1: Catch-MSY→制約）。現行値との drift も警告する。
      # ※ 自動上書きはしない。r_x1（マイワシ）は終端レンジ例外の暫定値で
      #   教授相談が要るため、貼り替えは人間が判断する。
"""
import os
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
# fixed_params.py（親フォルダ 現行コード/）を drift チェック用に import できるようにする。
_parent = os.path.dirname(_here)
if _parent not in sys.path:
    sys.path.append(_parent)

# PNG出力先: catch_msy/outputs/
_out_dir = os.path.join(_here, "outputs")
os.makedirs(_out_dir, exist_ok=True)

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


# 段1（Catch-MSY→制約）: Catch-MSY 結果を fixed_params.py の各キーへ写す対応。
#   被食者 sardine/urume の r → r_x1/r_x2（自然増殖率そのもの）
#   捕食者 buri/sawara の r → S1/S2（= c1+d1 / c2+d2 の実効推定値, 解釈A・Phase7）
_SPECIES_TO_FIXED = {"sardine": "r_x1", "urume": "r_x2",
                     "buri": "S1", "sawara": "S2"}
_FIXED_LABEL = {"r_x1": "マイワシ", "r_x2": "ウルメイワシ",
                "S1": "ブリ c1+d1", "S2": "サワラ c2+d2"}
_FIXED_ORDER = ["r_x1", "r_x2", "S1", "S2"]


def emit_fixed_params(results):
    """Catch-MSY 結果を fixed_params.py 貼り付け形で印字し、現行値との drift を警告する。

    results は 4魚種（sardine, urume, buri, sawara）を含む必要がある（1つでも欠けると
    完全な _POINT/_CI ブロックを作れないので、その旨を告げて何もしない）。
    点推定は r_geomean、区間は (r_lo, r_hi)=25/75%点を採用（現行 fixed_params.py と同じ）。
    自動書き込みは一切しない。印字と警告のみ。
    """
    missing = [sp for sp in _SPECIES_TO_FIXED if sp not in results]
    if missing:
        print(f"\n[emit] 4魚種すべてが必要です（不足: {missing}）。"
              f"引数なしの `python3 run_catch_msy.py --emit-fixed-params` を実行してください。")
        return

    point = {_SPECIES_TO_FIXED[sp]: results[sp]["r_geomean"] for sp in _SPECIES_TO_FIXED}
    ci = {_SPECIES_TO_FIXED[sp]: (results[sp]["r_lo"], results[sp]["r_hi"])
          for sp in _SPECIES_TO_FIXED}

    print()
    print("=" * 72)
    print("fixed_params.py 貼り付け用ブロック（Catch-MSY 太平洋12県 の最新推定）")
    print("=" * 72)
    print("_POINT = {")
    for key in _FIXED_ORDER:
        qkey = '"%s":' % key
        print(f"    {qkey:<7} {point[key]:.3f},  # {_FIXED_LABEL[key]}")
    print("}")
    print()
    print("_CI = {")
    for key in _FIXED_ORDER:
        lo, hi = ci[key]
        qkey = '"%s":' % key
        print(f"    {qkey:<7} ({lo:.3f}, {hi:.3f}),")
    print("}")
    print("=" * 72)

    # drift チェック: 現行 fixed_params.py と 3桁で突き合わせ（貼れば変わる箇所を可視化）。
    import fixed_params
    cur_point = fixed_params.get_point()
    print("[drift チェック] 現行 fixed_params.py との差分（3桁比較）:")
    any_drift = False
    for key in _FIXED_ORDER:
        cur, new = cur_point[key], point[key]
        cur_lo, cur_hi = fixed_params.get_ci(key)
        new_lo, new_hi = ci[key]
        if (round(new, 3) != round(cur, 3) or round(new_lo, 3) != round(cur_lo, 3)
                or round(new_hi, 3) != round(cur_hi, 3)):
            any_drift = True
            print(f"  ⚠ {key:<4}: 現行 {cur:.3f} [{cur_lo:.3f},{cur_hi:.3f}]  →  "
                  f"新 {new:.3f} [{new_lo:.3f},{new_hi:.3f}]")
    if not any_drift:
        print("  差分なし（現行 fixed_params.py は最新の Catch-MSY と一致）。")
    else:
        print("  ※ 上記を fixed_params.py に貼るかは人間が判断。特に r_x1（マイワシ）は")
        print("    終端レンジ例外の暫定値で、貼り替えは教授相談の上で。")


def main():
    args = sys.argv[1:]
    emit = "--emit-fixed-params" in args
    species_args = [a for a in args if not a.startswith("--")]
    # emit は完全な _POINT/_CI ブロックを作るため4魚種を固定で回す（種指定は無視）。
    keys = list(MAIN_KEYS) if emit else parse_species_args(species_args,
                                                           default_keys=MAIN_KEYS)
    results = run_all(keys)
    print_table(results)
    if emit:
        emit_fixed_params(results)
    out = os.path.join(_out_dir, "catch_msy_概要_マイワシ_ウルメイワシ_ブリ_サワラ.png")
    plot_overview(results, out)


if __name__ == "__main__":
    main()
