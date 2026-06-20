"""
6.8 の推定を「ベースライン（元コード相当）」と「改善版」で走らせ、
適合度を数値とグラフで比較する。

ベースライン : 単一スタート・正則化なし（元の 6.8 スクリプトと同じ条件）
改善版       : マルチスタート + 軽い正則化（過剰適合・局所解を抑制）

使い方:
  python3 run_estimation.py              # full(16変数) モデル
  python3 run_estimation.py capacity     # capacity(10変数) モデル
出力:
  - コンソールに R^2 / NRMSE / 境界張り付き警告の比較
  - improved/fit_<model>.png
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loader import load_clean_dataframe, get_series, SPECIES_LABELS, KEYS
from model import estimate

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Hiragino Sans", "DejaVu Sans", "Arial", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False


def slice_series(series, mask):
    return {k: v[mask] for k, v in series.items()}


def regime_masks(series):
    years = series["years"]
    return ((years >= 2006) & (years <= 2016),
            (years >= 2017) & (years <= 2024))


def fmt(m):
    return " | ".join(f"{k}: R²={m[k]['R2']:+.3f}" for k in KEYS)


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "full"
    print(f"モデル: {model}")

    df = load_clean_dataframe()
    series = get_series(df)
    nlm_mask, lm_mask = regime_masks(series)
    regimes = {
        "NLM (非大蛇行 2006-2016)": slice_series(series, nlm_mask),
        "LM  (大蛇行   2017-2024)": slice_series(series, lm_mask),
    }
    # 正則化強度はパラメータ数の多い full をやや強めに
    reg = 0.02 if model == "full" else 0.005

    results = {}
    for name, sl in regimes.items():
        print(f"\n{'='*72}\nレジーム: {name}  (年数 {len(sl['years'])})\n{'='*72}")

        base = estimate(sl, model=model, n_starts=1, reg_lambda=0.0, seed=0)
        imp = estimate(sl, model=model, n_starts=40, reg_lambda=reg, seed=0)

        print("[ベースライン 単一スタート・正則化なし]")
        print("   " + fmt(base["metrics"]) +
              f"\n   全体: 平均R²={base['metrics']['overall']['mean_R2']:+.3f}  "
              f"平均NRMSE={base['metrics']['overall']['mean_NRMSE']:.3f}  "
              f"cost={base['cost']:.3f}")
        if base["at_bounds"]:
            print("   ⚠ 境界張り付き:", ", ".join(base["at_bounds"]))

        print("[改善版 マルチスタート+正則化]")
        print("   " + fmt(imp["metrics"]) +
              f"\n   全体: 平均R²={imp['metrics']['overall']['mean_R2']:+.3f}  "
              f"平均NRMSE={imp['metrics']['overall']['mean_NRMSE']:.3f}  "
              f"cost={imp['cost']:.3f}")
        if imp["at_bounds"]:
            print("   ⚠ 境界張り付き:", ", ".join(imp["at_bounds"]))

        results[name] = {"base": base, "imp": imp, "slice": sl}

    plot(results, model)
    out = f"fit_{model}.png"
    print(f"\nグラフを improved/{out} に保存しました。")


def plot(results, model):
    names = list(results.keys())
    fig, axes = plt.subplots(4, 2, figsize=(15, 17))
    for col, name in enumerate(names):
        sl = results[name]["slice"]
        base = results[name]["base"]
        imp = results[name]["imp"]
        years = sl["years"]
        for row in range(4):
            ax = axes[row, col]
            obs = sl[KEYS[row]]
            ax.plot(years, obs, "ko", ms=7, label="実データ", zorder=5)
            ax.plot(years, base["trajectory_abs"][row], "r--", lw=1.6,
                    label=f"ベースライン (R²={base['metrics'][KEYS[row]]['R2']:.2f})")
            ax.plot(years, imp["trajectory_abs"][row], "b-", lw=2.4,
                    label=f"改善版 (R²={imp['metrics'][KEYS[row]]['R2']:.2f})")
            ax.set_title(f"{name.split('(')[0].strip()}: {SPECIES_LABELS[row]}")
            ax.set_ylabel("資源量（千トン）")
            ax.grid(True, ls="--", alpha=0.5)
            ax.legend(fontsize=8)
    fig.suptitle(f"黒潮レジーム別フィット比較 — {model} モデル", fontsize=14, y=1.005)
    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"fit_{model}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
