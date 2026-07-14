"""
推定結果を「積分結果だけの滑らかな1本」で描く。

従来 (run_rank4.py) は軌道を年次点だけでサンプリングして直線接続していたため
折れ線（カクカク）になっていた。ここでは推定後に細かい時間グリッド
(t_eval=linspace(t0, tend, N_FINE)) で ODE を解き直し、滑らかな曲線1本を描く。

  被食者(x): マアジ x1, ウルメイワシ x2
  捕食者(y): ブリ y1, サワラ y2   （capacity_ry 12変数）

使い方:
  cd 現行コード/msy && python3 plot_fit_smooth.py
出力:
  現行コード/msy/outputs/fit_マアジ_ウルメイワシ_ブリ_サワラ_capacity_ry.png
  コンソールに R²/NRMSE
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)                       # msy/ の data_loader（ブリ/サワラ版）
sys.path.append(os.path.dirname(_here))         # 現行コード/ の model

# PNG出力先: msy/outputs/
_out_dir = os.path.join(_here, "outputs")
os.makedirs(_out_dir, exist_ok=True)

from data_loader import (
    load_clean_dataframe, get_series, SPECIES_LABELS, KEYS,
    slice_series, regime_masks,
)
from model import make_ode, simulate
from estimate_cache import load_estimates, estimate_regime

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Hiragino Sans", "DejaVu Sans", "Arial", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

N_FINE = 300   # 滑らか描画用の時間グリッド点数
# 探索設定・推定・キャッシュ入出力は estimate_cache.py に集約（run_msy.py と共通）。


def smooth_trajectory(sl, res, n=N_FINE):
    """推定パラメータを細かい時間グリッドで再積分し、絶対スケール軌道を返す。"""
    years = sl["years"]
    t_rel = (years - years.min()).astype(float)
    means = res["means"]
    init = [sl[KEYS[i]][0] / means[i] for i in range(4)]   # 正規化初期値

    f_interp = [interp1d(t_rel, sl["f" + KEYS[i]], kind="linear",
                         fill_value="extrapolate") for i in range(4)]
    ode = make_ode(*f_interp)

    t_fine = np.linspace(t_rel[0], t_rel[-1], n)
    y_norm = simulate(res["params_norm"], ode, t_fine, init)
    traj_abs = np.vstack([y_norm[i] * means[i] for i in range(4)])
    years_fine = years.min() + t_fine
    return years_fine, traj_abs


def main():
    df = load_clean_dataframe()
    series = get_series(df)
    nlm_mask, lm_mask = regime_masks(series)
    regimes = {
        "NLM": slice_series(series, nlm_mask),
        "LM":  slice_series(series, lm_mask),
    }
    cached = load_estimates()
    if cached is not None:
        print("推定結果のキャッシュを再利用（run_msy.py が保存したもの）")

    results = {}
    for name, sl in regimes.items():
        res = cached[name] if cached is not None else estimate_regime(sl, name)
        yrs_fine, traj_fine = smooth_trajectory(sl, res)
        results[name] = {"slice": sl, "res": res,
                         "yrs_fine": yrs_fine, "traj_fine": traj_fine}

        m = res["metrics"]
        print(f"\n=== {name} (年数 {len(sl['years'])}) ===")
        for k, lab in zip(KEYS, SPECIES_LABELS):
            print(f"  {lab:18s} R²={m[k]['R2']:+.3f}  NRMSE={m[k]['NRMSE']:.3f}")
        print(f"  {'全体':18s} 平均R²={m['overall']['mean_R2']:+.3f}  "
              f"平均NRMSE={m['overall']['mean_NRMSE']:.3f}")

    plot(results)
    print(f"\n図を保存: {os.path.join(_out_dir, 'fit_マアジ_ウルメイワシ_ブリ_サワラ_capacity_ry.png')}")


def plot(results):
    names = list(results.keys())
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    for col, name in enumerate(names):
        sl        = results[name]["slice"]
        res       = results[name]["res"]
        yrs_fine  = results[name]["yrs_fine"]
        traj_fine = results[name]["traj_fine"]
        years     = sl["years"]
        for row in range(4):
            ax = axes[row, col]
            ax.plot(years, sl[KEYS[row]], "ko", ms=7, label="実データ", zorder=5)
            ax.plot(yrs_fine, traj_fine[row], "b-", lw=2.2,
                    label=f"推定（積分結果） R²={res['metrics'][KEYS[row]]['R2']:.2f}")
            ax.set_title(f"{name}: {SPECIES_LABELS[row]}")
            ax.set_ylabel("資源量（千トン）")
            ax.grid(True, ls="--", alpha=0.5)
            ax.legend(fontsize=8)
    fig.suptitle("マアジ+ウルメイワシ / ブリ+サワラ — capacity_ry（積分結果の滑らか軌道）",
                 fontsize=14, y=1.003)
    plt.tight_layout()
    plt.savefig(os.path.join(_out_dir, "fit_マアジ_ウルメイワシ_ブリ_サワラ_capacity_ry.png"),
                dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
