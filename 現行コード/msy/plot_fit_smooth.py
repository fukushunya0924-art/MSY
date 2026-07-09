"""
推定結果を「積分結果だけの滑らかな1本」で描く。

従来 (run_rank4.py) は軌道を年次点だけでサンプリングして直線接続していたため
折れ線（カクカク）になっていた。ここでは推定後に細かい時間グリッド
(t_eval=linspace(t0, tend, N_FINE)) で ODE を解き直し、滑らかな曲線1本を描く。

  被食者(x): マイワシ x1, ウルメイワシ x2
  捕食者(y): ブリ y1, サワラ y2   （capacity_ry 12変数）

使い方:
  cd 現行コード/msy && python3 plot_fit_smooth.py
出力:
  現行コード/msy/fit_smooth_capacity_ry.png
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

from data_loader import load_clean_dataframe, get_series, SPECIES_LABELS, KEYS
from model import estimate_robust, make_ode, simulate

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Hiragino Sans", "DejaVu Sans", "Arial", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

N_FINE = 300   # 滑らか描画用の時間グリッド点数

# レジーム別正則化強度（run_msy.py の REG_LAMBDA と同じ方針）:
#   NLM は 11 点・12 変数で識別性が保てるため正則化不要
#   LM  は  8 点・12 変数で識別性が弱いため安定化
REG_LAMBDA = {"NLM": 0.0, "LM": 0.005}


def regime_masks(series):
    y = series["years"]
    return ((y >= 2006) & (y <= 2016), (y >= 2017) & (y <= 2024))


def slice_series(series, mask):
    return {k: v[mask] for k, v in series.items()}


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
    results = {}
    for name, sl in regimes.items():
        reg_lambda = REG_LAMBDA[name]
        res = estimate_robust(sl, n_starts=64, reg_lambda=reg_lambda, n_seeds=12, seed0=0)
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
    print(f"\n図を保存: {os.path.join(_here, 'fit_smooth_capacity_ry.png')}")


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
    fig.suptitle("マイワシ+ウルメイワシ / ブリ+サワラ — capacity_ry（積分結果の滑らか軌道）",
                 fontsize=14, y=1.003)
    plt.tight_layout()
    plt.savefig(os.path.join(_here, "fit_smooth_capacity_ry.png"),
                dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
