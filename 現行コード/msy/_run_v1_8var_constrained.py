"""
旧v1（8自由変数, S1=c1+d1・S2=c2+d2固定 + theta配分）の制約推定を、マアジ版・マイワシ版
両方について実データの固定値で再実行し、消失していた物理パラメータを復元する一回限りの
スクリプト（2026-07-15, docs/research_log.md Phase 13参照）。

v1のコードは git commit f23acb4 時点の 現行コード/model_constrained.py を
model_constrained_v1.py としてこのディレクトリに複製したもの（現行の
model_constrained.py は v2, 10自由変数版へ書き換え済みのため触らない）。

固定値（docs/research_log.md Phase 10/11 に記録された実際の実行値）:
  マアジ  : r_x1=0.228, r_x2=0.739, S1=0.395, S2=0.260 → 目標 NLM 0.452 / LM 0.338
  マイワシ: r_x1=0.940, r_x2=0.739, S1=0.395, S2=0.260 → 目標 NLM 0.408 / LM 0.122

使い方: cd 現行コード/msy && python3 _run_v1_8var_constrained.py
出力:
  現行コード/msy/estimates_<種>_capacity_ry_constrained_v1_8var.pkl（種ごと）
  現行コード/msy/outputs/<種>版/制約_旧8var_S固定/fit_制約8var_復元_<種>_..._constrained.png
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
from data_loader import load_clean_dataframe, get_series, KEYS, slice_series, regime_masks
import model_constrained_v1 as mcv1
from estimate_cache import N_STARTS_C, N_SEEDS_C, REG_LAMBDA_C
from plot_fit_smooth import smooth_trajectory

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Hiragino Sans", "DejaVu Sans", "Arial", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

_out_dir = os.path.join(_here, "outputs")

# 種ごとの設定: (x1のASSIGN先, ラベル, 固定値dict, outputsサブフォルダ名)
SPECIES_CFG = {
    "マアジ":   dict(x1="マアジ",   fixed={"r_x1": 0.228, "r_x2": 0.739, "S1": 0.395, "S2": 0.260}),
    "マイワシ": dict(x1="マイワシ", fixed={"r_x1": 0.940, "r_x2": 0.739, "S1": 0.395, "S2": 0.260}),
}


def cache_path(species):
    return os.path.join(_here, f"estimates_{species}_capacity_ry_constrained_v1_8var.pkl")


def run_species(species):
    cfg = SPECIES_CFG[species]
    labels = [f"{cfg['x1']} (x1)", "ウルメイワシ (x2)", "ブリ (y1)", "サワラ (y2)"]

    data_loader.ASSIGN["x1"] = cfg["x1"]
    df = load_clean_dataframe()
    series = get_series(df)
    nlm_mask, lm_mask = regime_masks(series)
    regimes = [("NLM", slice_series(series, nlm_mask)),
               ("LM", slice_series(series, lm_mask))]

    print("=" * 64)
    print(f"■■■ {species}版（旧v1, 8自由変数, S固定={cfg['fixed']}） ■■■")

    est_results = {}
    for rname, sl in regimes:
        n_y = len(sl["years"])
        print(f"推定中: {species}/{rname} ({n_y}年, {N_STARTS_C[rname]}x{N_SEEDS_C[rname]}, "
              f"reg_lambda={REG_LAMBDA_C[rname]}) ...", flush=True)
        t0 = time.time()
        res = mcv1.estimate_constrained_robust(
            sl, n_starts=N_STARTS_C[rname], reg_lambda=REG_LAMBDA_C[rname],
            n_seeds=N_SEEDS_C[rname], seed0=0, fixed=cfg["fixed"])
        dt = time.time() - t0
        est_results[rname] = res
        m = res["metrics"]["overall"]
        print(f"  完了 ({dt:.0f}s)  平均R²={m['mean_R2']:+.3f}  平均NRMSE={m['mean_NRMSE']:.3f}")
        if res["at_bounds"]:
            print(f"  ⚠ 境界張り付き: {', '.join(res['at_bounds'])}")

    with open(cache_path(species), "wb") as f:
        pickle.dump({"est_results": est_results, "fixed": cfg["fixed"],
                     "n_starts_c": N_STARTS_C, "n_seeds_c": N_SEEDS_C,
                     "reg_lambda_c": REG_LAMBDA_C, "model_version": "v1_8var_S_fixed"}, f)
    print(f"→ キャッシュ保存: {cache_path(species)}")

    print("\n" + "-" * 64)
    for rname, sl in regimes:
        res = est_results[rname]
        m = res["metrics"]
        ov = m["overall"]
        ap = res["params_abs"]
        pf = res["params_free"]
        print(f"■ {species} {rname}")
        print(f"  cost      = {res['cost']:.4f}")
        print(f"  平均NRMSE = {ov['mean_NRMSE']:.4f}   平均R² = {ov['mean_R2']:+.4f}")
        print("  魚種別NRMSE: " + "  ".join(
            f"{labels[i]}={m[KEYS[i]]['NRMSE']:.4f}" for i in range(4)))
        print("  魚種別R²   : " + "  ".join(
            f"{labels[i]}={m[KEYS[i]]['R2']:+.3f}" for i in range(4)))
        print(f"  自由8変数 : r_y1={pf[0]:.4f} r_y2={pf[1]:.4f} "
              f"L11={pf[2]:.4f} L12={pf[3]:.4f} L21={pf[4]:.4f} L22={pf[5]:.4f} "
              f"theta1={pf[6]:.4f} theta2={pf[7]:.4f}")
        print(f"  物理c/d(abs): c1={ap['c1']:.4f} d1={ap['d1']:.4f} "
              f"c2={ap['c2']:.4f} d2={ap['d2']:.4f}  (c1+d1={ap['c1']+ap['d1']:.4f}, "
              f"c2+d2={ap['c2']+ap['d2']:.4f} ← S1,S2固定と一致するはず)")
        print(f"  固定 r_x,S : r_x1={res['fixed']['r_x1']:.3f} r_x2={res['fixed']['r_x2']:.3f} "
              f"S1={res['fixed']['S1']:.3f} S2={res['fixed']['S2']:.3f}")
        if res["at_bounds"]:
            print(f"  ⚠ 境界張り付き: {', '.join(res['at_bounds'])}")
        print("-" * 64)

    # fit図（既存のoutputs/<種>版/制約_旧8var_S固定/ に「復元」印つきで追加保存）
    dest_dir = os.path.join(_out_dir, f"{species}版", "制約_旧8var_S固定")
    os.makedirs(dest_dir, exist_ok=True)
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
            ax.set_title(f"{rname}: {labels[row]}")
            ax.set_ylabel("資源量（千トン）")
            ax.grid(True, ls="--", alpha=0.5)
            ax.legend(fontsize=8)
    fig.suptitle(f"【制約推定8変数, S固定・復元版】{species}+ウルメイワシ / ブリ+サワラ — "
                 "capacity_ry_constrained（積分結果の滑らか軌道）", fontsize=13, y=1.003)
    plt.tight_layout()
    out = os.path.join(dest_dir,
        f"fit_制約8var_復元_{species}_ウルメイワシ_ブリ_サワラ_capacity_ry_constrained.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"→ fit図保存: {out}")

    return est_results


def main():
    for species in ["マアジ", "マイワシ"]:
        run_species(species)


if __name__ == "__main__":
    main()
