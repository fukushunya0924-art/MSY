"""自由推定12変数のグリッド系 PNG を、漁獲率の刻み0.05で作り直す。

対象は2構成 × 2図：
  - マアジ版（現行, estimates_capacity_ry.pkl）
  - マイワシ版（Phase 11 以前, research_log Phase 8 の値から再構成）
  × msy_グリッド散布_持続可能制約 / msy_NLM_LM比較_持続可能制約

なぜ n_grid=20 か:
  run_msy.py の戦略グリッドは linspace(F_MIN, F_MAX, N_GRID_STRATEGIC) = linspace(0, 0.95, 8)
  で刻み 0.1357。上限は 0.95 固定なので「上限を上げると刻みが粗くなる」問題は起きないが、
  刻み自体が粗く、上限感度側（Phase 15 で刻み0.05に統一）と解像度が揃っていなかった。
  0.95/19 = 0.05 ちょうどなので、n_grid=20 にするだけで刻み0.05の格子になる（msy_core 無改変）。

計算量: 20^4 = 160,000 評価/レジーム。実測 約2.9ms/評価 → 1レジーム約8分。

実行: cd 現行コード/msy && python3 -u _regen_grid_figs_step05.py
（スレッド過剰割り当てを避けるため OMP_NUM_THREADS=1 等を付けて起動すること）
"""
import os
import sys
import time

_here = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_here)
if _here not in sys.path:
    sys.path.insert(0, _here)
if _parent not in sys.path:
    sys.path.append(_parent)

import numpy as np
import matplotlib
matplotlib.use("Agg")

import msy_core
import run_msy
from msy_core import grid_search_msy

# F_MAX=0.95 に対し 0.95/19 = 0.05 ちょうど。上限か点数を変えるならここで弾かれる。
N_GRID_STEP05 = 20
_STEP = (msy_core.F_MAX - msy_core.F_MIN) / (N_GRID_STEP05 - 1)
assert abs(_STEP - 0.05) < 1e-12, f"刻みが0.05にならない: {_STEP}"


def _compute(est_results, regimes, get_T, get_X0n, label):
    """両レジームの制約グリッド探索を n_grid=20 で実行する。"""
    grids = []
    for rname, sl in regimes:
        est = est_results[rname]
        pn, mn = est["params_norm"], est["means"]
        T = get_T(sl)
        X0n = get_X0n(sl, mn)
        t0 = time.time()
        print(f"  [{label}/{rname}] グリッド探索 {N_GRID_STEP05}^4="
              f"{N_GRID_STEP05 ** 4} 評価 ...", flush=True)
        g = grid_search_msy(pn, mn, T, X0n,
                            n_grid=N_GRID_STEP05, sustain=run_msy.SUSTAIN_CFG)
        print(f"  [{label}/{rname}] {time.time() - t0:.0f}秒  "
              f"制約MSY={g['msy_constrained']:.2f}  "
              f"f*={np.round(g['f_star_constrained'], 3)}  "
              f"持続可能点={g['n_feasible']}/{g['n_evaluated']}", flush=True)
        grids.append(g)
    return grids


def _draw(grids, out_dirs, species_tag, species_labels, model_str):
    """run_msy.py の描画関数を、出力先と種名だけ差し替えて呼ぶ。"""
    saved_dir = run_msy._out_dir
    saved_tag = run_msy.SPECIES_TAG
    saved_lab = run_msy.SPECIES_LABELS
    try:
        run_msy.SPECIES_TAG = species_tag
        run_msy.SPECIES_LABELS = species_labels
        for d in out_dirs:
            os.makedirs(d, exist_ok=True)
            run_msy._out_dir = d
            run_msy.plot_grid_scatter_constrained(grids[0], grids[1], model_str)
            run_msy.plot_nlm_lm_comparison_constrained(grids[0], grids[1], model_str)
    finally:
        run_msy._out_dir = saved_dir
        run_msy.SPECIES_TAG = saved_tag
        run_msy.SPECIES_LABELS = saved_lab


def run_aji():
    from data_loader import (load_clean_dataframe, get_series, regime_masks,
                             slice_series, get_regime_T, get_regime_X0_norm,
                             SPECIES_LABELS)
    import estimate_cache

    est_results = estimate_cache.load_estimates()
    if est_results is None:
        raise RuntimeError("estimates_capacity_ry.pkl が無い。先に run_msy.py を実行すること。")

    series = get_series(load_clean_dataframe())
    nlm, lm = regime_masks(series)
    regimes = [("NLM", slice_series(series, nlm)), ("LM", slice_series(series, lm))]

    grids = _compute(est_results, regimes, get_regime_T, get_regime_X0_norm, "マアジ")
    tag = "_".join(l.split(" (")[0] for l in SPECIES_LABELS)
    _draw(grids, [os.path.join(_here, "outputs"),
                  os.path.join(_here, "outputs", "マアジ版", "自由推定")],
          tag, SPECIES_LABELS, "capacity_ry")
    return grids


def run_iwashi():
    from data_loader_iwashi import (load_clean_dataframe, get_series, regime_masks,
                                     slice_series, get_regime_T, get_regime_X0_norm,
                                     SPECIES_LABELS)
    from run_sustainability_diagnostics_iwashi import reconstruct_iwashi_estimates

    series = get_series(load_clean_dataframe())
    nlm, lm = regime_masks(series)
    regimes = [("NLM", slice_series(series, nlm)), ("LM", slice_series(series, lm))]
    est_results = reconstruct_iwashi_estimates(regimes)

    grids = _compute(est_results, regimes, get_regime_T, get_regime_X0_norm, "マイワシ")
    tag = "_".join(l.split(" (")[0] for l in SPECIES_LABELS)
    _draw(grids, [os.path.join(_here, "outputs", "マイワシ版", "自由推定")],
          tag, SPECIES_LABELS, "capacity_ry")
    return grids


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    t0 = time.time()
    if which in ("aji", "both"):
        run_aji()
    if which in ("iwashi", "both"):
        run_iwashi()
    print(f"\n完了: {time.time() - t0:.0f}秒")
