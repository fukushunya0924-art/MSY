"""刻み0.05で求めた f* に対して、Mode 3（trajectory_floor）長期検証をやり直す。

なぜ専用スクリプトか:
  ドライバのセクション[5]は同[1]が出した f* を使うため、これだけを回すには
  ドライバ全体（セクション[3]の 698,449 評価＝約35分）を通す必要があった。
  だが f* はすでに確定している（刻み0.05・上限0.95・legacy制約）ので、
  evaluate_trajectory_floor を直接呼べば1構成1秒未満で済む。

f* の出どころ（2つの独立な実装で一致を確認済み）:
  - sustainability.grid_search_general 経由 → outputs/sustainability_sensitivity_*_step0.05.csv
    の upper_bound_legacy_constrained・漁獲率上限0.95 の行
  - msy_core.grid_search_msy 経由（n_grid=20）→ _regen_grid_figs_step05.py のログ

実行: cd 現行コード/msy && python3 -u _recheck_mode3_step05.py
"""
import copy
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_here)
if _here not in sys.path:
    sys.path.insert(0, _here)
if _parent not in sys.path:
    sys.path.append(_parent)

import numpy as np

import sustainability as sus
import run_sustainability_diagnostics as rsd

F_STAR = {
    ("マアジ", "NLM"): [0.40, 0.10, 0.90, 0.95],
    ("マアジ", "LM"): [0.45, 0.05, 0.55, 0.95],
    ("マイワシ", "NLM"): [0.10, 0.15, 0.85, 0.95],
    ("マイワシ", "LM"): [0.95, 0.15, 0.75, 0.50],
}


def _setup(kind):
    """種構成ごとに (推定結果, レジーム別スライス, X0正規化関数) を返す。"""
    if kind == "マアジ":
        import estimate_cache
        from data_loader import (load_clean_dataframe, get_series, regime_masks,
                                 slice_series, get_regime_X0_norm)
        est = estimate_cache.load_estimates()
        if est is None:
            raise RuntimeError("estimates_capacity_ry.pkl が無い。先に run_msy.py を実行すること。")
    else:
        from data_loader_iwashi import (load_clean_dataframe, get_series, regime_masks,
                                         slice_series, get_regime_X0_norm)
        from run_sustainability_diagnostics_iwashi import reconstruct_iwashi_estimates
        est = None

    series = get_series(load_clean_dataframe())
    nlm, lm = regime_masks(series)
    regimes = {"NLM": slice_series(series, nlm), "LM": slice_series(series, lm)}
    if est is None:
        est = reconstruct_iwashi_estimates(list(regimes.items()))
    return est, regimes, get_regime_X0_norm


def main():
    for kind in ("マアジ", "マイワシ"):
        est, regimes, get_X0n = _setup(kind)
        for reg in ("NLM", "LM"):
            pn, mn = est[reg]["params_norm"], est[reg]["means"]
            X0n = get_X0n(regimes[reg], mn)
            f = np.array(F_STAR[(kind, reg)], dtype=float)

            print(f"\n== {kind} {reg}  f*={list(f)} ==")
            for label, tv in (("長期100+100", rsd.TRAJ_LONG), ("短期50+50", rsd.TRAJ_SHORT)):
                cfg = copy.deepcopy(sus.DEFAULT_SUSTAINABILITY)
                cfg["mode"] = "trajectory_floor"
                cfg["trajectory_validation"] = tv
                tf = sus.evaluate_trajectory_floor(pn, mn, X0n, f, cfg)
                print(f"  {label}: feasible={tf['feasible']}  reason={tf.get('reason')}  "
                      f"最初の違反種={tf.get('first_violating_species')}  "
                      f"負値あり={tf.get('any_negative')}")
                if tf["solver_success"]:
                    print(f"      最小資源量={np.round(tf['min_biomass'], 4)}  "
                          f"基準={tf['reference_kind']}")


if __name__ == "__main__":
    main()
