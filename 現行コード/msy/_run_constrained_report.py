"""制約推定(10変数)を1レジームずつ走らせ、best-of をレジーム別pklに蓄積する補助スクリプト。

サンドボックスが呼び出し間でプロセスを凍結するため、run_msy.py --constrained を
一度に完走できない。そこでレジーム単位・小バジェットで複数回呼び、各回のbest-of
（cost最小）を _partial_<regime>.pkl に蓄積する。最後に mode=report で両レジームを
まとめてレポートし、run_msy が使う正規キャッシュ（estimate_cache.CACHE_FILE_CONSTRAINED,
モデル版数タグ付きファイル名）にも保存する。

使い方:
  python3 _run_constrained_report.py fit NLM <n_starts> <n_seeds> <seed0>
  python3 _run_constrained_report.py fit LM  <n_starts> <n_seeds> <seed0>
  python3 _run_constrained_report.py report
"""
import os, sys, pickle, time
_here = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_here)
sys.path.insert(0, _parent)
sys.path.insert(0, _here)  # msy/ を最優先（msy/data_loader.py が regime_masks 等を持つ）

import numpy as np
from data_loader import (load_clean_dataframe, get_series, regime_masks,
                         slice_series, SPECIES_LABELS, KEYS)
import model_constrained as mc
from estimate_cache import (REG_LAMBDA_C, N_STARTS_C, N_SEEDS_C,
                            save_estimates_constrained)

PARTIAL = lambda r: os.path.join(_here, f"_partial_{r}.pkl")


def _load_slices():
    df = load_clean_dataframe(); s = get_series(df)
    nlm, lm = regime_masks(s)
    return {"NLM": slice_series(s, nlm), "LM": slice_series(s, lm)}


def fit(regime, n_starts, n_seeds, seed0):
    sl = _load_slices()[regime]
    t = time.time()
    res = mc.estimate_constrained_robust(
        sl, n_starts=n_starts, reg_lambda=REG_LAMBDA_C[regime],
        n_seeds=n_seeds, seed0=seed0)
    dt = time.time() - t
    prev = None
    if os.path.exists(PARTIAL(regime)):
        with open(PARTIAL(regime), "rb") as f:
            prev = pickle.load(f)
    keep = res
    kept_from = "new"
    if prev is not None and prev["cost"] <= res["cost"]:
        keep = prev; kept_from = "prev"
    with open(PARTIAL(regime), "wb") as f:
        pickle.dump(keep, f)
    m = keep["metrics"]["overall"]
    print(f"[{regime}] this_run cost={res['cost']:.4f} NRMSE={res['metrics']['overall']['mean_NRMSE']:.4f} "
          f"({dt:.0f}s, {n_starts}x{n_seeds} seed0={seed0})")
    print(f"[{regime}] KEPT({kept_from}) cost={keep['cost']:.4f} mean_NRMSE={m['mean_NRMSE']:.4f}")


def _report_regime(regime, res):
    m = res["metrics"]
    ov = m["overall"]
    ap = res["params_abs"]
    pf = res["params_free"]
    print("=" * 64)
    print(f"■ {regime}")
    print(f"  cost        = {res['cost']:.4f}")
    print(f"  平均NRMSE   = {ov['mean_NRMSE']:.4f}   平均R² = {ov['mean_R2']:+.4f}")
    print(f"  魚種別NRMSE : "
          + "  ".join(f"{SPECIES_LABELS[i]}={m[KEYS[i]]['NRMSE']:.4f}" for i in range(4)))
    print(f"  魚種別R²    : "
          + "  ".join(f"{SPECIES_LABELS[i]}={m[KEYS[i]]['R2']:+.3f}" for i in range(4)))
    print(f"  推定(自由10): r_y1={pf[0]:.4f} r_y2={pf[1]:.4f}  "
          f"L11={pf[2]:.4f} L12={pf[3]:.4f} L21={pf[4]:.4f} L22={pf[5]:.4f}")
    print(f"               C1={pf[6]:.4f} D1={pf[7]:.4f} C2={pf[8]:.4f} D2={pf[9]:.4f}")
    print(f"  物理c/d(abs): c1={ap['c1']:.4f} d1={ap['d1']:.4f}  "
          f"c2={ap['c2']:.4f} d2={ap['d2']:.4f}")
    print(f"  固定 r_x    : r_x1={res['fixed']['r_x1']:.3f} r_x2={res['fixed']['r_x2']:.3f}")
    if res["at_bounds"]:
        print(f"  ⚠ 境界張り付き: {', '.join(res['at_bounds'])}")


def report():
    est = {}
    for r in ["NLM", "LM"]:
        with open(PARTIAL(r), "rb") as f:
            est[r] = pickle.load(f)
    for r in ["NLM", "LM"]:
        _report_regime(r, est[r])
    print("=" * 64)
    path = save_estimates_constrained(est)
    print(f"→ 正規キャッシュ保存: {os.path.basename(path)}")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "fit":
        fit(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
    elif mode == "report":
        report()
