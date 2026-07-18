"""
持続性制約 診断ドライバ（マイワシ版, Phase 11以前の種構成）。

run_sustainability_diagnostics.py（マアジ版）と全く同じ診断ロジック（section_legacy /
section_equilibrium_lrp / section_upper_bound / section_trajectory_floor）を再利用し、
種構成だけを Phase 11 以前の「マイワシ(x1)+ウルメイワシ(x2)+ブリ(y1)+サワラ(y2)」に
差し替えて実行する。

自由推定12変数（capacity_ry）のマイワシ版フィット結果は、`docs/research_log.md` の
Phase 8（2026-07-10〜11時点、`estimates_capacity_ry.pkl` がまだマイワシ版だった頃に
記録された物理パラメータ値, NLM平均NRMSE=0.146 / LM平均NRMSE=0.079）に残っている。
本スクリプトは **再推定を行わず**、この記録値と実データから決定論的に計算できる
`means`（各種の全期間平均資源量。フィット結果ではなくデータそのものから求まる値）から
`model._to_absolute()` の逆変換で `params_norm` を復元する。round-trip 検証
（復元した params_norm を再度 _to_absolute にかけて Phase 8 記載の絶対値と一致するか）
を起動時に必ず行い、不一致なら即エラーで停止する。

sustainability.py・msy_core.py・run_sustainability_diagnostics.py は一切変更しない。
"""
import os
import sys
import time

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_here)
if _here not in sys.path:
    sys.path.insert(0, _here)
if _parent not in sys.path:
    sys.path.append(_parent)

import msy_core                                     # noqa: E402 (np.trapz シムを道連れで適用)
import sustainability as sus                         # noqa: E402
from model import _to_absolute                       # noqa: E402
from data_loader_iwashi import (                     # noqa: E402  マイワシ版データローダ
    load_clean_dataframe, get_series, regime_masks, slice_series,
    get_regime_T, get_regime_X0_norm,
)
# 診断本体は マアジ版ドライバから関数を再利用（ロジック重複を避ける）
from run_sustainability_diagnostics import (         # noqa: E402
    section_legacy, section_equilibrium_lrp, section_upper_bound,
    section_trajectory_floor, fmt_vec, ffmt, _sep,
)

_out_dir = os.path.join(_here, "outputs")
os.makedirs(_out_dir, exist_ok=True)

# -----------------------------------------------------------------------
# docs/research_log.md Phase 8（2026-07-10〜11）記載の自由推定12変数・絶対パラメータ値
# （マイワシ(x1)+ウルメイワシ(x2)+ブリ(y1)+サワラ(y2), capacity_ry, estimates_capacity_ry.pkl
#  当時の内容）。NLM平均NRMSE=0.146(R²=+0.146欄参照), LM平均NRMSE=0.079（research_log記載）。
# -----------------------------------------------------------------------
_ABS_PARAMS_PHASE8 = {
    "NLM": dict(r_x1=0.6236, r_x2=0.4718, r_y1=0.0318, r_y2=0.0110,
                l11=0.000122, l12=0.01133, l21=0.000774, l22=0.03627,
                c1=0.2458, d1=2.5211, c2=0.00161, d2=0.05032),
    "LM": dict(r_x1=3.0035, r_x2=0.6514, r_y1=0.0941, r_y2=0.0691,
               l11=0.00569, l12=0.07893, l21=0.000377, l22=0.06693,
               c1=0.00832, d1=2.7287, c2=0.0000481, d2=0.02707),
}
_DOCUMENTED_METRICS = {
    "NLM": dict(mean_NRMSE=0.146, note="マイワシR²=+0.99, ブリ+0.72, サワラ+0.93, ウルメ−2.63"),
    "LM":  dict(mean_NRMSE=0.079, note="マイワシR²=+0.95, ウルメ+0.59, ブリ−1.13, サワラ+0.64"),
}
_ROUND_TRIP_RTOL = 1e-3


def _invert_to_absolute(ap, means):
    """model._to_absolute() の逆変換。ap=物理パラメータdict, means=[mx1,mx2,my1,my2]。"""
    mx1, mx2, my1, my2 = means
    L11 = ap["l11"] * my1
    L12 = ap["l12"] * my2
    L21 = ap["l21"] * my1
    L22 = ap["l22"] * my2
    C1 = ap["c1"] * mx1 / my1
    D1 = ap["d1"] * mx2 / my1
    C2 = ap["c2"] * mx1 / my2
    D2 = ap["d2"] * mx2 / my2
    return np.array([ap["r_x1"], ap["r_x2"], ap["r_y1"], ap["r_y2"],
                      L11, L12, L21, L22, C1, D1, C2, D2])


def reconstruct_iwashi_estimates(regimes):
    """research_log.md Phase 8 の絶対パラメータ値から params_norm を再構成する（再推定なし）。"""
    est_results = {}
    for rname, sl in regimes:
        means = np.array([np.mean(sl[k]) for k in ("x1", "x2", "y1", "y2")])
        ap = _ABS_PARAMS_PHASE8[rname]
        pn = _invert_to_absolute(ap, means)

        # round-trip 検証: 復元した params_norm を再度 _to_absolute にかけ、
        # research_log 記載の絶対値と一致するか確認する。
        back = _to_absolute(pn, means)
        mismatches = [k for k in ap if not np.isclose(back[k], ap[k], rtol=_ROUND_TRIP_RTOL)]
        if mismatches:
            raise RuntimeError(
                f"[{rname}] round-trip 不一致: {mismatches}  "
                f"back={ {k: back[k] for k in mismatches} }  ap={ {k: ap[k] for k in mismatches} }"
            )
        print(f"  [{rname}] round-trip 検証: 一致（rtol={_ROUND_TRIP_RTOL}）  means={fmt_vec(means, '.2f')}")

        est_results[rname] = {
            "params_norm": pn,
            "means": means,
            "metrics": {"overall": {"mean_NRMSE": _DOCUMENTED_METRICS[rname]["mean_NRMSE"],
                                     "mean_R2": float("nan")}},
            "names": ["r_x1", "r_x2", "r_y1", "r_y2", "L11", "L12", "L21", "L22",
                      "C1", "D1", "C2", "D2"],
            "at_bounds": [],
            "_source": "docs/research_log.md Phase 8（再推定なし、絶対値からの逆変換）",
            "_documented_note": _DOCUMENTED_METRICS[rname]["note"],
        }
    return est_results


def main():
    print(_sep())
    print("持続性制約 診断ドライバ（マイワシ版, Phase11以前の種構成）")
    print(_sep())

    df = load_clean_dataframe()
    series = get_series(df)
    nlm_mask, lm_mask = regime_masks(series)
    regimes = [
        ("NLM", slice_series(series, nlm_mask)),
        ("LM", slice_series(series, lm_mask)),
    ]

    print("\n[Step 1] 自由推定12変数（capacity_ry, マイワシ版, research_log.md Phase 8 記載値から再構成・再推定なし）")
    est_results = reconstruct_iwashi_estimates(regimes)

    all_csv_rows = {}
    section1_results = {}
    section5_results = {}

    for rname, sl in regimes:
        t_regime0 = time.time()
        print("\n" + _sep())
        print(f"レジーム: {rname}（マイワシ版）")
        print(_sep())

        est = est_results[rname]
        pn, mn = est["params_norm"], est["means"]
        T = get_regime_T(sl)
        X0n = get_regime_X0_norm(sl, mn)
        print(f"推定メタ情報: 平均NRMSE={est['metrics']['overall']['mean_NRMSE']:.3f}  "
              f"（{est['_documented_note']}）  means={fmt_vec(mn, '.2f')}")
        print(f"出典: {est['_source']}")

        csv_rows = []
        r1 = section_legacy(rname, pn, mn, T, X0n, csv_rows)
        r2 = section_equilibrium_lrp(rname, pn, mn, X0n, csv_rows)
        r3 = section_upper_bound(rname, pn, mn, T, X0n, csv_rows)
        f_chosen = r1["cm"]["f_opt"] if np.all(np.isfinite(r1["cm"]["f_opt"])) else np.full(4, np.nan)
        r5 = section_trajectory_floor(rname, pn, mn, X0n, f_chosen, csv_rows)

        section1_results[rname] = r1
        section5_results[rname] = r5
        all_csv_rows[rname] = csv_rows

        wall = time.time() - t_regime0
        print(f"\n[wall-clock] レジーム {rname} 合計: {wall:.1f} 秒")

    print("\n" + _sep())
    print("[CSV出力]（マイワシ版）")
    csv_paths = {}
    for rname in ["NLM", "LM"]:
        path = os.path.join(_out_dir, f"sustainability_sensitivity_マイワシ_{rname}.csv")
        sus.sensitivity_to_csv(all_csv_rows[rname], path)
        csv_paths[rname] = path
        print(f"  CSV[{rname}] = {path}  ({len(all_csv_rows[rname])} 行)")

    print("\n" + _sep())
    print("[マアジ版との比較サマリ]")
    print(_sep("-"))
    for rname in ["NLM", "LM"]:
        r1 = section1_results[rname]
        r5 = section5_results[rname]
        print(f"  {rname}: legacy制約 f*={fmt_vec(r1['cm']['f_opt'])}  "
              f"収量={ffmt(r1['cm']['yield'])}  分類={r1['cm']['classification']}  "
              f"解釈={r1['cm']['msy_interpretation']}")
        print(f"       長期(100+100yr)軌道: feasible={r5['long']['feasible']}  "
              f"reason={r5['long'].get('reason')}")

    print(_sep())
    print("診断ドライバ完了（マイワシ版）。")
    print(_sep())


if __name__ == "__main__":
    main()
