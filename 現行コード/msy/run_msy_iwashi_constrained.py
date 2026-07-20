"""
マイワシ版（旧種構成: マイワシ+ウルメイワシ / ブリ+サワラ）の
制約推定10変数版（v2, r_x1,r_x2のみ固定）で、run_msy.py --constrained と
同じ Step2〜4（戦略的MSY・戦術的MSY・図6枚）を実行する。

Step1（推定）は既存キャッシュ estimates_マイワシ_capacity_ry_constrained.pkl
（_run_iwashi_rx_constrained.py が生成済み）をそのまま再利用し、再推定しない。

data_loader.ASSIGN["x1"] をこのプロセス内でのみ「マイワシ」に一時的に上書きしてから
run_msy をimportすることで、run_msy.SPECIES_TAG 等が自動的に「マイワシ_ウルメイワシ_
ブリ_サワラ」になり、run_msy.py 本体のプロット関数をそのまま無改修で再利用できる
（ファイル名・タイトルは動的化済み, 2026-07-18改修）。

estimate_cache.py のマアジ用キャッシュ（estimates_capacity_ry_constrained_v2_...pkl）
には一切触れない（署名にデータ種構成の区別が無く誤って読み書きすると衝突するため、
本スクリプトは estimate_cache.load_estimates_constrained/save_estimates_constrained を
呼ばない）。

使い方: cd 現行コード/msy && python3 run_msy_iwashi_constrained.py
"""
import os
import pickle
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_here)
if _here not in sys.path:
    sys.path.insert(0, _here)
if _parent not in sys.path:
    sys.path.append(_parent)

import numpy as np

# --- data_loader をマイワシ版へプロセス内限定で上書き（run_msy import前に行う） ---
import data_loader
data_loader.ASSIGN["x1"] = "マイワシ"
data_loader.SPECIES_LABELS = ["マイワシ (x1)", "ウルメイワシ (x2)", "ブリ (y1)", "サワラ (y2)"]

# run_msy は import 時に `from data_loader import SPECIES_LABELS` するため、
# 上の上書き後に import することで SPECIES_TAG 等が自動的にマイワシ版になる。
import run_msy  # noqa: E402

_IWASHI_CACHE = os.path.join(_here, "estimates_マイワシ_capacity_ry_constrained.pkl")


def main():
    model_str = "capacity_ry_constrained"
    print(run_msy._sep())
    print(f"MSY 計算（マイワシ版）  モデル: {model_str}  推定: 制約推定(10変数, r_xのみ固定)")
    print(f"持続性制約: scope={run_msy.SUSTAIN_CFG['scope']}  mode={run_msy.SUSTAIN_CFG['mode']}  "
          f"tol={run_msy.SUSTAIN_CFG['tol']}")
    print(f"SPECIES_TAG = {run_msy.SPECIES_TAG}")
    print(run_msy._sep())

    # ------------------------------------------------------------------
    # データ読み込みとレジーム分割（マイワシ版 data_loader.ASSIGN で）
    # ------------------------------------------------------------------
    df = run_msy.load_clean_dataframe()
    series = run_msy.get_series(df)
    nlm_mask, lm_mask = run_msy.regime_masks(series)
    sl_nlm = run_msy.slice_series(series, nlm_mask)
    sl_lm = run_msy.slice_series(series, lm_mask)
    regimes = [
        ("NLM", sl_nlm, run_msy.NLM_YEARS),
        ("LM", sl_lm, run_msy.LM_YEARS),
    ]

    # ------------------------------------------------------------------
    # Step 1: 既存キャッシュを読み込む（再推定しない）
    # ------------------------------------------------------------------
    print(f"\n[Step 1] 制約付き ODE 推定結果（マイワシ版, 既存キャッシュを再利用・再推定なし）")
    if not os.path.exists(_IWASHI_CACHE):
        print(f"[ERROR] キャッシュが見つかりません: {_IWASHI_CACHE}")
        print("        先に _run_iwashi_rx_constrained.py を実行してください。")
        sys.exit(1)
    with open(_IWASHI_CACHE, "rb") as f:
        cache = pickle.load(f)
    est_results = cache["est_results"]
    for rname, _, _ in regimes:
        res = est_results[rname]
        m = res["metrics"]["overall"]
        ap = res["params_abs"]
        bnd = f"  ⚠境界: {', '.join(res['at_bounds'])}" if res["at_bounds"] else ""
        print(f"    {rname}: 平均NRMSE={m['mean_NRMSE']:.3f}  "
              f"c1={ap['c1']:.3f} d1={ap['d1']:.3f} "
              f"c2={ap['c2']:.3f} d2={ap['d2']:.3f}{bnd}")

    # ------------------------------------------------------------------
    # Step 2: 戦略的 MSY（レジーム全期間） — run_msy.main() の Step2 と同一ロジック
    # ------------------------------------------------------------------
    print(f"\n[Step 2] 戦略的 MSY（レジーム全期間）")
    strategic = {}
    sweep_res_list = []
    grid_res_list = []
    sens_res_list = []
    sustain_margins_dict = {}

    for rname, sl, _ in regimes:
        est = est_results[rname]
        pn = est["params_norm"]
        mn = est["means"]
        T = run_msy.get_regime_T(sl)
        X0n = run_msy.get_regime_X0_norm(sl, mn)

        print(f"\n  ── {rname}  T={T:.1f} 年 ──")

        print(f"  1. 共通漁獲率スイープ ({run_msy.N_COMMON} 点, 制約付き) ...", flush=True)
        sweep = run_msy.scan_common_rate(pn, mn, T, X0n, sustain=run_msy.SUSTAIN_CFG)
        sweep_res_list.append(sweep)
        print(f"     最大収量（制約）  : {sweep['best_yield_constrained']:.3f} 千トン/年  "
              f"at f_common={sweep['best_f_constrained']:.3f}")

        print(f"  2. グリッド探索 ({run_msy.N_GRID_STRATEGIC}^4={run_msy.N_GRID_STRATEGIC**4} "
              f"評価, 制約付き) ...", flush=True)
        grid = run_msy.grid_search_msy(pn, mn, T, X0n,
                                        n_grid=run_msy.N_GRID_STRATEGIC, sustain=run_msy.SUSTAIN_CFG)
        grid_res_list.append(grid)

        margins_con = None
        f_con = grid["f_star_constrained"]
        if np.isfinite(grid["msy_constrained"]):
            _res_con = run_msy.average_yield(f_con, pn, mn, T, X0n)
            if _res_con["success"]:
                sc = run_msy.check_sustainability(_res_con["traj_abs"], **run_msy.SUSTAIN_CFG)
                margins_con = sc["margins"]
        sustain_margins_dict[rname] = margins_con

        run_msy.print_strategic_result(rname, model_str, T, grid,
                                        sustain_cfg=run_msy.SUSTAIN_CFG,
                                        sustain_margins=margins_con)

        print(f"  3. 種別感度スイープ ({run_msy.N_SENS} 点 × 4 種, 基準: 制約 f*) ...", flush=True)
        f_base = f_con if np.isfinite(grid["msy_constrained"]) else grid["f_star"]
        sens = run_msy.species_sensitivity(f_base, pn, mn, T, X0n)
        sens_res_list.append(sens)
        print(f"     完了")

        strategic[rname] = {
            "sweep": sweep, "grid": grid, "sens": sens,
            "T": T, "X0_norm": X0n, "means": mn, "params_norm": pn,
        }

    # ------------------------------------------------------------------
    # Step 3: 戦術的 MSY（1 年ごと）
    # ------------------------------------------------------------------
    print(f"\n[Step 3] 戦術的 MSY（T=1 年ごと, {run_msy.N_GRID}^4={run_msy.N_GRID**4} 評価 × 年数）")
    tactical = {}
    for rname, sl, _ in regimes:
        est = est_results[rname]
        n_y = len(sl["years"])
        print(f"  {rname} ({n_y} 年 × {run_msy.N_GRID**4} 評価 = {n_y * run_msy.N_GRID**4} 評価) ...",
              flush=True)
        tac = run_msy.tactical_msy_per_year(sl, est["params_norm"], est["means"])
        tactical[rname] = tac
        run_msy.print_tactical_summary(rname, model_str, tac)

    # ------------------------------------------------------------------
    # Step 4: PNG 出力（SPECIES_TAG がマイワシ版に自動追従済みなのでファイル名も正しい）
    # ------------------------------------------------------------------
    print(f"\n[Step 4] PNG 出力")
    run_msy.plot_fit(est_results, regimes, model_str, constrained=True)
    run_msy.plot_sensitivity(sens_res_list[0], sens_res_list[1], model_str)
    run_msy.plot_tactical(tactical["NLM"], tactical["LM"], model_str)
    run_msy.plot_grid_scatter_constrained(grid_res_list[0], grid_res_list[1], model_str)
    run_msy.plot_common_sweep_constrained(sweep_res_list, model_str)
    run_msy.plot_nlm_lm_comparison_constrained(grid_res_list[0], grid_res_list[1], model_str)

    # ------------------------------------------------------------------
    # Step 5: 全体サマリ
    # ------------------------------------------------------------------
    print("\n" + run_msy._sep())
    print("[全体サマリ]（マイワシ版）")
    print(run_msy._sep("-"))
    print(f"{'レジーム':>6}  {'制約MSY':>10}  "
          f"持続可能点/評価点  制約 f*（x1, x2, y1, y2）")
    for rname, data in strategic.items():
        g = data["grid"]
        msy_con = g["msy_constrained"]
        f_con = g["f_star_constrained"]
        n_feas = g["n_feasible"]
        n_eval = g["n_evaluated"]
        f_con_str = (f"{f_con[0]:.3f}  {f_con[1]:.3f}  {f_con[2]:.3f}  {f_con[3]:.3f}"
                     if np.isfinite(msy_con) else "N/A")
        print(f"  {rname:>4}  {msy_con:>10.3f}  "
              f"{n_feas:>7d} / {n_eval:<6d}  {f_con_str}")
    print(run_msy._sep())
    print("MSY 計算完了（マイワシ版）。")

    # ------------------------------------------------------------------
    # 出力先を マイワシ版/制約_rxのみ固定10var/ に整理
    # ------------------------------------------------------------------
    dest_dir = os.path.join(_here, "outputs", "マイワシ版", "制約_rxのみ固定10var")
    os.makedirs(dest_dir, exist_ok=True)
    moved = []
    for fn in os.listdir(run_msy._out_dir):
        if fn.startswith(("fit_制約_" + run_msy.SPECIES_TAG, "msy_")) and run_msy.SPECIES_TAG in fn \
                and fn.endswith(f"{model_str}.png"):
            src = os.path.join(run_msy._out_dir, fn)
            dst = os.path.join(dest_dir, fn)
            os.replace(src, dst)
            moved.append(dst)
    print(f"\n[整理] {len(moved)} 枚を {dest_dir} へ移動しました。")
    for m in moved:
        print(f"  → {m}")


if __name__ == "__main__":
    main()
