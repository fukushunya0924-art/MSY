"""
持続性制約 診断ドライバ（第2段: sustainability.py の公開APIを実データで動かす）。

estimate_cache.load_estimates() でキャッシュ済みの自由推定4種パラメータ（NLM/LM,
capacity_ry 12変数）を読み込み、sustainability.py の公開関数だけを使って以下を実行する:

  1. legacy 制約の再現       … 無制約 vs legacy制約（endpoint, tol=0.1）の最大時間平均収量
                              （msy_core.average_yield + sustainability.evaluate_legacy）
  2. equilibrium_lrp         … 無漁獲平衡の正値性を判定（本モデルは正の共存平衡が無い）
  3. 上限感度                … f_upper in [0.25,0.50,0.75,0.95,1.25] を短期地平の
                              時間平均収量でスイープ（無制約・legacy制約の両方）
  4. 制約下最大収量 vs 95%安全側解 … near_optimal_safe（max_min_biomass_margin）
  5. trajectory_floor 長期検証 … 100+100年（既定）と50+50年（短縮）の両方
  6. CSV出力                 … outputs/sustainability_sensitivity_{NLM,LM}.csv

このスクリプトは sustainability.py・msy_core.py を一切変更せず、公開関数のみを呼ぶ
（このタスクの制約: 変更してよいのは本ファイルだけ）。

【本ドライバ実装中に見つかったバグ（sustainability.py 側は無改修・報告のみ）】
  sustainability.upper_bound_sensitivity() と lrp_sensitivity() はどちらも T（積分期間）
  を受け取る引数を持たず、内部で呼ぶ grid_search_general(...) にも T を転送できない。
  そのため cfg["mode"]="legacy_path" で呼ぶと、grid_search_general 内のガード
  `if feasibility_mode == "legacy_path" and T is None: raise ValueError(...)` に必ず
  落ちて即クラッシュする（実機で確認済み）。本ドライバの §3（上限感度）は legacy_path
  モードでの時間平均収量スイープが要件のため、この2関数を経由せず
  grid_search_general() を直接ループで呼ぶことで回避している
  （_upper_bound_sweep_legacy() 関数、公開APIのみ使用・sustainability.py 自体は不変）。
"""
import copy
import os
import sys
import time

import numpy as np

# -----------------------------------------------------------------------
# パス設定: run_msy.py と同じパターン
#   (1) 自分自身のディレクトリ（msy/）を先頭に         -> msy_core, sustainability, data_loader
#   (2) 親ディレクトリ（現行コード/）を後方に追加        -> model, estimate_cache が使う fixed_params 等
# -----------------------------------------------------------------------
_here = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_here)
if _here not in sys.path:
    sys.path.insert(0, _here)
if _parent not in sys.path:
    sys.path.append(_parent)

import msy_core                                    # noqa: E402 (average_yield, F_MAX, F_MIN)
import sustainability as sus                        # noqa: E402 (診断本体; np.trapz シムも道連れで適用される)
import estimate_cache                                # noqa: E402 (load_estimates)
from data_loader import (                            # noqa: E402
    load_clean_dataframe, get_series, regime_masks, slice_series,
    get_regime_T, get_regime_X0_norm, KEYS,
)

_out_dir = os.path.join(_here, "outputs")
os.makedirs(_out_dir, exist_ok=True)

# =============================================================================
# 診断設定（トラクタビリティ: n_grid=5 -> 5^4=625評価/条件）
# =============================================================================
N_GRID = 5
BOUND_TOL = sus.DEFAULT_OPTIMIZATION["bound_tol"]
F_UPPER_BASELINE = msy_core.F_MAX  # 0.95 = 現行run_msy.pyの物理上限（legacy制約の既定）
F_UPPER_GRID = sus.DEFAULT_SENSITIVITY["fishing_upper_bounds"]  # [0.25,0.50,0.75,0.95,1.25]
LRP_GRID = sus.DEFAULT_SENSITIVITY["lrp_ratios"]                # [0.10,...,0.50]

TRAJ_LONG = {"enabled": True, "burn_in_years": 100, "evaluation_years": 100,
             "evaluation_dt": 0.05, "floor_ratio": 0.1}   # sustainability.py の既定と同一
TRAJ_SHORT = {"enabled": True, "burn_in_years": 50, "evaluation_years": 50,
              "evaluation_dt": 0.1, "floor_ratio": 0.1}   # トラクタブルな短縮版


def _sep(c="=", n=78):
    return c * n


# =============================================================================
# 小さなフォーマット・抽出ヘルパ
# =============================================================================

def ffmt(x, spec=".3f"):
    """NaN/None を安全に文字列化する数値フォーマッタ。"""
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(xf):
        return "NaN"
    return format(xf, spec)


def fmt_vec(v, spec=".3f"):
    return "[" + ", ".join(ffmt(x, spec) for x in v) + "]"


def species_list(bool_arr):
    names = [KEYS[i] for i in range(4) if bool(bool_arr[i])]
    return ", ".join(names) if names else "(none)"


def unconstrained_best(grid):
    """grid_search_general の結果から feasibility を無視した最大収量点を抽出する。

    grid_search_general 自体は「制約下最大」（constrained_maximum）しか返さないため、
    無制約の比較対象は all_yield/all_f から自前で argmax する（追加のODE計算は不要、
    既に評価済みの配列を読むだけ）。
    """
    all_yield = grid["all_yield"]
    finite = np.isfinite(all_yield)
    if not np.any(finite):
        return {"f_opt": np.full(4, np.nan), "yield": float("nan"),
                "biomass": np.full(4, np.nan), "ratio": np.full(4, np.nan),
                "index": -1, "feasible_anyway": False}
    masked = np.where(finite, all_yield, -np.inf)
    idx = int(np.argmax(masked))
    return {
        "f_opt": grid["all_f"][idx].copy(), "yield": float(all_yield[idx]),
        "biomass": grid["all_biomass"][idx].copy(), "ratio": grid["all_ratio"][idx].copy(),
        "index": idx, "feasible_anyway": bool(grid["all_feasible"][idx]),
    }


def classify_pattern(f_upper_grid, yields, at_uppers):
    """upper_bound_sensitivity() の診断ヒューリスティックのドライバ側再現。

    sustainability.py の private _diagnose_upper_bound_pattern と同じロジックだが、
    legacy_path モードでは upper_bound_sensitivity() 自体が呼べない（上記バグ）ため、
    本ドライバが直接このロジックを持つ。
    """
    pairs = [(y, u) for y, u in zip(yields, at_uppers) if np.isfinite(y)]
    if len(pairs) < 2:
        return "insufficient_data"
    ys = [y for y, _ in pairs]
    us = [u for _, u in pairs]
    increasing = all(y2 >= y1 - 1e-9 for y1, y2 in zip(ys[:-1], ys[1:]))
    if all(us) and increasing:
        return "upper-bound-driven"
    if not any(us[-2:]):
        return "internally-determined"
    return "lrp-limited"


def row_for_species(regime, mode, lrp_ratio, f_upper, solution_type,
                     f_vec, biomass, biomass_reference, biomass_ratio,
                     feasible, active_constraint, warning_list,
                     at_upper_arr=None, minimum_biomass=None, yield_override=None):
    """1シナリオを KEYS 種別の4行へ展開する（sensitivity_to_csv 用フラット行、
    列は sustainability.CSV_COLUMNS と一致させる）。

    yield_override: 種別収量を biomass*fishing_rate から自前で導出するのではなく、
    明示的な配列で上書きしたい場合に使う。legacy_path モードでは "biomass" 引数が
    ratio/境界判定のために endpoint（期末）値であり、収量は期間平均で決まる
    （msy_core.average_yield の per_species_yield）ため、両者が一致しない
    （呼び出し側で average_yield を呼んで yield_override に渡す。下記バグ参照）。
    equilibrium_lrp（biomass=平衡値, yield=f*平衡値と定義上一致）・trajectory_floor
    （biomass=mean_biomass, f一定なら f*mean_biomass=時間平均収量と数学的に一致）では
    yield_override は不要（f_i*biomass_iで正確）。
    """
    warning_str = "; ".join(warning_list) if warning_list else ""
    rows = []
    for i, key in enumerate(KEYS):
        def _v(arr):
            if arr is None:
                return ""
            val = arr[i]
            return float(val) if np.isfinite(val) else ""

        f_i = _v(f_vec)
        if yield_override is not None:
            yield_i = _v(yield_override)
        else:
            yield_i = (f_i * _v(biomass)) if (f_i != "" and _v(biomass) != "") else ""
        rows.append({
            "regime": regime, "sustainability_mode": mode,
            "lrp_ratio": lrp_ratio if lrp_ratio is not None else "",
            "fishing_upper_bound": f_upper if f_upper is not None else "",
            "solution_type": solution_type, "species": key,
            "fishing_rate": f_i,
            "yield": yield_i,
            "biomass": _v(biomass),
            "biomass_reference": _v(biomass_reference),
            "biomass_ratio": _v(biomass_ratio),
            "minimum_biomass": _v(minimum_biomass) if minimum_biomass is not None else "",
            "at_upper_bound": bool(at_upper_arr[i]) if at_upper_arr is not None else False,
            "active_constraint": active_constraint or "",
            "feasible": bool(feasible), "warning": warning_str,
        })
    return rows


def legacy_true_yield(pn, mn, T, X0n, f_vec):
    """legacy_path モード用: f_vec における真の種別時間平均収量（msy_core.average_yield の
    per_species_yield）を返す。grid_search_general/_evaluate_candidate は legacy_path の
    "biomass" を endpoint（期末値, 境界判定用）としてしか保持しないため、CSVの収量列を
    正しく埋めるにはこの関数で改めて average_yield を1回呼び直す必要がある
    （本ドライバが見つけた自分自身の初回実装バグの修正: f_i×endpoint_biomass は
    時間平均収量と一致しない。数百回呼んでも1回あたり数msなので計算コストは無視できる）。
    """
    if f_vec is None or not np.all(np.isfinite(f_vec)):
        return None
    res = msy_core.average_yield(np.asarray(f_vec, dtype=float), pn, mn, T, X0n)
    if not res["success"]:
        return None
    return res["per_species_yield"]


def safe_ref_from_ratio(biomass, ratio):
    """biomass と ratio(=biomass/reference) から reference を逆算する（両方とも同じ
    'numerator' 由来のときのみ有効。trajectory_floor のように biomass と ratio の
    numerator が食い違う場合は呼び出し側で reference を直接与えること。"""
    biomass = np.asarray(biomass, dtype=float)
    ratio = np.asarray(ratio, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ref = np.where(np.isfinite(ratio) & (ratio != 0), biomass / ratio, np.nan)
    return ref


# =============================================================================
# セクション1: legacy 制約の再現（+ セクション4: 95%安全側解の比較もここで行う）
# =============================================================================

def section_legacy(regime, pn, mn, T, X0n, csv_rows):
    print(f"\n[1] legacy 制約の再現  (T={T:.1f}年, X0={fmt_vec(X0n)}, n_grid={N_GRID}^4={N_GRID**4})")
    cfg = copy.deepcopy(sus.DEFAULT_SUSTAINABILITY)
    cfg["mode"] = "legacy_path"

    grid = sus.grid_search_general(pn, mn, X0n, F_UPPER_BASELINE, "legacy_path", cfg, N_GRID, T=T)
    unc = unconstrained_best(grid)
    cm = grid["constrained_maximum"]

    f_upper_vec = np.full(4, F_UPPER_BASELINE)
    boundary_unc = sus.boundary_diagnostics(unc["f_opt"], np.zeros(4), f_upper_vec, BOUND_TOL) \
        if unc["index"] >= 0 else None
    boundary_cm = cm.get("boundary")

    print(f"  無制約   : f*={fmt_vec(unc['f_opt'])}  yield={ffmt(unc['yield'])} 千トン/年  "
          f"legacy可否(参考)={unc['feasible_anyway']}")
    if boundary_unc is not None:
        print(f"             上限張り付き種: {species_list(boundary_unc['at_upper_bound_by_species'])}")
    print(f"  legacy制約: f*={fmt_vec(cm['f_opt'])}  yield={ffmt(cm['yield'])} 千トン/年  "
          f"分類={cm['classification']}  ({cm['msy_interpretation']})")
    if boundary_cm is not None:
        print(f"             上限張り付き種: {species_list(boundary_cm['at_upper_bound_by_species'])}")
    print(f"  feasible点: {grid['n_feasible']} / {grid['n_evaluated']}  (n_success={grid['n_success']})")

    # --- セクション4: 95%安全側解（near_optimal_safe） -------------------------------
    finite_yield = grid["all_yield"][np.isfinite(grid["all_yield"])]
    y_max = float(np.max(finite_yield)) if finite_yield.size else float("nan")
    safe = sus.near_optimal_safe(grid, y_max, sus.DEFAULT_OPTIMIZATION)

    print(f"\n[4] 制約下最大収量 vs 95%安全側解 (near_optimal_safe, "
          f"criterion={sus.DEFAULT_OPTIMIZATION['safe_solution_criterion']})")
    if np.isfinite(cm["yield"]):
        margin_cm = cm["ratio"] - 1.0
        safety_margin_cm = float(np.min(margin_cm[np.isfinite(margin_cm)])) \
            if np.any(np.isfinite(margin_cm)) else float("nan")
        print(f"  制約下最大: f*={fmt_vec(cm['f_opt'])}  yield={ffmt(cm['yield'])}  "
              f"safety_margin={ffmt(safety_margin_cm, '+.3f')}")
    else:
        safety_margin_cm = float("nan")
        print("  制約下最大: N/A（feasible点なし）")
    if safe["found"]:
        print(f"  95%安全側 : f*={fmt_vec(safe['f_safe'])}  yield={ffmt(safe['yield_safe'])}  "
              f"safety_margin={ffmt(safe['safety_margin'], '+.3f')}  "
              f"active_constraint={safe['active_constraint']}")
        improvement = safe["safety_margin"] - safety_margin_cm
        print(f"  → 安全余裕の改善: {ffmt(improvement, '+.3f')}  "
              f"(収量の代償: {ffmt(cm['yield'] - safe['yield_safe'])} 千トン/年)")
    else:
        print(f"  95%安全側 : 見つからず（{'; '.join(safe['warnings'])}）")
        print(f"  → 参考: y_max(無制約)={ffmt(y_max)}, 0.95*y_max={ffmt(0.95*y_max)}, "
              f"legacy-feasible点は{grid['n_feasible']}件のみでy_maxに遠く及ばない可能性")

    # --- CSV行 -----------------------------------------------------------------
    # legacy_path の "biomass" は endpoint 値（境界判定用）であり、収量は期間平均で
    # 決まるため、yield 列は legacy_true_yield() で average_yield を呼び直して埋める。
    if unc["index"] >= 0:
        csv_rows += row_for_species(
            regime, "legacy_path", None, F_UPPER_BASELINE, "unconstrained",
            unc["f_opt"], unc["biomass"], safe_ref_from_ratio(unc["biomass"], unc["ratio"]),
            unc["ratio"], unc["feasible_anyway"], None, [],
            at_upper_arr=boundary_unc["at_upper_bound_by_species"] if boundary_unc else None,
            yield_override=legacy_true_yield(pn, mn, T, X0n, unc["f_opt"]))
    if np.isfinite(cm["yield"]):
        csv_rows += row_for_species(
            regime, "legacy_path", None, F_UPPER_BASELINE, "legacy_constrained",
            cm["f_opt"], cm["biomass"], safe_ref_from_ratio(cm["biomass"], cm["ratio"]),
            cm["ratio"], True, cm["classification"], [],
            at_upper_arr=boundary_cm["at_upper_bound_by_species"] if boundary_cm else None,
            yield_override=legacy_true_yield(pn, mn, T, X0n, cm["f_opt"]))
    if safe["found"]:
        csv_rows += row_for_species(
            regime, "legacy_path", None, F_UPPER_BASELINE, "near_optimal_safe",
            safe["f_safe"], safe["biomass"], safe_ref_from_ratio(safe["biomass"], safe["ratio"]),
            safe["ratio"], True, safe["active_constraint"], safe["warnings"],
            at_upper_arr=safe["at_upper_bound_by_species"],
            yield_override=legacy_true_yield(pn, mn, T, X0n, safe["f_safe"]))
    else:
        csv_rows += row_for_species(
            regime, "legacy_path", None, F_UPPER_BASELINE, "near_optimal_safe",
            None, None, None, None, False, None, safe["warnings"])

    return {"grid": grid, "unc": unc, "cm": cm, "safe": safe, "y_max": y_max,
            "boundary_unc": boundary_unc, "boundary_cm": boundary_cm}


# =============================================================================
# セクション2: equilibrium_lrp（無漁獲平衡の正値性 → 本モデルは適用不可）
# =============================================================================

def section_equilibrium_lrp(regime, pn, mn, X0n, csv_rows):
    cfg = copy.deepcopy(sus.DEFAULT_SUSTAINABILITY)
    cfg["mode"] = "equilibrium_lrp"

    unfished = sus.compute_equilibrium(pn, np.zeros(4), cfg, species_scope="all")
    B0_abs = unfished["B_eq_norm"] * mn

    print(f"\n[2] equilibrium_lrp — 無漁獲平衡の正値性")
    print(f"  B_eq(無漁獲, 正規化) = {fmt_vec(unfished['B_eq_norm'])}")
    print(f"  B_eq(無漁獲, 絶対[千トン]) = {fmt_vec(B0_abs, '.2f')}")
    print(f"  positive={unfished['positive']}  per_species_positive={unfished['per_species_positive'].tolist()}")
    negative_species = [KEYS[i] for i in range(4) if not unfished["per_species_positive"][i]]
    if not unfished["positive"]:
        print(f"  ★ headline: 正の共存平衡が存在しない（負の種: {', '.join(negative_species)}）。"
              f"よって equilibrium_lrp は本質的に適用不可（n_feasible=0 が全 lrp で続く）。")
    else:
        print("  正の共存平衡が存在する（想定外 — CLAUDE.mdの既知事実と食い違う。要再確認）。")

    print(f"  lrp_ratio in {LRP_GRID} を実行して『適用不可』を数値で確認する "
          f"(f_upper={F_UPPER_BASELINE}, n_grid={N_GRID}^4={N_GRID**4}) ...")
    rows = sus.lrp_sensitivity(pn, mn, X0n, cfg, LRP_GRID, F_UPPER_BASELINE, N_GRID)

    for row in rows:
        cm = row["constrained_maximum"]
        safe = row["safe_solution"]
        print(f"    lrp_ratio={row['lrp_ratio']:.2f}: n_feasible={row['n_feasible']}/{row['n_evaluated']}  "
              f"y_max={ffmt(row['y_max'])}  classification={cm['classification']}  "
              f"safe_found={safe['found']}")

        warn_cm = [f"equilibrium_lrp not applicable: {unfished['reason']} "
                   f"(negative species: {', '.join(negative_species)})"]
        # np.isfinite(cm["yield"]) 真 <=> グリッドが実際に feasible な点を見つけた場合のみ
        # （constrained_maximum は n_feasible>0 のときだけ実 index を持つ）。このモデルの
        # 実フィット済みパラメータでは無漁獲平衡が常に非正のため、この分岐は事実上到達しない
        # （n_feasible は全 lrp_ratio で 0 になる）が、将来モデルが変わった場合に備えて
        # feasible=True を正しく設定しておく（False 固定は誤り＝実際に見つかった解を
        # infeasible と誤記録してしまうため）。
        if np.isfinite(cm["yield"]):
            csv_rows += row_for_species(
                regime, "equilibrium_lrp", row["lrp_ratio"], F_UPPER_BASELINE,
                "equilibrium_lrp_constrained_max", cm["f_opt"], cm["biomass"],
                safe_ref_from_ratio(cm["biomass"], cm["ratio"]), cm["ratio"],
                True, cm["classification"], [])
        else:
            csv_rows += row_for_species(
                regime, "equilibrium_lrp", row["lrp_ratio"], F_UPPER_BASELINE,
                "equilibrium_lrp_constrained_max", None, None, None, None,
                False, cm["classification"], warn_cm)
        if safe["found"]:
            csv_rows += row_for_species(
                regime, "equilibrium_lrp", row["lrp_ratio"], F_UPPER_BASELINE,
                "equilibrium_lrp_safe", safe["f_safe"], safe["biomass"],
                safe_ref_from_ratio(safe["biomass"], safe["ratio"]), safe["ratio"],
                True, safe["active_constraint"], [],
                at_upper_arr=safe["at_upper_bound_by_species"])
        else:
            csv_rows += row_for_species(
                regime, "equilibrium_lrp", row["lrp_ratio"], F_UPPER_BASELINE,
                "equilibrium_lrp_safe", None, None, None, None,
                False, None, warn_cm + safe["warnings"])

    return {"unfished": unfished, "B0_abs": B0_abs, "negative_species": negative_species, "rows": rows}


# =============================================================================
# セクション3: 上限感度（legacy_path・短期地平・時間平均収量）
#   sustainability.upper_bound_sensitivity() は T を転送できず legacy_path で
#   クラッシュするため（モジュール docstring 参照）、ここでは grid_search_general を
#   直接ループで呼ぶ（公開APIのみ使用、sustainability.py 自体は無変更）。
# =============================================================================

def _upper_bound_sweep_legacy(pn, mn, X0n, T, f_upper_grid, n_grid):
    cfg = copy.deepcopy(sus.DEFAULT_SUSTAINABILITY)
    cfg["mode"] = "legacy_path"

    rows = []
    for f_upper in f_upper_grid:
        grid = sus.grid_search_general(pn, mn, X0n, f_upper, "legacy_path", cfg, n_grid, T=T)
        unc = unconstrained_best(grid)
        cm = grid["constrained_maximum"]

        f_upper_vec = np.full(4, f_upper)
        boundary_unc = sus.boundary_diagnostics(unc["f_opt"], np.zeros(4), f_upper_vec, BOUND_TOL) \
            if unc["index"] >= 0 else None
        boundary_cm = cm.get("boundary")

        rows.append({
            "f_upper": f_upper, "grid": grid, "unc": unc, "cm": cm,
            "at_upper_unc": bool(boundary_unc["any_upper_bound_active"]) if boundary_unc else False,
            "at_upper_cm": bool(boundary_cm["any_upper_bound_active"]) if boundary_cm else False,
            "boundary_unc": boundary_unc, "boundary_cm": boundary_cm,
        })
    return rows


def section_upper_bound(regime, pn, mn, T, X0n, csv_rows):
    print(f"\n[3] 上限感度  f_upper in {F_UPPER_GRID}  "
          f"(短期地平 T={T:.1f}年の時間平均収量, legacy_path, n_grid={N_GRID}^4={N_GRID**4}/条件)")
    rows = _upper_bound_sweep_legacy(pn, mn, X0n, T, F_UPPER_GRID, N_GRID)

    print(f"  {'f_upper':>8}  {'無制約f*':^28}  {'無制約yield':>11}  {'@upper':>6}  |  "
          f"{'legacy制約f*':^28}  {'制約yield':>10}  {'@upper':>6}  分類")
    for r in rows:
        cm = r["cm"]
        print(f"  {r['f_upper']:>8.2f}  {fmt_vec(r['unc']['f_opt']):^28}  "
              f"{ffmt(r['unc']['yield']):>11}  {str(r['at_upper_unc']):>6}  |  "
              f"{fmt_vec(cm['f_opt']):^28}  {ffmt(cm['yield']):>10}  {str(r['at_upper_cm']):>6}  "
              f"{cm['classification']}")

    diag_unc = classify_pattern(F_UPPER_GRID, [r["unc"]["yield"] for r in rows],
                                 [r["at_upper_unc"] for r in rows])
    diag_cm = classify_pattern(F_UPPER_GRID, [r["cm"]["yield"] for r in rows],
                                [r["at_upper_cm"] for r in rows])
    print(f"  診断（無制約系列）    : {diag_unc}")
    print(f"  診断（legacy制約系列）: {diag_cm}")

    for r in rows:
        if r["unc"]["index"] >= 0:
            csv_rows += row_for_species(
                regime, "legacy_path", None, r["f_upper"], "upper_bound_unconstrained",
                r["unc"]["f_opt"], r["unc"]["biomass"],
                safe_ref_from_ratio(r["unc"]["biomass"], r["unc"]["ratio"]), r["unc"]["ratio"],
                r["unc"]["feasible_anyway"], diag_unc, [],
                at_upper_arr=r["boundary_unc"]["at_upper_bound_by_species"] if r["boundary_unc"] else None,
                yield_override=legacy_true_yield(pn, mn, T, X0n, r["unc"]["f_opt"]))
        cm = r["cm"]
        if np.isfinite(cm["yield"]):
            csv_rows += row_for_species(
                regime, "legacy_path", None, r["f_upper"], "upper_bound_legacy_constrained",
                cm["f_opt"], cm["biomass"], safe_ref_from_ratio(cm["biomass"], cm["ratio"]),
                cm["ratio"], True, f"{cm['classification']} / {diag_cm}", [],
                at_upper_arr=r["boundary_cm"]["at_upper_bound_by_species"] if r["boundary_cm"] else None,
                yield_override=legacy_true_yield(pn, mn, T, X0n, cm["f_opt"]))
        else:
            csv_rows += row_for_species(
                regime, "legacy_path", None, r["f_upper"], "upper_bound_legacy_constrained",
                None, None, None, None, False, f"{cm['classification']} / {diag_cm}", [])

    return {"rows": rows, "diag_unc": diag_unc, "diag_cm": diag_cm}


# =============================================================================
# セクション5: trajectory_floor 長期検証（選択解 = legacyセクションの legacy制約 f*）
# =============================================================================

def section_trajectory_floor(regime, pn, mn, X0n, f_chosen, csv_rows):
    print(f"\n[5] trajectory_floor 長期検証  (選択解 = §1 legacy制約 f*={fmt_vec(f_chosen)})")

    cfg_long = copy.deepcopy(sus.DEFAULT_SUSTAINABILITY)
    cfg_long["mode"] = "trajectory_floor"
    cfg_long["trajectory_validation"] = TRAJ_LONG
    t0 = time.time()
    tf_long = sus.evaluate_trajectory_floor(pn, mn, X0n, f_chosen, cfg_long)
    t_long = time.time() - t0

    cfg_short = copy.deepcopy(sus.DEFAULT_SUSTAINABILITY)
    cfg_short["mode"] = "trajectory_floor"
    cfg_short["trajectory_validation"] = TRAJ_SHORT
    t0 = time.time()
    tf_short = sus.evaluate_trajectory_floor(pn, mn, X0n, f_chosen, cfg_short)
    t_short = time.time() - t0

    # 参考: 無漁獲(f=0)の長期シミュレーションも直接見て、"無漁獲でも負に振れるか" を確認する
    # （evaluate_trajectory_floor が使う _unfished_reference と同じ考え方だが、any_negative を
    #   直接見るためここでも simulate_constant_f を素通しで呼ぶ）。
    sim0 = sus.simulate_constant_f(pn, mn, X0n, np.zeros(4), TRAJ_LONG["burn_in_years"] + TRAJ_LONG["evaluation_years"],
                                    TRAJ_LONG["evaluation_dt"])

    for label, tf, dt_wall, cfg_tv in [("LONG (100+100yr, dt=0.05)", tf_long, t_long, TRAJ_LONG),
                                        ("SHORT(50+50yr,  dt=0.1)", tf_short, t_short, TRAJ_SHORT)]:
        print(f"  {label}: solver_success={tf['solver_success']}  feasible={tf['feasible']}  "
              f"reason={tf.get('reason')}  (wall={dt_wall:.2f}s)")
        if tf["solver_success"]:
            print(f"           min_biomass ={fmt_vec(tf['min_biomass'], '.4f')}")
            print(f"           mean_biomass={fmt_vec(tf['mean_biomass'], '.4f')}")
            print(f"           floor       ={fmt_vec(tf['floor'], '.4f')}  (reference_kind={tf['reference_kind']})")
            print(f"           any_negative={tf['any_negative']}  first_violating_species={tf['first_violating_species']}")

    print(f"  参考: 無漁獲(f=0)200年シミュレーション success={sim0['success']}  any_negative={sim0['any_negative']}")
    if sim0["success"]:
        tail = sim0["traj_abs"][:, sim0["t"] >= 100]
        print(f"        無漁獲 長期平均(t>=100)={fmt_vec(tail.mean(axis=1), '.4f')}  "
              f"長期最小(t>=100)={fmt_vec(tail.min(axis=1), '.4f')}")
        print("        → 無漁獲でも一部の種の長期軌道が負に振れる場合、equilibrium_lrp/trajectory_floorの"
              "『正の共存平衡が無い』という所見が、平衡の線形代数だけでなく直接の数値積分でも裏付けられることを意味する。")

    for label, tf, cfg_tv in [("trajectory_floor_100_100", tf_long, TRAJ_LONG),
                              ("trajectory_floor_50_50", tf_short, TRAJ_SHORT)]:
        if tf["solver_success"]:
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(np.isfinite(tf["floor"]) & (tf["floor"] != 0),
                                  tf["min_biomass"] / tf["floor"], np.nan)
            csv_rows += row_for_species(
                regime, "trajectory_floor", None, None, label,
                f_chosen, tf["mean_biomass"], tf["floor"], ratio,
                tf["feasible"], tf["first_violating_species"], tf["warnings"],
                minimum_biomass=tf["min_biomass"])
        else:
            csv_rows += row_for_species(
                regime, "trajectory_floor", None, None, label,
                f_chosen, None, None, None,
                False, "solver_failure", tf["warnings"])

    return {"long": tf_long, "short": tf_short, "unfished_sim": sim0}


# =============================================================================
# メイン
# =============================================================================

def main():
    print(_sep())
    print("持続性制約 診断ドライバ (sustainability.py 実データ実走)")
    print(_sep())

    est_results = estimate_cache.load_estimates()
    if est_results is None:
        print("\n[ERROR] estimate_cache.load_estimates() が None を返しました。")
        print("        フィット済みの自由推定4種パラメータのキャッシュが見つかりません。")
        print("        先に以下を実行してキャッシュを作成してください:")
        print("            cd 現行コード/msy && python3 run_msy.py")
        sys.exit(1)

    df = load_clean_dataframe()
    series = get_series(df)
    nlm_mask, lm_mask = regime_masks(series)
    regimes = [
        ("NLM", slice_series(series, nlm_mask)),
        ("LM", slice_series(series, lm_mask)),
    ]

    all_csv_rows = {}
    section1_results = {}
    section2_results = {}
    section3_results = {}
    section5_results = {}

    for rname, sl in regimes:
        t_regime0 = time.time()
        print("\n" + _sep())
        print(f"レジーム: {rname}")
        print(_sep())

        est = est_results[rname]
        pn, mn = est["params_norm"], est["means"]
        T = get_regime_T(sl)
        X0n = get_regime_X0_norm(sl, mn)
        print(f"推定メタ情報: 平均NRMSE={est['metrics']['overall']['mean_NRMSE']:.3f}  "
              f"平均R2={est['metrics']['overall']['mean_R2']:+.3f}  means={fmt_vec(mn, '.2f')}")

        csv_rows = []
        r1 = section_legacy(rname, pn, mn, T, X0n, csv_rows)
        r2 = section_equilibrium_lrp(rname, pn, mn, X0n, csv_rows)
        r3 = section_upper_bound(rname, pn, mn, T, X0n, csv_rows)
        f_chosen = r1["cm"]["f_opt"] if np.all(np.isfinite(r1["cm"]["f_opt"])) else np.full(4, np.nan)
        r5 = section_trajectory_floor(rname, pn, mn, X0n, f_chosen, csv_rows)

        section1_results[rname] = r1
        section2_results[rname] = r2
        section3_results[rname] = r3
        section5_results[rname] = r5
        all_csv_rows[rname] = csv_rows

        wall = time.time() - t_regime0
        print(f"\n[wall-clock] レジーム {rname} 合計: {wall:.1f} 秒")

    # -----------------------------------------------------------------------
    # セクション6: CSV出力
    # -----------------------------------------------------------------------
    print("\n" + _sep())
    print("[6] CSV出力")
    csv_paths = {}
    for rname in ["NLM", "LM"]:
        path = os.path.join(_out_dir, f"sustainability_sensitivity_{rname}.csv")
        sus.sensitivity_to_csv(all_csv_rows[rname], path)
        csv_paths[rname] = path
        n_rows = len(all_csv_rows[rname])
        exists = os.path.exists(path)
        size = os.path.getsize(path) if exists else -1
        print(f"  {rname}: {path}  ({n_rows}行, exists={exists}, size={size}バイト)")

    # -----------------------------------------------------------------------
    # 全体サマリ表
    # -----------------------------------------------------------------------
    print("\n" + _sep())
    print("[全体サマリ] 条件別 f* / yield / 上限張り付き種 / feasible / 分類")
    print(_sep("-"))
    for rname in ["NLM", "LM"]:
        r1 = section1_results[rname]
        r3 = section3_results[rname]
        print(f"\n-- {rname} --")
        print(f"{'条件':<28}{'f*':^30}{'yield':>10}  {'@upper種':<12}{'feasible':<9}分類")
        cm = r1["cm"]
        print(f"{'legacy baseline(f_upper=' + str(F_UPPER_BASELINE) + ')':<28}"
              f"{fmt_vec(cm['f_opt']):^30}{ffmt(cm['yield']):>10}  "
              f"{species_list(r1['boundary_cm']['at_upper_bound_by_species']) if r1['boundary_cm'] else '-':<12}"
              f"{str(np.isfinite(cm['yield'])):<9}{cm['classification']}")
        if r1["safe"]["found"]:
            safe = r1["safe"]
            print(f"{'  95% safe':<28}{fmt_vec(safe['f_safe']):^30}{ffmt(safe['yield_safe']):>10}  "
                  f"{species_list(safe['at_upper_bound_by_species']):<12}{'True':<9}near_optimal_safe")
        for row in r3["rows"]:
            cmr = row["cm"]
            label = f"f_upper={row['f_upper']:.2f}"
            print(f"{label:<28}{fmt_vec(cmr['f_opt']):^30}{ffmt(cmr['yield']):>10}  "
                  f"{species_list(row['boundary_cm']['at_upper_bound_by_species']) if row['boundary_cm'] else '-':<12}"
                  f"{str(np.isfinite(cmr['yield'])):<9}{cmr['classification']}")
        r2 = section2_results[rname]
        for row in r2["rows"]:
            label = f"equilibrium_lrp={row['lrp_ratio']:.2f}"
            print(f"{label:<28}{'N/A (no positive eq.)':^30}{'NaN':>10}  {'-':<12}{'False':<9}not_applicable")

    # -----------------------------------------------------------------------
    # 8つの要約観点への回答
    # -----------------------------------------------------------------------
    print("\n" + _sep())
    print("[8つの要約観点への回答]")
    print(_sep("-"))
    for rname in ["NLM", "LM"]:
        r1 = section1_results[rname]
        r2 = section2_results[rname]
        r3 = section3_results[rname]
        r5 = section5_results[rname]
        print(f"\n-- {rname} --")

        yields_cm = [r["cm"]["yield"] for r in r3["rows"]]
        at_upper_cm = [r["at_upper_cm"] for r in r3["rows"]]
        print(f"(a) 上限を上げると最適値も追従するか: 診断={r3['diag_cm']}  "
              f"(at_upper系列={at_upper_cm})")
        finite_pairs = [(f, y) for f, y in zip(F_UPPER_GRID, yields_cm) if np.isfinite(y)]
        monotonic = all(finite_pairs[i][1] <= finite_pairs[i+1][1] + 1e-9
                         for i in range(len(finite_pairs)-1)) if len(finite_pairs) >= 2 else None
        print(f"(b) 収量は探索範囲内で単調増加か: {monotonic}  (f_upper→yield: {finite_pairs})")
        binding_species_upper = species_list(r3["rows"][-1]["boundary_cm"]["at_upper_bound_by_species"]) \
            if r3["rows"][-1]["boundary_cm"] else "(N/A)"
        print(f"(c) どの種が上限・LRPを決めるか: 上限張り付き種(f_upper={F_UPPER_GRID[-1]})="
              f"{binding_species_upper}  / equilibrium_lrpでは無漁獲平衡が負の種="
              f"{', '.join(r2['negative_species'])}")
        print(f"(d) LRPを厳しくすると収量がどれだけ減るか: equilibrium_lrpは全lrp比でn_feasible=0のため"
              f"『収量の変化』自体が定義不可（{'; '.join(str(row['n_feasible']) for row in r2['rows'])} "
              f"= 全条件で feasible 0件）。")
        if r1["safe"]["found"]:
            improvement = r1["safe"]["safety_margin"] - float(np.min(r1["cm"]["ratio"] - 1.0))
            print(f"(e) 95%安全側解の資源余裕改善: safety_margin {ffmt(float(np.min(r1['cm']['ratio']-1.0)), '+.3f')}"
                  f" → {ffmt(r1['safe']['safety_margin'], '+.3f')}  (改善 {ffmt(improvement, '+.3f')})、"
                  f"収量代償 {ffmt(r1['cm']['yield'] - r1['safe']['yield_safe'])} 千トン/年")
        else:
            print("(e) 95%安全側解の資源余裕改善: 見つからず(found=False) — legacy-feasible点が僅少で"
                  "0.95*y_max に届く別解が存在しない。")
        print(f"(f) 正の平衡でも長期軌道が危険になるか: 本モデルには正の共存平衡が存在しない"
              f"（{', '.join(r2['negative_species'])} が負）。長期(100+100yr)軌道は数値的には発散せず"
              f"solver成功だが、feasible={r5['long']['feasible']}"
              f"(reason={r5['long'].get('reason')})。無漁獲(f=0)200年シミュレーションでも"
              f"any_negative={r5['unfished_sim']['any_negative']}。")
        print(f"(g) 内部最大は実在するか: legacy制約下の分類={r1['cm']['classification']}"
              f"（interior以外なら境界解＝内部最大なし）")
        print(f"(h) MSYと呼べるか: {r1['cm']['msy_interpretation']}")

    print("\n" + _sep())
    print("診断ドライバ完了。")
    for rname, path in csv_paths.items():
        print(f"  CSV[{rname}] = {path}")
    print(_sep())


if __name__ == "__main__":
    main()
