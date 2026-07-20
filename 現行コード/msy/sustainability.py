"""
持続性制約（sustainability constraint）の新モード群。

`msy_core.check_sustainability`（legacy: 初期資源量比・位相依存）に代わる/併用する
判定方式を提供する:

  - equilibrium_lrp   : 内部平衡（無漁獲 B0eq・漁獲下 Bfeq）を解析的に求め、
                        Bfeq が limit reference point（LRP = lrp_ratio * B0eq）を
                        上回るかで判定する。X0（初期値）に依存しない。
  - trajectory_floor  : burn-in を捨てた評価窓で軌道の「最小値」が
                        floor_ratio * B0eq を下回らないかを判定する。
  - time_average_lrp  : 評価窓の「平均」資源量が average_lrp_ratio * 基準値以上かを判定する。
  - legacy_path       : msy_core.check_sustainability に委譲（既存の run_msy.py と完全互換）。

ODE右辺（正規化空間, model.make_ode, capacity_ry 12変数）:
    dx1 = (r_x1 - fx1)*x1 - L11*x1*y1 - L12*x1*y2
    dx2 = (r_x2 - fx2)*x2 - L21*x2*y1 - L22*x2*y2
    dy1 = (-r_y1 - fy1)*y1 + C1*L11*x1*y1 + D1*L21*x2*y1
    dy2 = (-r_y2 - fy2)*y2 + C2*L12*x1*y2 + D2*L22*x2*y2

一般化 Lotka-Volterra 表現（1株あたり成長率 = rho + A·B, B=[x1,x2,y1,y2]）:
    rho = [ r_x1-fx1,  r_x2-fx2,  -r_y1-fy1,  -r_y2-fy2 ]
    A   = [[0,      0,      -L11,   -L12 ],
           [0,      0,      -L21,   -L22 ],
           [C1*L11, D1*L21, 0,      0    ],
           [C2*L12, D2*L22, 0,      0    ]]
    （A の対角は常に 0 = 種内競争・自己制限が無いモデルであることに対応。
      密度依存項が無いため、内部平衡が存在しても中立安定（固有値の実部が
      ちょうど0）になり得る古典 Lotka-Volterra 的な系である点に注意。）

内部平衡 A@B_eq = -rho は 2 つの 2x2 系に分解できる（np.linalg.solve のみ使用、
逆行列は使わない）:
    [[L11,L12],[L21,L22]]           @ [y1,y2] = [r_x1-fx1, r_x2-fx2]
    [[C1*L11,D1*L21],[C2*L12,D2*L22]] @ [x1,x2] = [r_y1+fy1, r_y2+fy2]

平衡でのヤコビアン: dB_i/dt = B_i*(rho_i + (A B)_i) を微分し、平衡上で
(rho_i + (A B)_i) = 0 であることを使うと J_ij = B_i^eq * A_ij、すなわち
J = diag(B_eq) @ A。固有値の実部の符号で安定性を診断する（密度依存無しの
系では実部がちょうど0近辺になり得るため stability_tol で丸め誤差を吸収する）。

既知の事実（2026-07-14 時点の実フィット済みパラメータ, capacity_ry 自由推定）:
  無漁獲でも正の共存平衡が存在しない（NLM は x1<0、LM は x2<0）。
  よって compute_equilibrium / evaluate_equilibrium_lrp は「解けるが正でない」
  状態を必ず区別して返し、絶対にクラッシュしない設計にしてある。

設定は Python の dict/定数（YAML は使わない）。年数などはすべて cfg から読む。
"""
import copy
import csv as _csv
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp

# -----------------------------------------------------------------------
# パス設定: msy/run_msy.py 等と同じパターン
#   (1) 自分自身のディレクトリ（msy/）を先頭に         -> msy_core, data_loader(ブリ/サワラ版)
#   (2) 親ディレクトリ（現行コード/）を後方に追加        -> model
# -----------------------------------------------------------------------
_here = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_here)
if _here not in sys.path:
    sys.path.insert(0, _here)
if _parent not in sys.path:
    sys.path.append(_parent)

import model            # noqa: E402  (現行コード/model.py: make_ode, MODELS, _to_absolute)
import msy_core         # noqa: E402  (msy/msy_core.py: average_yield, check_sustainability, F_MIN, _const_f_ode)
from data_loader import KEYS  # noqa: E402  (msy/data_loader.py: ["x1","x2","y1","y2"])


# =============================================================================
# 設定（YAML不使用・Python定数）
# =============================================================================
DEFAULT_SUSTAINABILITY = {
    "mode": "equilibrium_lrp",          # legacy_path|equilibrium_lrp|trajectory_floor|time_average_lrp
    "lrp_ratio": 0.3,                   # scalar=全種共通 / list[4] / dict{key:ratio} を resolve_per_species で解決
    "equilibrium_eps": 1e-10,
    "cond_number_max": 1e12,
    "require_stable_equilibrium": False,
    "stability_tol": 1e-8,
    # legacy_path 用（現行ドライバ run_msy.py の SUSTAIN_CFG と完全一致させる）
    "legacy": {"scope": "all", "mode": "endpoint", "tol": 0.1},
    "trajectory_validation": {
        "enabled": True, "burn_in_years": 100, "evaluation_years": 100,
        "evaluation_dt": 0.05, "floor_ratio": 0.1,
    },
    "time_average": {
        "average_lrp_ratio": 0.3, "minimum_floor_ratio": 0.1, "require_minimum_floor": False,
        # reference_B: 明示的な基準資源量（4要素）。無漁獲平衡も長期シミュ平均も
        # 得られない場合の最終フォールバックとして参照する（既定では未設定=None）。
        "reference_B": None,
    },
}

DEFAULT_OPTIMIZATION = {
    "bound_tol": 1e-6, "near_optimal_fraction": 0.95,
    "safe_solution_criterion": "max_min_biomass_margin",  # min_total_fishing|max_min_biomass_margin|max_total_biomass_margin
}

DEFAULT_SENSITIVITY = {
    "fishing_upper_bounds": [0.25, 0.50, 0.75, 0.95, 1.25],
    "lrp_ratios": [0.10, 0.20, 0.30, 0.40, 0.50],
}

# constrained_maximum の分類が「境界がアクティブか」を判定するときの許容誤差
# （boundary_diagnostics に渡す既定 bound_tol と同じ値を使う）。
_ACTIVE_BOUND_TOL = DEFAULT_OPTIMIZATION["bound_tol"]

# scope 文字列 -> インデックス（msy_core._scope_map と同じ対応）
_SCOPE_MAP = {"all": [0, 1, 2, 3], "prey": [0, 1], "predator": [2, 3]}

# sensitivity_to_csv の推奨カラム
CSV_COLUMNS = [
    "regime", "sustainability_mode", "lrp_ratio", "fishing_upper_bound",
    "solution_type", "species", "fishing_rate", "yield", "biomass",
    "biomass_reference", "biomass_ratio", "minimum_biomass",
    "at_upper_bound", "active_constraint", "feasible", "warning",
]


def _scope_indices(scope):
    return _SCOPE_MAP.get(scope, _SCOPE_MAP["all"])


# =============================================================================
# 汎用ヘルパ
# =============================================================================

def resolve_per_species(value, keys=KEYS):
    """scalar / list・tuple・ndarray(len=len(keys)) / dict{key:val} を4要素 ndarray に解決する。

    - scalar          : 全種へブロードキャスト。
    - list/tuple/ndarray : 長さが len(keys) と一致することを検証して ndarray 化。
    - dict            : keys の順に値を取り出す（欠けているキーがあれば KeyError）。
    """
    n = len(keys)
    if isinstance(value, dict):
        return np.array([float(value[k]) for k in keys], dtype=float)
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value, dtype=float)
        if arr.shape != (n,):
            raise ValueError(
                f"resolve_per_species: expected shape ({n},), got {arr.shape} (value={value!r})")
        return arr
    # スカラー（int/float/np.floating 等）
    return np.full(n, float(value), dtype=float)


# =============================================================================
# 平衡（一般化 Lotka-Volterra）
# =============================================================================

def equilibrium_generalized_lv(A, rho, eps=1e-10, cond_max=1e12):
    """汎用の一般化 LV 内部平衡ソルバ: A @ B = -rho を np.linalg.solve で解く（逆行列は使わない）。

    次元非依存（n×n の A, 長さ n の rho ならなんでもよい）。テスト3の2種古典LV
    （A=[[0,-a],[b,0]], rho=[r,-m]）にも、capacity_ry の4種系にも同じ関数を使う。

    「解ける」("solvable"=True) と判定する条件:
      - A, rho が全て有限であること
      - A の最小特異値が eps を上回ること（cond が良くても絶対スケールで
        退化している病的ケース、例えば A 全体が一様に極小スケールで
        cond=1 だが実質ゼロ行列に近い場合を弾くため。cond_max だけでは
        相対的な悪条件しか検出できない）
      - 条件数 cond = smax/smin が cond_max 以下であること
      - np.linalg.solve が例外なく有限解を返すこと

    Parameters
    ----------
    A : array-like, shape (n, n)
    rho : array-like, shape (n,)
    eps : float
        最小特異値の下限（これ未満なら実質特異とみなす）。
    cond_max : float
        条件数の上限（これを超えたら数値的に不安定とみなす）。

    Returns
    -------
    dict: B_eq (n,), solvable (bool), cond (float), reason (str)
    """
    A = np.asarray(A, dtype=float)
    rho = np.asarray(rho, dtype=float)
    n = rho.shape[0]

    if not (np.all(np.isfinite(A)) and np.all(np.isfinite(rho))):
        return {"B_eq": np.full(n, np.nan), "solvable": False,
                "cond": float("nan"), "reason": "non_finite_input"}

    try:
        singular_values = np.linalg.svd(A, compute_uv=False)
    except np.linalg.LinAlgError as exc:
        return {"B_eq": np.full(n, np.nan), "solvable": False,
                "cond": float("inf"), "reason": f"svd_failed: {exc}"}

    smin = float(singular_values[-1])
    smax = float(singular_values[0])
    cond = float(smax / smin) if smin > 0 else float("inf")

    if not (smin >= eps):  # smin が NaN の場合もここで弾く
        return {"B_eq": np.full(n, np.nan), "solvable": False, "cond": cond,
                "reason": f"near_singular_matrix (min_singular_value={smin:.3e} < eps={eps:.3e})"}
    if not np.isfinite(cond) or cond > cond_max:
        return {"B_eq": np.full(n, np.nan), "solvable": False, "cond": cond,
                "reason": f"ill_conditioned (cond={cond:.3e} > cond_max={cond_max:.3e})"}

    try:
        B_eq = np.linalg.solve(A, -rho)
    except np.linalg.LinAlgError as exc:
        return {"B_eq": np.full(n, np.nan), "solvable": False, "cond": cond,
                "reason": f"linalg_solve_failed: {exc}"}

    if not np.all(np.isfinite(B_eq)):
        return {"B_eq": B_eq, "solvable": False, "cond": cond, "reason": "non_finite_solution"}

    return {"B_eq": B_eq, "solvable": True, "cond": cond, "reason": "ok"}


def build_A_rho(params_norm, f_vec):
    """capacity_ry の params_norm（12要素）と f_vec（4要素）から一般化LVの A, rho を作る。

    params_norm の並びは model.MODELS["capacity_ry"]["names"] と同一:
      [r_x1, r_x2, r_y1, r_y2, L11, L12, L21, L22, C1, D1, C2, D2]
    f_vec の並びは KEYS と同一: [fx1, fx2, fy1, fy2]

    model.make_ode の右辺 dB_i/dt = B_i * (rho_i + sum_j A_ij B_j) と整合するように
    構成する（本モジュール冒頭のモジュールdocstringに導出済み）。
    """
    params_norm = np.asarray(params_norm, dtype=float)
    f_vec = np.asarray(f_vec, dtype=float)
    r_x1, r_x2, r_y1, r_y2, L11, L12, L21, L22, C1, D1, C2, D2 = params_norm
    fx1, fx2, fy1, fy2 = f_vec

    rho = np.array([r_x1 - fx1, r_x2 - fx2, -r_y1 - fy1, -r_y2 - fy2])
    A = np.array([
        [0.0,      0.0,      -L11,     -L12],
        [0.0,      0.0,      -L21,     -L22],
        [C1 * L11, D1 * L21, 0.0,      0.0],
        [C2 * L12, D2 * L22, 0.0,      0.0],
    ])
    return A, rho


def compute_equilibrium(params_norm, f_vec, cfg, species_scope="all"):
    """capacity_ry の内部平衡を求め、対象種（species_scope）が全て正かを判定する。

    平衡自体は常に4種フル連立で解く（4状態は結合しているため部分系には分解できない）。
    species_scope は「どの成分が正であるべきか」という判定対象を絞るだけで、
    連立方程式そのものは変えない。

    Returns
    -------
    dict: B_eq_norm(4,), positive(bool), solvable(bool), cond(float),
          per_species_positive(4,bool), reason(str)
    """
    eps = cfg.get("equilibrium_eps", DEFAULT_SUSTAINABILITY["equilibrium_eps"])
    cond_max = cfg.get("cond_number_max", DEFAULT_SUSTAINABILITY["cond_number_max"])

    A, rho = build_A_rho(params_norm, f_vec)
    sol = equilibrium_generalized_lv(A, rho, eps=eps, cond_max=cond_max)
    B_eq = sol["B_eq"]
    idx = _scope_indices(species_scope)

    if sol["solvable"]:
        per_species_positive = np.array([bool(B_eq[i] > eps) for i in range(4)], dtype=bool)
        positive = bool(np.all(per_species_positive[idx]))
        reason = "ok" if positive else "non_positive_equilibrium_component"
    else:
        per_species_positive = np.zeros(4, dtype=bool)
        positive = False
        reason = sol["reason"]

    return {
        "B_eq_norm": B_eq,
        "positive": positive,
        "solvable": sol["solvable"],
        "cond": sol["cond"],
        "per_species_positive": per_species_positive,
        "reason": reason,
    }


def equilibrium_jacobian(params_norm, f_vec, B_eq_norm, stability_tol=1e-8):
    """平衡でのヤコビアン J=diag(B_eq)@A と固有値から安定性を診断する。

    stability の判定（優先順）:
      1. 固有値やB_eqが非有限 -> "unknown"
      2. いずれかの Re(λ) > stability_tol -> "unstable"
      3. 全ての Re(λ) < -stability_tol -> "stable"
      4. それ以外（|Re(λ)|<=stability_tol の固有値がある等）-> "neutral_or_marginal"
         （密度依存の無い古典LV系は内部平衡が中立中心になり得るため、この
         ケースを丸め誤差ごと吸収するのが stability_tol の役割）
    """
    B_eq_norm = np.asarray(B_eq_norm, dtype=float)

    if not np.all(np.isfinite(B_eq_norm)):
        return {"J": np.full((4, 4), np.nan), "eigenvalues": np.full(4, np.nan, dtype=complex),
                "max_real_eigenvalue": float("nan"), "equilibrium_stability": "unknown"}

    A, _ = build_A_rho(params_norm, f_vec)
    J = np.diag(B_eq_norm) @ A

    try:
        eigenvalues = np.linalg.eigvals(J)
    except np.linalg.LinAlgError:
        return {"J": J, "eigenvalues": np.full(4, np.nan, dtype=complex),
                "max_real_eigenvalue": float("nan"), "equilibrium_stability": "unknown"}

    if not np.all(np.isfinite(eigenvalues)):
        return {"J": J, "eigenvalues": eigenvalues,
                "max_real_eigenvalue": float("nan"), "equilibrium_stability": "unknown"}

    real_parts = eigenvalues.real
    max_real = float(np.max(real_parts))

    if np.any(real_parts > stability_tol):
        stability = "unstable"
    elif np.all(real_parts < -stability_tol):
        stability = "stable"
    else:
        stability = "neutral_or_marginal"

    return {"J": J, "eigenvalues": eigenvalues,
            "max_real_eigenvalue": max_real, "equilibrium_stability": stability}


# =============================================================================
# 定数漁獲圧シミュレーション
# =============================================================================

def simulate_constant_f(params_norm, means, X0_norm, f_vec, t_end, dt):
    """定数漁獲圧 f_vec のもとで capacity_ry ODE を積分する（msy_core._const_f_ode を再利用）。

    success は「solve_ivp が status==0 で完走し、かつ軌道が全て有限」であることのみで
    決まる（負値の有無は success に影響しない。負値は any_negative で別途報告する。
    model.make_ode 内部の状態フロアは微分評価にのみ適用され、solve_ivp が保持する
    状態そのものは負に振れうるため、この区別が必要）。

    Returns
    -------
    dict: t(n,), traj_abs(4,n) or None, success(bool), any_negative(bool), any_nonfinite(bool)
    """
    means = np.asarray(means, dtype=float)
    X0_norm = np.asarray(X0_norm, dtype=float)
    f_vec = np.asarray(f_vec, dtype=float)
    t_eval = np.arange(0.0, float(t_end), float(dt))

    ode = msy_core._const_f_ode(f_vec)

    try:
        sol = solve_ivp(
            ode, [0.0, float(t_end)], X0_norm.tolist(), args=(params_norm,),
            method="LSODA", rtol=1e-7, atol=1e-9, t_eval=t_eval,
        )
    except Exception:
        return {"t": t_eval, "traj_abs": None, "success": False,
                "any_negative": False, "any_nonfinite": True}

    finite_ok = bool(np.all(np.isfinite(sol.y))) if sol.y.size else False
    shape_ok = (sol.y.ndim == 2) and (sol.y.shape[1] == len(t_eval))
    success = bool(sol.status == 0 and shape_ok and finite_ok)

    if not success:
        return {"t": t_eval, "traj_abs": None, "success": False,
                "any_negative": False, "any_nonfinite": not finite_ok}

    traj_abs = sol.y * means[:, np.newaxis]
    any_negative = bool(np.any(traj_abs < 0))
    any_nonfinite = bool(not np.all(np.isfinite(traj_abs)))

    return {"t": t_eval, "traj_abs": traj_abs, "success": success and not any_nonfinite,
            "any_negative": any_negative, "any_nonfinite": any_nonfinite}


# =============================================================================
# 判定モード
# =============================================================================

def evaluate_equilibrium_lrp(params_norm, means, f_vec, cfg):
    """無漁獲平衡 B0eq と漁獲下平衡 Bfeq を解析的に求め、LRP（limit reference point）判定する。

    feasible の条件: 両平衡が正であり、かつ全対象種で Bfeq_i >= lrp_i * B0eq_i。
    収量: Y_eq = sum_i f_i * Bfeq_abs_i（平衡収量。X0 に依存しない）。

    平衡が「解けるが正でない」「解けない」場合は絶対にクラッシュせず、
    feasible=False と明確な reason 文字列を返す（実フィット済みパラメータで
    無漁獲平衡が正でないケースが実際に起こることが分かっているための設計）。
    """
    means = np.asarray(means, dtype=float)
    f_vec = np.asarray(f_vec, dtype=float)
    lrp_ratio = resolve_per_species(cfg.get("lrp_ratio", DEFAULT_SUSTAINABILITY["lrp_ratio"]))
    warnings = []

    unfished = compute_equilibrium(params_norm, np.zeros(4), cfg, species_scope="all")
    B0_norm = unfished["B_eq_norm"]
    B0_abs = B0_norm * means if np.all(np.isfinite(B0_norm)) else np.full(4, np.nan)

    if not unfished["positive"]:
        reason = "unfished_equilibrium_not_solvable" if not unfished["solvable"] \
            else "no_positive_unfished_equilibrium"
        warnings.append(f"detail: {unfished['reason']}")
        return {
            "feasible": False,
            "B_eq_unfished_norm": B0_norm, "B_eq_unfished_abs": B0_abs,
            "B_eq_fished_norm": np.full(4, np.nan), "B_eq_fished_abs": np.full(4, np.nan),
            "lrp_by_species": np.full(4, np.nan), "biomass_ratio": np.full(4, np.nan),
            "per_species_yield": np.full(4, np.nan), "total_yield": float("nan"),
            "active_lrp_species": None,
            "stability": {"equilibrium_stability": "unknown"},
            "warnings": warnings, "reason": reason,
        }

    fished = compute_equilibrium(params_norm, f_vec, cfg, species_scope="all")
    Bf_norm = fished["B_eq_norm"]
    Bf_abs = Bf_norm * means if np.all(np.isfinite(Bf_norm)) else np.full(4, np.nan)

    if not fished["positive"]:
        reason = "fished_equilibrium_not_solvable" if not fished["solvable"] \
            else "no_positive_fished_equilibrium"
        warnings.append(f"detail: {fished['reason']}")
        return {
            "feasible": False,
            "B_eq_unfished_norm": B0_norm, "B_eq_unfished_abs": B0_abs,
            "B_eq_fished_norm": Bf_norm, "B_eq_fished_abs": Bf_abs,
            "lrp_by_species": lrp_ratio * B0_abs, "biomass_ratio": np.full(4, np.nan),
            "per_species_yield": np.full(4, np.nan), "total_yield": float("nan"),
            "active_lrp_species": None,
            "stability": {"equilibrium_stability": "unknown"},
            "warnings": warnings, "reason": reason,
        }

    lrp_by_species = lrp_ratio * B0_abs
    biomass_ratio = Bf_abs / B0_abs
    feasible = bool(np.all(biomass_ratio >= lrp_ratio))
    active_lrp_species = KEYS[int(np.argmin(biomass_ratio))]

    per_species_yield = f_vec * Bf_abs
    total_yield = float(np.sum(per_species_yield))

    stability_tol = cfg.get("stability_tol", DEFAULT_SUSTAINABILITY["stability_tol"])
    stability = equilibrium_jacobian(params_norm, f_vec, Bf_norm, stability_tol=stability_tol)

    reason = "ok"
    if cfg.get("require_stable_equilibrium", DEFAULT_SUSTAINABILITY["require_stable_equilibrium"]):
        if stability["equilibrium_stability"] not in ("stable", "neutral_or_marginal"):
            feasible = False
            reason = "fished_equilibrium_unstable"
            warnings.append(
                f"Fished equilibrium is {stability['equilibrium_stability']}; "
                "require_stable_equilibrium=True rejects it.")
    if feasible is False and reason == "ok":
        reason = "lrp_violation"

    return {
        "feasible": feasible,
        "B_eq_unfished_norm": B0_norm, "B_eq_unfished_abs": B0_abs,
        "B_eq_fished_norm": Bf_norm, "B_eq_fished_abs": Bf_abs,
        "lrp_by_species": lrp_by_species, "biomass_ratio": biomass_ratio,
        "per_species_yield": per_species_yield, "total_yield": total_yield,
        "active_lrp_species": active_lrp_species,
        "stability": stability,
        "warnings": warnings, "reason": reason,
    }


def _unfished_reference(params_norm, means, X0_norm, cfg, t_end, dt, burn_in, warnings):
    """無漁獲平衡（正なら優先）→無漁獲長期シミュ平均、の順で基準資源量(絶対スケール)を返す。

    evaluate_trajectory_floor と evaluate_time_average_lrp の共通ロジック。
    長期シミュ平均は burn_in 以降（評価窓と同じ tail）で取る。burn_in 以前の
    遷移過程を含めると、X0_norm（観測初期値）由来の過渡応答に基準値が引きずられて
    しまうため、floor/mean_biomass 側と同じ「定常化した後の窓」で揃える。
    Returns (reference_abs(4,), reference_kind(str))。両方失敗時は (nan*4, "unavailable")。
    """
    unfished_eq = compute_equilibrium(params_norm, np.zeros(4), cfg, species_scope="all")
    if unfished_eq["positive"]:
        return unfished_eq["B_eq_norm"] * np.asarray(means, dtype=float), "unfished_equilibrium"

    warnings.append(
        "No positive unfished equilibrium; falling back to long-run unfished "
        "simulation average as reference.")
    sim0 = simulate_constant_f(params_norm, means, X0_norm, np.zeros(4), t_end, dt)
    if sim0["success"]:
        mask0 = sim0["t"] >= burn_in
        if not np.any(mask0):
            mask0 = np.ones_like(sim0["t"], dtype=bool)
        return sim0["traj_abs"][:, mask0].mean(axis=1), "unfished_long_run_mean"

    warnings.append("Unfished reference simulation also failed.")
    return np.full(4, np.nan), "unavailable"


def evaluate_trajectory_floor(params_norm, means, X0_norm, f_vec, cfg):
    """burn-in を捨てた評価窓で、軌道の最小値が floor(=floor_ratio*無漁獲平衡) を下回らないか判定する。

    solver 失敗時は feasible=False, solver_success=False を返す（クラッシュしない）。
    """
    tv_cfg = cfg.get("trajectory_validation", DEFAULT_SUSTAINABILITY["trajectory_validation"])
    burn_in = tv_cfg["burn_in_years"]
    eval_years = tv_cfg["evaluation_years"]
    dt = tv_cfg["evaluation_dt"]
    floor_ratio = tv_cfg["floor_ratio"]
    t_end = burn_in + eval_years

    means = np.asarray(means, dtype=float)
    sim = simulate_constant_f(params_norm, means, X0_norm, f_vec, t_end, dt)

    if not sim["success"]:
        return {
            "feasible": False, "solver_success": False,
            "min_biomass": np.full(4, np.nan), "mean_biomass": np.full(4, np.nan),
            "argmin_time": np.full(4, np.nan), "avg_yield": float("nan"),
            "first_violating_species": None,
            "any_negative": sim["any_negative"], "any_nonfinite": sim["any_nonfinite"],
            "floor": np.full(4, np.nan), "reference_abs": np.full(4, np.nan),
            "reference_kind": "unavailable",
            "reason": "solver_failure",
            "warnings": ["ODE integration failed for trajectory_floor evaluation."],
        }

    t = sim["t"]
    traj_abs = sim["traj_abs"]
    eval_mask = t >= burn_in
    if not np.any(eval_mask):
        eval_mask = np.ones_like(t, dtype=bool)
    traj_eval = traj_abs[:, eval_mask]
    t_eval_win = t[eval_mask]

    min_biomass = traj_eval.min(axis=1)
    mean_biomass = traj_eval.mean(axis=1)
    argmin_time = t_eval_win[traj_eval.argmin(axis=1)]

    warnings = []
    reference_abs, reference_kind = _unfished_reference(
        params_norm, means, X0_norm, cfg, t_end, dt, burn_in, warnings)
    floor = floor_ratio * reference_abs

    f_arr = np.asarray(f_vec, dtype=float)
    avg_yield = float(np.mean(np.sum(f_arr[:, np.newaxis] * traj_eval, axis=0)))

    if np.any(np.isnan(floor)):
        return {
            "feasible": False, "solver_success": True,
            "min_biomass": min_biomass, "mean_biomass": mean_biomass,
            "argmin_time": argmin_time, "avg_yield": avg_yield,
            "first_violating_species": None,
            "any_negative": sim["any_negative"], "any_nonfinite": sim["any_nonfinite"],
            "floor": floor, "reference_abs": reference_abs, "reference_kind": reference_kind,
            "reason": "no_reference_available",
            "warnings": warnings + ["Floor reference unavailable; treating as infeasible."],
        }

    violations = min_biomass < floor
    feasible = bool(not np.any(violations))
    first_violating_species = KEYS[int(np.argmax(violations))] if np.any(violations) else None

    return {
        "feasible": feasible, "solver_success": True,
        "min_biomass": min_biomass, "mean_biomass": mean_biomass,
        "argmin_time": argmin_time, "avg_yield": avg_yield,
        "first_violating_species": first_violating_species,
        "any_negative": sim["any_negative"], "any_nonfinite": sim["any_nonfinite"],
        "floor": floor, "reference_abs": reference_abs, "reference_kind": reference_kind,
        "reason": "ok" if feasible else "floor_violation",
        "warnings": warnings,
    }


def evaluate_time_average_lrp(params_norm, means, X0_norm, f_vec, cfg):
    """評価窓の平均資源量が average_lrp_ratio*基準値 以上かを判定する（必要なら最小値も追加要求）。

    基準値（reference）は 無漁獲平衡 -> 無漁獲長期シミュ平均 -> cfg の明示値 reference_B
    の優先順で決める。全て得られなければ feasible=False, reference種別="unavailable"。
    """
    ta_cfg = cfg.get("time_average", DEFAULT_SUSTAINABILITY["time_average"])
    tv_cfg = cfg.get("trajectory_validation", DEFAULT_SUSTAINABILITY["trajectory_validation"])
    average_ratio = ta_cfg["average_lrp_ratio"]
    min_ratio = ta_cfg["minimum_floor_ratio"]
    require_min = ta_cfg["require_minimum_floor"]
    burn_in = tv_cfg["burn_in_years"]
    eval_years = tv_cfg["evaluation_years"]
    dt = tv_cfg["evaluation_dt"]
    t_end = burn_in + eval_years

    means = np.asarray(means, dtype=float)
    sim = simulate_constant_f(params_norm, means, X0_norm, f_vec, t_end, dt)

    if not sim["success"]:
        return {
            "feasible": False, "solver_success": False,
            "mean_biomass": np.full(4, np.nan), "min_biomass": np.full(4, np.nan),
            "reference_abs": np.full(4, np.nan), "reference_kind": "unavailable",
            "reason": "solver_failure",
            "warnings": ["ODE integration failed for time_average_lrp evaluation."],
        }

    t = sim["t"]
    traj_abs = sim["traj_abs"]
    eval_mask = t >= burn_in
    if not np.any(eval_mask):
        eval_mask = np.ones_like(t, dtype=bool)
    traj_eval = traj_abs[:, eval_mask]
    mean_biomass = traj_eval.mean(axis=1)
    min_biomass = traj_eval.min(axis=1)

    warnings = []
    reference_abs, reference_kind = _unfished_reference(
        params_norm, means, X0_norm, cfg, t_end, dt, burn_in, warnings)

    if reference_kind == "unavailable":
        explicit = ta_cfg.get("reference_B")
        if explicit is not None:
            reference_abs = resolve_per_species(explicit)
            reference_kind = "explicit_reference_B"
            warnings.append("Using explicit reference_B from config (equilibrium and "
                             "long-run simulation both unavailable).")
        else:
            return {
                "feasible": False, "solver_success": True,
                "mean_biomass": mean_biomass, "min_biomass": min_biomass,
                "reference_abs": reference_abs, "reference_kind": reference_kind,
                "reason": "no_reference_available", "warnings": warnings,
            }

    average_ok = bool(np.all(mean_biomass >= average_ratio * reference_abs))
    feasible = average_ok
    reason = "ok"
    if not average_ok:
        reason = "time_average_below_threshold"
    if require_min:
        min_ok = bool(np.all(min_biomass >= min_ratio * reference_abs))
        feasible = average_ok and min_ok
        if average_ok and not min_ok:
            reason = "minimum_floor_violation"

    return {
        "feasible": feasible, "solver_success": True,
        "mean_biomass": mean_biomass, "min_biomass": min_biomass,
        "reference_abs": reference_abs, "reference_kind": reference_kind,
        "average_ratio_actual": mean_biomass / reference_abs,
        "reason": reason, "warnings": warnings,
    }


def evaluate_legacy(traj_abs, cfg):
    """msy_core.check_sustainability に完全委譲する（判定・収量は現行run_msy.pyと完全同一）。

    位相依存性についての注意を warnings に付す。
    """
    legacy_cfg = cfg.get("legacy", DEFAULT_SUSTAINABILITY["legacy"])
    result = dict(msy_core.check_sustainability(traj_abs, **legacy_cfg))
    result["warnings"] = [
        "legacy_path uses initial biomass as the reference and may be "
        "phase-dependent for oscillatory ODE trajectories."
    ]
    return result


# =============================================================================
# 境界診断・分類
# =============================================================================

def boundary_diagnostics(f_opt, f_lower, f_upper, bound_tol):
    """各種の漁獲率が下限・上限に「一致」しているかを判定する。

    at_upper_i = |f_opt_i - f_upper_i| <= bound_tol * max(1, |f_upper_i|)（at_lower も同様）。

    f_lower_i == f_upper_i（可動域の無い縮退軸。例: その種の漁獲率を 0 に固定して
    他種だけを走査するケース）は、常にその一点に「一致」してしまい at_upper/at_lower が
    無意味に True になるため、縮退軸は at_upper/at_lower とも常に False として扱う
    （「境界に張り付いている」という判定は、動ける範囲があって初めて意味を持つため）。
    """
    f_opt = resolve_per_species(f_opt)
    f_lower = resolve_per_species(f_lower)
    f_upper = resolve_per_species(f_upper)

    degenerate = np.abs(f_upper - f_lower) <= bound_tol * np.maximum(1.0, np.abs(f_upper))

    dist_to_upper = f_upper - f_opt
    dist_to_lower = f_opt - f_lower

    at_upper = (np.abs(f_opt - f_upper) <= bound_tol * np.maximum(1.0, np.abs(f_upper))) & ~degenerate
    at_lower = (np.abs(f_opt - f_lower) <= bound_tol * np.maximum(1.0, np.abs(f_lower))) & ~degenerate

    return {
        "at_upper_bound_by_species": at_upper,
        "at_lower_bound_by_species": at_lower,
        "dist_to_upper": dist_to_upper,
        "dist_to_lower": dist_to_lower,
        "any_upper_bound_active": bool(np.any(at_upper)),
        "any_lower_bound_active": bool(np.any(at_lower)),
    }


def classify_solution(feasible, boundary, lrp_active, floor_active, solver_ok):
    """解を interior|fishing_upper_bound|biomass_lrp_boundary|trajectory_floor_boundary|
    multiple_boundaries|infeasible|solver_failure に分類する。

    boundary は boundary_diagnostics() の返り dict、または any_upper_bound_active 相当の
    bool のどちらでも受け付ける。複数の境界が同時にアクティブなら multiple_boundaries。
    """
    if not solver_ok:
        return "solver_failure"
    if not feasible:
        return "infeasible"

    if isinstance(boundary, dict):
        at_upper = bool(boundary.get("any_upper_bound_active", False))
    else:
        at_upper = bool(boundary)

    lrp_active = bool(lrp_active)
    floor_active = bool(floor_active)
    n_active = int(at_upper) + int(lrp_active) + int(floor_active)

    if n_active >= 2:
        return "multiple_boundaries"
    if at_upper:
        return "fishing_upper_bound"
    if lrp_active:
        return "biomass_lrp_boundary"
    if floor_active:
        return "trajectory_floor_boundary"
    return "interior"


def _interpret_solution(classification):
    """classify_solution() の結果を、MSY用語の断定を避けた解釈ラベルへ変換する。"""
    is_interior = classification == "interior"
    is_bound_limited = classification in ("fishing_upper_bound", "multiple_boundaries")

    labels = {
        "interior": "MSY (interior optimum)",
        "fishing_upper_bound":
            "LRP-constrained maximum yield (upper-bound-limited, not a true interior MSY)",
        "biomass_lrp_boundary": "LRP-constrained maximum yield (biomass-LRP-limited)",
        "trajectory_floor_boundary": "LRP-constrained maximum yield (trajectory-floor-limited)",
        "multiple_boundaries": "LRP-constrained maximum yield (multiple boundaries active)",
        "infeasible": "infeasible: no candidate satisfies the sustainability constraint",
        "solver_failure": "undefined: ODE solver failed for all candidates",
    }
    return {
        "msy_interpretation": labels.get(classification, "LRP-constrained maximum yield"),
        "is_interior_optimum": is_interior,
        "is_bound_limited": is_bound_limited,
        "limiting_constraint": classification,
    }


# =============================================================================
# グリッド探索（一般化）
# =============================================================================

def _evaluate_candidate(params_norm, means, X0_norm, f_vec, feasibility_mode, cfg, T):
    """1候補 f_vec を feasibility_mode に応じて評価する。grid_search_general の内部ヘルパ。

    Returns (yield_ok(bool), yield_val(float), feasible(bool), biomass(4,), ratio(4,), reason(str))
    yield_ok=False の候補は「NaN/Inf/solver失敗」を意味し、呼び出し側で必ず
    infeasible 扱いにする（高収量誤採用の防止）。

    reason(str) は「なぜ feasible/infeasible になったか」の識別子。呼び出し側
    （grid_search_general）が「有効候補ゼロ」時の分類で、真の積分失敗（solver_failure）と
    構造的 infeasible（例: 平衡が非正で yield が定義できない）を区別するために使う。
    equilibrium 系は ODE を積分しない（線形平衡ソルバ）ので、その reason に "solver_failure"
    は現れない。legacy 系は ODE 積分するので、積分破綻時に "solver_failure" を返す。

    ratio(4,) は「各種の資源量 / その種の実効しきい値」で統一する
    （ratio_i>=1 <=> その種は閾値を満たす、が全モード共通で成り立つようにする）。
    near_optimal_safe の margin_i = ratio_i - 1（spec の margin_i=B_i/B_lrp_i-1 に対応）
    が feasibility_mode によらず同じ式で意味を持つようにするための設計。
    equilibrium_lrp/evaluate_equilibrium_lrp 自身が返す "biomass_ratio"
    （Bfeq/B0eq, 無漁獲平衡比）とは意味が異なる点に注意（そちらは spec 通り温存）。
    """
    biomass = np.full(4, np.nan)
    ratio = np.full(4, np.nan)

    if feasibility_mode == "equilibrium_lrp":
        res = evaluate_equilibrium_lrp(params_norm, means, f_vec, cfg)
        yv = res["total_yield"]
        yield_ok = np.isfinite(yv)
        lrp_ratio = resolve_per_species(cfg.get("lrp_ratio", DEFAULT_SUSTAINABILITY["lrp_ratio"]))
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = res["biomass_ratio"] / lrp_ratio
        return yield_ok, (float(yv) if yield_ok else float("nan")), res["feasible"], \
            res["B_eq_fished_abs"], ratio, res["reason"]

    if feasibility_mode == "trajectory_floor":
        res = evaluate_trajectory_floor(params_norm, means, X0_norm, f_vec, cfg)
        yv = res["avg_yield"]
        yield_ok = bool(res["solver_success"]) and np.isfinite(yv)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = res["min_biomass"] / res["floor"]
        return yield_ok, (float(yv) if yield_ok else float("nan")), res["feasible"], \
            res["mean_biomass"], ratio, res["reason"]

    if feasibility_mode == "time_average_lrp":
        res = evaluate_time_average_lrp(params_norm, means, X0_norm, f_vec, cfg)
        f_arr = np.asarray(f_vec, dtype=float)
        mb = res["mean_biomass"]
        yield_ok = bool(res["solver_success"]) and np.all(np.isfinite(mb))
        yv = float(np.sum(f_arr * mb)) if yield_ok else float("nan")
        ta_cfg = cfg.get("time_average", DEFAULT_SUSTAINABILITY["time_average"])
        average_ratio = ta_cfg["average_lrp_ratio"]
        ref = res["reference_abs"]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = mb / (average_ratio * ref)
        return yield_ok, yv, res["feasible"], mb, ratio, res["reason"]

    if feasibility_mode == "legacy_path":
        if T is None:
            raise ValueError("grid_search_general: T is required when feasibility_mode='legacy_path'.")
        ay = msy_core.average_yield(f_vec, params_norm, means, T, X0_norm)
        yield_ok = bool(ay["success"]) and np.isfinite(ay["mean_yield"])
        leg = evaluate_legacy(ay["traj_abs"], cfg)
        if ay["traj_abs"] is not None:
            biomass = ay["traj_abs"][:, -1]
            legacy_cfg = cfg.get("legacy", DEFAULT_SUSTAINABILITY["legacy"])
            tol = legacy_cfg.get("tol", 0.1)
            b_check = leg.get("B_check")
            b0 = leg.get("B0")
            if b_check is not None and b0 is not None:
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = b_check / (b0 * (1.0 - tol))
        if not yield_ok:
            reason = "solver_failure"           # ODE 積分が破綻（average_yield が失敗）
        elif leg["feasible"]:
            reason = "ok"
        else:
            reason = "legacy_infeasible"
        return yield_ok, (float(ay["mean_yield"]) if yield_ok else float("nan")), \
            leg["feasible"], biomass, ratio, reason

    raise ValueError(f"grid_search_general: unknown feasibility_mode {feasibility_mode!r}")


def grid_search_general(params_norm, means, X0_norm, f_upper, feasibility_mode, cfg,
                         n_grid, T=None):
    """各 f_i in linspace(F_MIN, f_upper_i, n_grid) の直積グリッドを全列挙し、
    feasibility_mode に応じた収量・持続性を評価する。

    NaN/Inf/solver失敗の候補は絶対に「高収量feasible」として採用しない
    （all_feasible は yield が有効な候補でのみ True になり得る）。

    Returns
    -------
    dict: constrained_maximum(dict), all_f(n,4), all_yield(n,), all_feasible(n,bool),
          all_biomass(n,4), all_ratio(n,4), n_feasible(int), n_evaluated(int)
    """
    means = np.asarray(means, dtype=float)
    X0_norm = np.asarray(X0_norm, dtype=float)
    f_upper_vec = resolve_per_species(f_upper)

    if feasibility_mode == "legacy_path" and T is None:
        raise ValueError("grid_search_general: T is required when feasibility_mode='legacy_path'.")

    axes = [np.linspace(msy_core.F_MIN, f_upper_vec[i], n_grid) for i in range(4)]
    mesh = np.meshgrid(*axes, indexing="ij")
    all_f = np.stack(mesh, axis=-1).reshape(-1, 4)
    n_total = all_f.shape[0]

    all_yield = np.full(n_total, np.nan)
    all_feasible = np.zeros(n_total, dtype=bool)
    all_biomass = np.full((n_total, 4), np.nan)
    all_ratio = np.full((n_total, 4), np.nan)
    all_reason = [""] * n_total   # 各候補の feasible/infeasible 理由（分類の代表理由に使う）
    n_success = 0

    for idx in range(n_total):
        f_vec = all_f[idx]
        yield_ok, yield_val, feasible, biomass, ratio, reason = _evaluate_candidate(
            params_norm, means, X0_norm, f_vec, feasibility_mode, cfg, T)

        if yield_ok:
            all_yield[idx] = yield_val
            n_success += 1
        # NaN/Inf/solver失敗の候補は絶対に feasible=True にしない
        all_feasible[idx] = bool(feasible) and yield_ok
        all_biomass[idx] = biomass
        all_ratio[idx] = ratio
        all_reason[idx] = reason

    valid = all_feasible & np.isfinite(all_yield)
    if np.any(valid):
        masked_yield = np.where(valid, all_yield, -np.inf)
        best_idx = int(np.argmax(masked_yield))
        constrained_maximum = {
            "f_opt": all_f[best_idx].copy(), "yield": float(all_yield[best_idx]),
            "biomass": all_biomass[best_idx].copy(), "ratio": all_ratio[best_idx].copy(),
            "index": best_idx,
        }
    else:
        constrained_maximum = {
            "f_opt": np.full(4, np.nan), "yield": float("nan"),
            "biomass": np.full(4, np.nan), "ratio": np.full(4, np.nan),
            "index": -1,
        }

    # --- constrained_maximum の境界分類（MSY用語を断定しないための注釈） ---
    if constrained_maximum["index"] >= 0:
        idx = constrained_maximum["index"]
        boundary = boundary_diagnostics(all_f[idx], np.zeros(4), f_upper_vec, _ACTIVE_BOUND_TOL)

        # 「資源量制約がアクティブ」= このグリッド軸沿いに一歩でも漁獲を増やすと
        # (かつ既に上限でない場合) 隣接候補が infeasible に転じる、という離散的な
        # binding-constraint の定義。LRP/floor/legacy の区別を問わず同じロジックで
        # 検出できるので、モード別のしきい値ヒアリスティックを避けられる。
        idx_multi = np.unravel_index(idx, (n_grid,) * 4)
        feasible_grid = all_feasible.reshape((n_grid,) * 4)
        biomass_active = False
        for ax in range(4):
            if idx_multi[ax] < n_grid - 1:
                nb = list(idx_multi)
                nb[ax] += 1
                if not feasible_grid[tuple(nb)]:
                    biomass_active = True
                    break

        if feasibility_mode in ("equilibrium_lrp", "time_average_lrp"):
            lrp_active, floor_active = biomass_active, False
        elif feasibility_mode in ("trajectory_floor", "legacy_path"):
            lrp_active, floor_active = False, biomass_active
        else:
            lrp_active, floor_active = False, False

        classification = classify_solution(True, boundary, lrp_active, floor_active, True)
        constrained_maximum["boundary"] = boundary
        constrained_maximum["classification"] = classification
        constrained_maximum["reason"] = all_reason[idx]   # 採用候補の理由（feasibleなら "ok"）
        constrained_maximum.update(_interpret_solution(classification))
    else:
        # 有効候補ゼロ。真の積分失敗（solver_failure）と、構造的に yield が定義できない
        # ケース（infeasible）を区別する:
        #   - equilibrium 系モードは ODE を積分しない（線形平衡ソルバ）。yield 未定義は
        #     非正/非可解平衡が原因なので solver_failure ではなく infeasible。
        #   - trajectory 系（trajectory_floor/time_average_lrp/legacy_path）は ODE 積分を
        #     行うため、n_success==0（全候補で積分が破綻）は真の solver_failure。一部でも
        #     積分成功していれば（n_success>0）「制約を満たす候補が無い」= infeasible。
        # 代表理由: 有効 yield のある候補があればその最良 yield の理由、無ければ先頭候補の理由。
        finite_mask = np.isfinite(all_yield)
        if np.any(finite_mask):
            rep_idx = int(np.argmax(np.where(finite_mask, all_yield, -np.inf)))
        else:
            rep_idx = 0
        failure_reason = all_reason[rep_idx] if all_reason else "no_candidate"

        if feasibility_mode == "equilibrium_lrp":
            classification = "infeasible"     # equilibrium 系は solver_failure になり得ない
        else:
            classification = "solver_failure" if n_success == 0 else "infeasible"
        constrained_maximum["boundary"] = None
        constrained_maximum["classification"] = classification
        constrained_maximum["reason"] = failure_reason
        constrained_maximum.update(_interpret_solution(classification))

    return {
        "constrained_maximum": constrained_maximum,
        "all_f": all_f, "all_yield": all_yield, "all_feasible": all_feasible,
        "all_biomass": all_biomass, "all_ratio": all_ratio,
        "n_feasible": int(np.sum(all_feasible)), "n_evaluated": n_total,
        "n_success": n_success, "feasibility_mode": feasibility_mode,
        "f_upper": f_upper_vec,
    }


# =============================================================================
# 最適化まわり: 95%安全解・境界到達点
# =============================================================================

def near_optimal_safe(grid_result, Y_max, cfg):
    """feasible かつ Y >= near_optimal_fraction*Y_max の候補から safe_solution_criterion で選ぶ。

    margin_i = B_i/B_lrp_i - 1 を計算する。grid_search_general（_evaluate_candidate）が
    all_ratio を「各種の資源量 / その種の実効しきい値」（>=1 なら閾値クリア）として
    統一的に構成しているため、margin_i = all_ratio[:, i] - 1 がそのまま
    spec の margin_i=B_i/B_lrp_i-1 に一致する（モードによらず同じ式でよい）。
    """
    frac = cfg.get("near_optimal_fraction", DEFAULT_OPTIMIZATION["near_optimal_fraction"])
    criterion = cfg.get("safe_solution_criterion", DEFAULT_OPTIMIZATION["safe_solution_criterion"])
    bound_tol = cfg.get("bound_tol", DEFAULT_OPTIMIZATION["bound_tol"])

    all_f = grid_result["all_f"]
    all_yield = grid_result["all_yield"]
    all_feasible = grid_result["all_feasible"]
    all_biomass = grid_result["all_biomass"]
    all_ratio = grid_result["all_ratio"]

    if not np.isfinite(Y_max) or Y_max <= 0:
        candidate_mask = all_feasible & np.isfinite(all_yield)
    else:
        candidate_mask = all_feasible & np.isfinite(all_yield) & (all_yield >= frac * Y_max)

    if not np.any(candidate_mask):
        return {
            "found": False, "f_safe": np.full(4, np.nan), "yield_safe": float("nan"),
            "per_species_yield": np.full(4, np.nan), "biomass": np.full(4, np.nan),
            "ratio": np.full(4, np.nan), "margin_per_species": np.full(4, np.nan),
            "safety_margin": float("nan"), "active_constraint": None,
            "at_upper_bound_by_species": np.zeros(4, dtype=bool),
            "any_upper_bound_active": False,
            "criterion_used": criterion,
            "warnings": ["No feasible candidate reached near_optimal_fraction of Y_max."],
        }

    idxs = np.where(candidate_mask)[0]
    margins_matrix = all_ratio[idxs] - 1.0

    if criterion == "min_total_fishing":
        totals = np.sum(all_f[idxs], axis=1)
        chosen = idxs[int(np.argmin(totals))]
    elif criterion == "max_total_biomass_margin":
        totals = np.sum(margins_matrix, axis=1)
        chosen = idxs[int(np.argmax(totals))]
    else:  # "max_min_biomass_margin"（既定）
        mins = np.min(margins_matrix, axis=1)
        chosen = idxs[int(np.argmax(mins))]

    f_safe = all_f[chosen].copy()
    yield_safe = float(all_yield[chosen])
    biomass = all_biomass[chosen].copy()
    ratio = all_ratio[chosen].copy()
    per_species_yield = f_safe * biomass
    margin_per_species = ratio - 1.0
    safety_margin = float(np.min(margin_per_species))
    active_constraint = KEYS[int(np.argmin(margin_per_species))]

    # f_upper が無い grid_result（想定外の呼び出し）では NaN にフォールバックする。
    # f_safe 自身をデフォルトにすると常に「上限に一致」と誤判定してしまうため NaN が安全
    # （NaN 比較は常に False なので at_upper_bound は誤って True にならず False になる）。
    boundary = boundary_diagnostics(
        f_safe, np.zeros(4), grid_result.get("f_upper", np.full(4, np.nan)), bound_tol)

    return {
        "found": True, "f_safe": f_safe, "yield_safe": yield_safe,
        "per_species_yield": per_species_yield, "biomass": biomass, "ratio": ratio,
        "margin_per_species": margin_per_species, "safety_margin": safety_margin,
        "active_constraint": active_constraint,
        "at_upper_bound_by_species": boundary["at_upper_bound_by_species"],
        "any_upper_bound_active": boundary["any_upper_bound_active"],
        "criterion_used": criterion, "warnings": [],
    }


def find_limit_point(grid_result, cfg):
    """f を0から増やしていったときに最初にLRP（feasible→infeasible遷移）または
    f_upper（張り付き）に到達する境界点を近似的に探す。

    grid_search_general が返す4次元グリッドは "f=0からの経路" ではないため、
    各候補を intensity=mean(f_i/f_upper_i)（0〜1、上限に対する漁獲強度の割合）で
    順序付けし、その順に沿って feasible→infeasible の遷移を探す近似手法を取る
    （grid_search_general 自身の docstring で言及の通り「近似」）。
    """
    all_f = grid_result["all_f"]
    all_feasible = grid_result["all_feasible"]
    all_yield = grid_result["all_yield"]
    f_upper = np.asarray(grid_result.get("f_upper", np.nanmax(all_f, axis=0)), dtype=float)
    bound_tol = cfg.get("bound_tol", DEFAULT_OPTIMIZATION["bound_tol"]) if isinstance(cfg, dict) \
        else DEFAULT_OPTIMIZATION["bound_tol"]

    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.divide(all_f, f_upper, out=np.zeros_like(all_f), where=f_upper > 0)
    intensity = np.mean(frac, axis=1)
    order = np.argsort(intensity)

    limit_idx = None
    last_feasible_idx = None
    for idx in order:
        if all_feasible[idx]:
            last_feasible_idx = idx
        elif last_feasible_idx is not None:
            limit_idx = last_feasible_idx
            break

    if limit_idx is None:
        if last_feasible_idx is None:
            return {"found": False, "f_limit": np.full(4, np.nan), "yield_limit": float("nan"),
                    "limiting_constraint": "infeasible_everywhere", "index": -1}
        limit_idx = last_feasible_idx  # 常に feasible = 上限まで到達しても infeasible にならない

    boundary = boundary_diagnostics(all_f[limit_idx], np.zeros(4), f_upper, bound_tol)
    limiting_constraint = "fishing_upper_bound" if boundary["any_upper_bound_active"] \
        else "biomass_lrp_boundary"

    yv = all_yield[int(limit_idx)]
    return {
        "found": True, "f_limit": all_f[int(limit_idx)].copy(),
        "yield_limit": float(yv) if np.isfinite(yv) else float("nan"),
        "limiting_constraint": limiting_constraint, "index": int(limit_idx),
    }


# =============================================================================
# 感度分析
# =============================================================================

def lrp_sensitivity(params_norm, means, X0_norm, cfg, lrp_ratios, f_upper, n_grid, T=None):
    """lrp_ratio を lrp_ratios で振り、各値について grid_search_general + near_optimal_safe
    （＋可能ならequilibrium最適点でのtrajectory_floorクロスチェック）を実行する。

    T : float or None
        積分期間（年）。legacy_path モード（および内部で軌道積分する任意のモード）で
        grid_search_general に必須。equilibrium_lrp は X0/時間非依存なので T=None のまま可。
        シグネチャ末尾に置くことで、既存のキーワード呼び出しを壊さない。

    Returns
    -------
    list[dict]: 1要素/lrp_ratio。将来のドライバがこれを平坦化してCSV行(sensitivity_to_csv)を作る。
    """
    mode = cfg.get("mode", DEFAULT_SUSTAINABILITY["mode"])
    rows = []
    for ratio in lrp_ratios:
        sub_cfg = copy.deepcopy(cfg)
        sub_cfg["lrp_ratio"] = ratio
        grid = grid_search_general(params_norm, means, X0_norm, f_upper, mode, sub_cfg, n_grid, T=T)

        finite_yield = grid["all_yield"][np.isfinite(grid["all_yield"])]
        y_max = float(np.max(finite_yield)) if finite_yield.size else float("nan")
        safe = near_optimal_safe(grid, y_max, DEFAULT_OPTIMIZATION)
        cm = grid["constrained_maximum"]

        traj_check = None
        tv_cfg = sub_cfg.get("trajectory_validation", DEFAULT_SUSTAINABILITY["trajectory_validation"])
        if mode == "equilibrium_lrp" and tv_cfg.get("enabled", True) and np.all(np.isfinite(cm["f_opt"])):
            traj_check = evaluate_trajectory_floor(params_norm, means, X0_norm, cm["f_opt"], sub_cfg)

        rows.append({
            "lrp_ratio": ratio, "sustainability_mode": mode,
            "n_feasible": grid["n_feasible"], "n_evaluated": grid["n_evaluated"],
            "y_max": y_max, "constrained_maximum": cm, "safe_solution": safe,
            "trajectory_check": traj_check,
        })
    return rows


def _diagnose_upper_bound_pattern(yields, at_uppers):
    """upper_bound_sensitivity の集約診断（近似ヒューリスティック）。

    - 収量が単調非減少で、かつ全点で at_upper -> "upper-bound-driven"
      （上限を上げ続ければ収量も上がり続ける = 上限そのものが効いている）
    - 直近の点で at_upper が外れている（内部最適に収束） -> "internally-determined"
    - それ以外（上限に張り付かないまま収量が頭打ち） -> "lrp-limited"
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


def upper_bound_sensitivity(params_norm, means, X0_norm, cfg, f_upper_grid, lrp_ratio, n_grid, T=None):
    """fishing_upper_bound を f_upper_grid で振り、各値について grid_search_general で最適化する。

    T : float or None
        積分期間（年）。legacy_path モード（および内部で軌道積分する任意のモード）で
        grid_search_general に必須。equilibrium_lrp は X0/時間非依存なので T=None のまま可。
        シグネチャ末尾に置くことで、既存のキーワード呼び出しを壊さない。

    Returns
    -------
    list[dict]: 1要素/f_upper。各要素に集約診断 "diagnosis" を付す
    （upper-bound-driven|internally-determined|lrp-limited|insufficient_data）。
    """
    mode = cfg.get("mode", DEFAULT_SUSTAINABILITY["mode"])
    bound_tol = DEFAULT_OPTIMIZATION["bound_tol"]
    rows = []
    for f_upper in f_upper_grid:
        sub_cfg = copy.deepcopy(cfg)
        sub_cfg["lrp_ratio"] = lrp_ratio
        grid = grid_search_general(params_norm, means, X0_norm, f_upper, mode, sub_cfg, n_grid, T=T)
        cm = grid["constrained_maximum"]

        if np.all(np.isfinite(cm["f_opt"])):
            boundary = boundary_diagnostics(cm["f_opt"], np.zeros(4),
                                             resolve_per_species(f_upper), bound_tol)
            at_upper = bool(boundary["any_upper_bound_active"])
        else:
            at_upper = False

        rows.append({
            "fishing_upper_bound": f_upper, "sustainability_mode": mode,
            "yield_opt": cm["yield"], "f_opt": cm["f_opt"], "at_upper_bound": at_upper,
            "n_feasible": grid["n_feasible"],
        })

    diagnosis = _diagnose_upper_bound_pattern(
        [r["yield_opt"] for r in rows], [r["at_upper_bound"] for r in rows])
    for r in rows:
        r["diagnosis"] = diagnosis
    return rows


def sensitivity_to_csv(rows, path):
    """rows（CSV_COLUMNS 形状の平坦な dict のリスト）を csv 標準ライブラリで書き出す。

    rows の各 dict に無いキーは空文字列で埋め、CSV_COLUMNS に無い余分なキーは無視する
    （extrasaction="ignore"）。将来のドライバが lrp_sensitivity/upper_bound_sensitivity の
    ネストした結果を種別に平坦化してから渡す想定。
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = _csv.DictWriter(f, fieldnames=CSV_COLUMNS, restval="", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path
