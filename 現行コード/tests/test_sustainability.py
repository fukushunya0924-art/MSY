"""
sustainability.py の往復テスト（pytest不使用の自己検証スクリプト）。

pytest がこの環境に入っていないため、直接実行できる形にしてある:
    python3 現行コード/tests/test_sustainability.py

test_model_constrained.py と同じ流儀: 各 test_* 関数が assert で検証し、
main() が結果を集計して sys.exit(0/1) する。

検証内容（spec section 2(C) の9群）:
  1. legacy互換        : evaluate_legacy が msy_core.check_sustainability 直呼びと完全一致。
  2. 位相依存           : legacy(path)は初期位相で判定が変わり得るが、equilibrium_lrp は
                         X0非依存なので同一params・同一fなら判定が変わらない。
  3. 平衡（解析解）      : 2種古典LV( dx=x(r-a y), dy=y(-m+b x) )の解析解と
                         equilibrium_generalized_lv の数値解が一致。
  4. 非正平衡           : fにより平衡成分が非正になる例で equilibrium_lrp が infeasible を返す。
  5. 上限張り付き        : 収量がfに単調な小例で f_opt=上限、classification="fishing_upper_bound"。
  6. LRP境界            : 上限より先にLRPが効く例で f_opt<上限、classification="biomass_lrp_boundary"。
  7. 95%安全側          : near_optimal_safe が最大収量解より安全余裕の大きい別解を選ぶ。
  8. 時間平均収量        : mean_t(sum f_i B_i(t)) == sum_i f_i*mean_t(B_i(t))（一定f）。
  9. solver failure     : NaN/Inf/積分失敗の候補が高収量として採用されない。
"""
import copy
import os
import sys
import warnings

import numpy as np

# 現行コード/ と 現行コード/msy/ を sys.path に追加（model.py と msy/sustainability.py 等を import するため）
_here = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_here)
_msy_dir = os.path.join(_parent, "msy")
sys.path.insert(0, _msy_dir)
sys.path.append(_parent)

import model                    # noqa: E402
import msy_core                 # noqa: E402
import sustainability as sus    # noqa: E402

_NAMES = model.MODELS["capacity_ry"]["names"]


def _params(**kwargs):
    """capacity_ry の12変数 dict を names 順の ndarray に変換する。"""
    return np.array([kwargs[n] for n in _NAMES], dtype=float)


# =============================================================================
# 1. legacy互換
# =============================================================================

def test_legacy_compatibility():
    """evaluate_legacy(traj_abs, cfg) が msy_core.check_sustainability 直呼びと完全一致すること。"""
    t = np.linspace(0.0, 5.0, 50)
    traj_abs = np.vstack([
        100.0 - 4.0 * t,          # x1: 100 -> 80 (20%減, tol=0.1 なら違反)
        50.0 + 2.0 * np.sin(t),   # x2: 50 前後で振動
        10.0 + 0.0 * t,           # y1: 一定
        5.0 + 0.2 * t,            # y2: 5 -> 6 (増加)
    ])

    configs = [
        {"scope": "all", "mode": "endpoint", "tol": 0.1},
        {"scope": "prey", "mode": "path", "tol": 0.2},
        {"scope": "predator", "mode": "path", "tol": 0.05},
    ]

    for legacy_cfg in configs:
        direct = msy_core.check_sustainability(traj_abs, **legacy_cfg)
        cfg = copy.deepcopy(sus.DEFAULT_SUSTAINABILITY)
        cfg["legacy"] = legacy_cfg
        wrapped = sus.evaluate_legacy(traj_abs, cfg)

        assert wrapped["feasible"] == direct["feasible"], \
            f"feasible不一致 cfg={legacy_cfg}: wrapped={wrapped['feasible']} direct={direct['feasible']}"
        assert np.allclose(wrapped["margins"], direct["margins"], equal_nan=True), \
            f"margins不一致 cfg={legacy_cfg}"
        assert np.array_equal(wrapped["B_check"], direct["B_check"]), \
            f"B_check不一致 cfg={legacy_cfg}"
        assert np.array_equal(wrapped["B0"], direct["B0"]), \
            f"B0不一致 cfg={legacy_cfg}"
        assert "warnings" in wrapped and len(wrapped["warnings"]) >= 1, \
            "evaluate_legacy は phase-dependent 注意の warnings を持つべき"

    # 積分失敗（traj_abs=None）でもクラッシュせず feasible=False を返すこと
    direct_none = msy_core.check_sustainability(None)
    wrapped_none = sus.evaluate_legacy(None, sus.DEFAULT_SUSTAINABILITY)
    assert direct_none["feasible"] is False and wrapped_none["feasible"] is False

    print("[OK] test_legacy_compatibility")


# =============================================================================
# 2. 位相依存性: legacy は位相で変わり得る／equilibrium_lrp は変わらない
# =============================================================================

def test_phase_dependence_vs_equilibrium_invariance():
    """同じ params/f で、legacy(path)判定は初期位相により変わり得るが equilibrium_lrp は不変であること。

    capacity_ry を L12=L21=0 で(x1,y1)と(x2,y2)に完全分離し、(x2,y2)は自分自身の
    固定点に置いて定数化、(x1,y1)だけを閉軌道(古典LV)にする（本体の simulate_constant_f を使用）。
    """
    p = _params(r_x1=1.0, r_x2=0.5, r_y1=0.5, r_y2=0.2,
                L11=0.5, L12=0.0, L21=0.0, L22=0.5,
                C1=1.0, D1=0.0, C2=0.0, D2=0.4)
    means = np.ones(4)
    f_vec = np.zeros(4)
    cfg = sus.DEFAULT_SUSTAINABILITY

    # (x2,y2) の固定点: y2_eq=(r_x2)/L22=1.0, x2_eq=(r_y2)/(D2*L22)=1.0
    X0 = np.array([2.0, 1.0, 2.0, 1.0])  # (x1,y1)=(2,2) は平衡(1,2)から外した閉軌道上の点

    # 長時間の基準軌道を積分し、そこから複数「位相」をサンプルする
    ref = sus.simulate_constant_f(p, means, X0, f_vec, t_end=30.0, dt=0.001)
    assert ref["success"], "参照軌道の積分に失敗した（テスト設計を見直す必要あり）"
    t = ref["t"]
    traj = ref["traj_abs"]

    period = 9.23  # 数値的に確認済みの概周期（本テストの許容誤差では厳密値は不要）
    phase_times = [0.0, period * 0.25, period * 0.5, period * 0.75]

    leg_feasibles = []
    for pt in phase_times:
        idx = int(np.argmin(np.abs(t - pt)))
        X0_phase = traj[:, idx].copy()
        sub = sus.simulate_constant_f(p, means, X0_phase, f_vec, t_end=2.0, dt=0.01)
        assert sub["success"], f"phase t={pt} からの積分に失敗"
        leg = sus.evaluate_legacy(sub["traj_abs"], cfg)
        leg_feasibles.append(leg["feasible"])

    # legacy(path) は位相依存 -> 全位相で同一判定にはならない
    assert len(set(leg_feasibles)) > 1, \
        f"legacy判定が位相によらず一定だった（位相依存性を再現できていない）: {leg_feasibles}"

    # equilibrium_lrp は X0 を受け取らないため、呼ぶたびに（同じ params/f/means/cfg なら）
    # 定義上つねに同一の結果になる。実際に複数回呼んでビット単位で一致することを確認する。
    eq_results = [sus.evaluate_equilibrium_lrp(p, means, f_vec, cfg) for _ in phase_times]
    feas0 = eq_results[0]["feasible"]
    ratio0 = eq_results[0]["biomass_ratio"]
    for r in eq_results[1:]:
        assert r["feasible"] == feas0
        assert np.array_equal(r["biomass_ratio"], ratio0)

    print(f"[OK] test_phase_dependence_vs_equilibrium_invariance "
          f"(legacy_feasibles={leg_feasibles}, equilibrium_lrp_feasible={feas0})")


# =============================================================================
# 3. 平衡の解析解（2種古典LV, equilibrium_generalized_lv を直接使用）
# =============================================================================

def test_equilibrium_analytical_two_species():
    """古典LV dx=x(r-fx-a*y), dy=y(-m-fy+b*x) の解析解と equilibrium_generalized_lv が一致すること。

    rho = [r-fx, -(m+fy)], A = [[0,-a],[b,0]] （本モジュールdocstring・spec節1の対応）。
    解析解: B0_eq=[m/b, r/a]（無漁獲）, Bf_eq=[(m+fy)/b, (r-fx)/a]（漁獲下）。
    """
    r, a, m, b = 1.0, 0.4, 0.5, 0.25
    fx, fy = 0.3, 0.1
    lrp_ratio = 0.3
    A = np.array([[0.0, -a], [b, 0.0]])

    # --- 無漁獲平衡 ---
    res0 = sus.equilibrium_generalized_lv(A, np.array([r, -m]))
    assert res0["solvable"]
    B0_expected = np.array([m / b, r / a])
    assert np.allclose(res0["B_eq"], B0_expected, rtol=1e-10), \
        f"B0_eq不一致: {res0['B_eq']} vs {B0_expected}"

    # --- 漁獲下平衡 ---
    resf = sus.equilibrium_generalized_lv(A, np.array([r - fx, -(m + fy)]))
    assert resf["solvable"]
    Bf_expected = np.array([(m + fy) / b, (r - fx) / a])
    assert np.allclose(resf["B_eq"], Bf_expected, rtol=1e-10), \
        f"Bf_eq不一致: {resf['B_eq']} vs {Bf_expected}"

    # --- LRP判定・平衡収量（手計算と一致） ---
    B0, Bf = res0["B_eq"], resf["B_eq"]
    ratio = Bf / B0
    ratio_expected = np.array([1.2, 0.7])
    assert np.allclose(ratio, ratio_expected, rtol=1e-10)

    feasible = bool(np.all(ratio >= lrp_ratio))
    assert feasible is True  # 0.7 >= 0.3 なので両種ともLRPを満たす

    yield_eq = fx * Bf[0] + fy * Bf[1]
    assert np.isclose(yield_eq, 0.895, rtol=1e-10), f"yield_eq={yield_eq}"

    # --- cond/reason の健全性 ---
    assert np.isfinite(res0["cond"]) and res0["reason"] == "ok"

    print("[OK] test_equilibrium_analytical_two_species")


# =============================================================================
# 4. 非正平衡 -> equilibrium_lrp が infeasible
# =============================================================================

def test_non_positive_equilibrium_infeasible():
    """無漁獲平衡は正だが、ある f で平衡成分が負になる例で equilibrium_lrp が infeasible を返すこと。

    分離系(L12=L21=0)で y1_eq=(r_x1-fx1)/L11 なので fx1 > r_x1 にすると y1_eq<0 になる。
    """
    p = _params(r_x1=0.2, r_x2=0.5, r_y1=0.3, r_y2=0.3,
                L11=0.4, L12=0.0, L21=0.0, L22=0.4,
                C1=0.5, D1=0.0, C2=0.0, D2=0.5)
    means = np.ones(4)
    cfg = sus.DEFAULT_SUSTAINABILITY

    unfished = sus.compute_equilibrium(p, np.zeros(4), cfg)
    assert unfished["positive"], "このテストは無漁獲平衡が正であることを前提にしている"

    f_vec = np.array([0.5, 0.0, 0.0, 0.0])  # fx1=0.5 > r_x1=0.2
    fished = sus.compute_equilibrium(p, f_vec, cfg)
    assert fished["solvable"], "この例は解ける（特異ではない）はず"
    assert fished["positive"] is False, f"fished平衡が正になってしまった: {fished['B_eq_norm']}"
    assert fished["B_eq_norm"][2] < 0, "y1成分が負になる設計のはずが違う"

    res = sus.evaluate_equilibrium_lrp(p, means, f_vec, cfg)
    assert res["feasible"] is False
    assert res["reason"] == "no_positive_fished_equilibrium"
    assert np.isnan(res["total_yield"])

    print("[OK] test_non_positive_equilibrium_infeasible")


# =============================================================================
# 5. 上限張り付き
# =============================================================================

def test_upper_bound_binding():
    """収量がf_x1に単調増加する小例で、grid_search_general の最適解が上限に張り付くこと。"""
    p = _params(r_x1=1.0, r_x2=0.5, r_y1=0.3, r_y2=0.3,
                L11=0.5, L12=0.0, L21=0.0, L22=0.4,
                C1=0.6, D1=0.0, C2=0.0, D2=0.5)
    means = np.ones(4)
    X0 = np.ones(4)

    cfg = copy.deepcopy(sus.DEFAULT_SUSTAINABILITY)
    cfg["mode"] = "equilibrium_lrp"
    cfg["lrp_ratio"] = 0.1  # 十分緩いのでLRPは効かず、上限まで単調増加する

    f_upper = [0.5, 0.0, 0.0, 0.0]  # r_x1=1.0 未満に上限を抑え、y1_eqが常に正であるようにする
    grid = sus.grid_search_general(p, means, X0, f_upper, "equilibrium_lrp", cfg, n_grid=9)

    assert grid["n_feasible"] == grid["n_evaluated"], "この例は全点feasibleになるよう設計している"

    cm = grid["constrained_maximum"]
    assert np.isclose(cm["f_opt"][0], 0.5, atol=1e-9), f"f_optが上限でない: {cm['f_opt']}"
    assert cm["boundary"]["any_upper_bound_active"] is True
    assert cm["classification"] == "fishing_upper_bound"
    assert cm["is_interior_optimum"] is False
    assert cm["is_bound_limited"] is True
    # MSY用語を断定しない解釈ラベルであること（"MSY" と断定せず、上限限界を明記）
    assert "upper-bound" in cm["msy_interpretation"] or "not a true interior MSY" in cm["msy_interpretation"]

    print(f"[OK] test_upper_bound_binding (f_opt={cm['f_opt']}, classification={cm['classification']})")


# =============================================================================
# 6. LRP境界（上限より先にLRPが効く）
# =============================================================================

def test_lrp_boundary_binding():
    """上限に達する前にLRPが効いて頭打ちになる例で classification=="biomass_lrp_boundary" になること。"""
    p = _params(r_x1=1.0, r_x2=0.5, r_y1=0.3, r_y2=0.3,
                L11=0.5, L12=0.0, L21=0.0, L22=0.4,
                C1=0.6, D1=0.0, C2=0.0, D2=0.5)
    means = np.ones(4)
    X0 = np.ones(4)

    cfg = copy.deepcopy(sus.DEFAULT_SUSTAINABILITY)
    cfg["mode"] = "equilibrium_lrp"
    cfg["lrp_ratio"] = 0.5  # y1のratio=(1-fx1/r_x1) が fx1=0.5 で lrp を割り込む

    f_upper = [0.9, 0.0, 0.0, 0.0]  # 上限をLRPが効く点より十分先に置く
    grid = sus.grid_search_general(p, means, X0, f_upper, "equilibrium_lrp", cfg, n_grid=9)

    assert 0 < grid["n_feasible"] < grid["n_evaluated"], \
        f"feasible/infeasibleが混在するはずが n_feasible={grid['n_feasible']}/{grid['n_evaluated']}"

    cm = grid["constrained_maximum"]
    assert cm["f_opt"][0] < 0.9 - 1e-9, f"f_optが上限近くまで行ってしまった: {cm['f_opt']}"
    assert cm["boundary"]["any_upper_bound_active"] is False
    assert cm["classification"] == "biomass_lrp_boundary"
    assert cm["is_interior_optimum"] is False
    assert cm["is_bound_limited"] is False
    assert cm["limiting_constraint"] == "biomass_lrp_boundary"

    print(f"[OK] test_lrp_boundary_binding (f_opt={cm['f_opt']}, n_feasible={grid['n_feasible']}/{grid['n_evaluated']})")


# =============================================================================
# 7. 95%安全側解
# =============================================================================

def test_near_optimal_safe_solution():
    """near_optimal_safe が、最大収量解より低いfで95%以上の収量かつ大きい安全余裕を持つ解を選ぶこと。

    near_optimal_safe は grid_result の配列（all_f/all_yield/all_feasible/all_biomass/all_ratio）
    だけを見て選択するので、本体の関数を直接叩く目的でここでは意図的に単純化した
    grid_result を組み立てる（収量が収穫逓減、marginがfとともに縮む設計）。
    """
    all_f = np.array([[0.1, 0, 0, 0], [0.2, 0, 0, 0], [0.3, 0, 0, 0],
                       [0.4, 0, 0, 0], [0.5, 0, 0, 0]])
    all_yield = np.array([50.0, 90.0, 100.0, 102.0, 103.0])
    all_feasible = np.ones(5, dtype=bool)
    # ratio_i = B_i/B_lrp_i（種0のみ変化、他種は常に安全一定=3）
    all_ratio = np.array([
        [3.00, 3, 3, 3],
        [2.20, 3, 3, 3],
        [1.60, 3, 3, 3],
        [1.20, 3, 3, 3],
        [1.05, 3, 3, 3],
    ])
    all_biomass = np.ones((5, 4)) * 2.0
    grid_result = {
        "all_f": all_f, "all_yield": all_yield, "all_feasible": all_feasible,
        "all_biomass": all_biomass, "all_ratio": all_ratio,
        "f_upper": np.array([0.5, 0.0, 0.0, 0.0]),
    }

    Y_max = float(np.max(all_yield))
    opt_cfg = dict(sus.DEFAULT_OPTIMIZATION)
    safe = sus.near_optimal_safe(grid_result, Y_max, opt_cfg)

    max_idx = int(np.argmax(all_yield))
    max_margin = all_ratio[max_idx, 0] - 1.0

    assert safe["found"] is True
    assert safe["yield_safe"] >= 0.95 * Y_max, \
        f"safe解が95%収量条件を満たさない: {safe['yield_safe']} < {0.95*Y_max}"
    assert safe["safety_margin"] > max_margin, \
        f"safe解の安全余裕が最大収量解以下: {safe['safety_margin']} <= {max_margin}"
    assert not np.allclose(safe["f_safe"], all_f[max_idx]), \
        "safe解が最大収量解と同じ点になってしまった（別解であるべき）"
    assert np.isclose(safe["margin_per_species"][0], safe["safety_margin"])

    print(f"[OK] test_near_optimal_safe_solution "
          f"(f_safe={safe['f_safe']}, yield_safe={safe['yield_safe']}, margin={safe['safety_margin']:.4f} "
          f"vs max_yield_margin={max_margin:.4f})")


# =============================================================================
# 8. 時間平均収量の恒等式
# =============================================================================

def test_time_average_yield_identity():
    """一定fのもとで mean_t(sum_i f_i*B_i(t)) == sum_i f_i*mean_t(B_i(t)) が rtol~1e-6 で成り立つこと。"""
    p = _params(r_x1=1.0, r_x2=0.5, r_y1=0.3, r_y2=0.3,
                L11=0.5, L12=0.0, L21=0.0, L22=0.4,
                C1=0.6, D1=0.0, C2=0.0, D2=0.5)
    means = np.array([2.0, 3.0, 4.0, 5.0])
    X0 = np.array([1.2, 0.8, 1.0, 1.0])
    f_vec = np.array([0.2, 0.05, 0.1, 0.0])

    cfg = copy.deepcopy(sus.DEFAULT_SUSTAINABILITY)
    cfg["trajectory_validation"] = {
        "enabled": True, "burn_in_years": 5, "evaluation_years": 5,
        "evaluation_dt": 0.02, "floor_ratio": 0.1,
    }

    res = sus.evaluate_trajectory_floor(p, means, X0, f_vec, cfg)
    assert res["solver_success"], "積分に失敗した（テスト設計を見直す必要あり）"

    # 同じ評価窓を simulate_constant_f から直接再構成して両辺を独立に計算する
    burn_in = cfg["trajectory_validation"]["burn_in_years"]
    eval_years = cfg["trajectory_validation"]["evaluation_years"]
    dt = cfg["trajectory_validation"]["evaluation_dt"]
    sim = sus.simulate_constant_f(p, means, X0, f_vec, burn_in + eval_years, dt)
    assert sim["success"]
    mask = sim["t"] >= burn_in
    traj_eval = sim["traj_abs"][:, mask]

    lhs = np.mean(np.sum(f_vec[:, np.newaxis] * traj_eval, axis=0))       # mean_t(sum_i f_i B_i(t))
    rhs = np.sum(f_vec * traj_eval.mean(axis=1))                          # sum_i f_i*mean_t(B_i(t))

    assert np.isclose(lhs, rhs, rtol=1e-6), f"恒等式が成り立たない: lhs={lhs} rhs={rhs}"
    assert np.isclose(res["avg_yield"], lhs, rtol=1e-6), \
        f"evaluate_trajectory_floorのavg_yieldが恒等式のlhsと不一致: {res['avg_yield']} vs {lhs}"

    print(f"[OK] test_time_average_yield_identity (lhs={lhs:.6f}, rhs={rhs:.6f}, "
          f"reldiff={abs(lhs-rhs)/abs(lhs):.2e})")


# =============================================================================
# 9. solver failure は絶対に高収量feasibleとして採用されない
# =============================================================================

def test_solver_failure_never_high_yield():
    """NaN/Inf/積分失敗の候補が高収量として採用されず、常にinfeasible扱いになること。"""
    # --- 9a: simulate_constant_f レベル: 非有限なX0は必ず失敗として扱われる ---
    p_normal = _params(r_x1=0.5, r_x2=0.5, r_y1=0.2, r_y2=0.2,
                       L11=0.1, L12=0.05, L21=0.05, L22=0.1,
                       C1=0.3, D1=0.2, C2=0.25, D2=0.15)
    means = np.ones(4)
    f_vec = np.zeros(4)

    for bad_val in (np.nan, np.inf, -np.inf):
        X0_bad = np.array([1.0, bad_val, 1.0, 1.0])
        sim = sus.simulate_constant_f(p_normal, means, X0_bad, f_vec, t_end=5.0, dt=0.1)
        assert sim["success"] is False, f"bad_val={bad_val} で success=Trueになってしまった"
        assert sim["traj_abs"] is None
        assert sim["any_nonfinite"] is True

    # --- 9b: grid_search_general レベル: 一部の候補がオーバーフローで破綻する例 ---
    # L11=L12=0 で x1 は dx1=(r_x1-fx1)*x1 の純粋な指数成長（自己抑制なし）。
    # fx1=0 は評価窓内でオーバーフローするが、fx1>=20 は十分抑制されて有限にとどまる
    # （数値的に確認済み）。y1,y2,x2 は r_y1=0/固定点初期値により定数化し、
    # このテストの本題（fx1=0のオーバーフロー）以外の理由でinfeasibleにならないようにする。
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    p_blowup = _params(r_x1=60.0, r_x2=0.5, r_y1=0.0, r_y2=0.3,
                       L11=0.0, L12=0.0, L21=0.0, L22=0.4,
                       C1=0.0, D1=0.0, C2=0.0, D2=0.5)
    X0 = np.array([10.0, 1.5, 10.0, 1.25])  # x2,y2 は自分自身の固定点(=1.5,1.25)で定数

    cfg = copy.deepcopy(sus.DEFAULT_SUSTAINABILITY)
    cfg["mode"] = "time_average_lrp"
    cfg["trajectory_validation"] = {
        "enabled": True, "burn_in_years": 5, "evaluation_years": 10,
        "evaluation_dt": 0.02, "floor_ratio": 0.1,
    }
    cfg["time_average"] = {
        "average_lrp_ratio": 0.1, "minimum_floor_ratio": 0.05,
        "require_minimum_floor": False,
        "reference_B": [10.0, 1.5, 10.0, 1.25],  # 明示的な基準値（tier3フォールバック）
    }

    grid = sus.grid_search_general(p_blowup, means, X0, [60.0, 0.0, 0.0, 0.0],
                                    "time_average_lrp", cfg, n_grid=4)

    fx1_vals = grid["all_f"][:, 0]
    zero_mask = np.isclose(fx1_vals, 0.0)
    assert np.any(zero_mask), "fx1=0の候補が含まれていない（グリッド設計を確認）"
    # fx1=0 の候補は必ず yield=NaN かつ infeasible（オーバーフローで破綻するため）
    assert np.all(np.isnan(grid["all_yield"][zero_mask])), \
        "fx1=0（オーバーフロー）の候補にNaNでないyieldが混入した"
    assert np.all(~grid["all_feasible"][zero_mask]), \
        "fx1=0（オーバーフロー）の候補がfeasibleとして扱われた"

    nonzero_mask = ~zero_mask
    assert np.any(grid["all_feasible"][nonzero_mask]), \
        "fx1>0の候補が一つもfeasibleにならなかった（テスト設計を見直す必要あり）"

    cm = grid["constrained_maximum"]
    assert np.isfinite(cm["yield"]), f"最適解の収量が非有限: {cm['yield']}"
    assert not np.isclose(cm["f_opt"][0], 0.0), \
        f"オーバーフロー候補(fx1=0)が最適解として選ばれてしまった: f_opt={cm['f_opt']}"
    assert cm["index"] >= 0

    print(f"[OK] test_solver_failure_never_high_yield "
          f"(f_opt={cm['f_opt']}, yield={cm['yield']:.3e}, n_feasible={grid['n_feasible']}/{grid['n_evaluated']})")


# =============================================================================
# 10. 感度ラッパの T スレッド（DEFECT 1 回帰）
# =============================================================================

def test_sensitivity_wrappers_thread_T_for_legacy_path():
    """upper_bound_sensitivity / lrp_sensitivity が T を grid_search_general に転送し、
    mode="legacy_path" で ValueError を出さずに走り、規定キーを持つ行を返すこと（DEFECT 1）。

    以前は両ラッパに T 引数が無く、legacy_path で grid_search_general が
    「T is required」で無条件クラッシュしていた。少なくとも一部のグリッド点が
    legacy-feasible になる設計（X0 を無漁獲平衡に置き f=0 の隅を feasible にする）で、
    feasible 経路も確実に踏ませる。
    """
    # 分離系（L12=L21=0）で安定。X0 を無漁獲平衡に置くと f=0 では定数軌道=legacy feasible。
    p = _params(r_x1=1.0, r_x2=0.5, r_y1=0.3, r_y2=0.3,
                L11=0.5, L12=0.0, L21=0.0, L22=0.4,
                C1=0.6, D1=0.0, C2=0.0, D2=0.5)
    means = np.ones(4)
    X0 = np.array([1.0, 1.5, 2.0, 1.25])  # 無漁獲平衡（数値確認済み）: f=0 で定数
    T = 5.0

    cfg = copy.deepcopy(sus.DEFAULT_SUSTAINABILITY)
    cfg["mode"] = "legacy_path"
    cfg["legacy"] = {"scope": "all", "mode": "endpoint", "tol": 0.1}

    # --- upper_bound_sensitivity ---
    ub_rows = sus.upper_bound_sensitivity(p, means, X0, cfg,
                                          f_upper_grid=[0.2, 0.4], lrp_ratio=0.3,
                                          n_grid=3, T=T)
    assert len(ub_rows) == 2, f"upper_bound_sensitivity の行数が想定外: {len(ub_rows)}"
    ub_required = {"fishing_upper_bound", "sustainability_mode", "yield_opt", "f_opt",
                   "at_upper_bound", "n_feasible", "diagnosis"}
    for r in ub_rows:
        assert ub_required.issubset(r.keys()), \
            f"upper_bound_sensitivity 行に不足キー: {ub_required - set(r.keys())}"
        assert r["sustainability_mode"] == "legacy_path"
    assert any(r["n_feasible"] > 0 for r in ub_rows), \
        "legacy-feasible 点が一つも無い（feasible経路を踏めていない）"

    # --- lrp_sensitivity ---
    lr_rows = sus.lrp_sensitivity(p, means, X0, cfg,
                                  lrp_ratios=[0.2, 0.4], f_upper=[0.4, 0.4, 0.4, 0.4],
                                  n_grid=3, T=T)
    assert len(lr_rows) == 2, f"lrp_sensitivity の行数が想定外: {len(lr_rows)}"
    lr_required = {"lrp_ratio", "sustainability_mode", "n_feasible", "n_evaluated",
                   "y_max", "constrained_maximum", "safe_solution", "trajectory_check"}
    for r in lr_rows:
        assert lr_required.issubset(r.keys()), \
            f"lrp_sensitivity 行に不足キー: {lr_required - set(r.keys())}"
    assert any(r["n_feasible"] > 0 for r in lr_rows), \
        "legacy-feasible 点が一つも無い（feasible経路を踏めていない）"

    # legacy_path を T なしで呼ぶと（従来通り）ValueError になることも確認（回帰の裏取り）
    raised = False
    try:
        sus.upper_bound_sensitivity(p, means, X0, cfg, f_upper_grid=[0.2],
                                    lrp_ratio=0.3, n_grid=2)  # T 省略
    except ValueError:
        raised = True
    assert raised, "legacy_path で T 省略時に ValueError が出るべき（T が本当に転送されている証拠）"

    print(f"[OK] test_sensitivity_wrappers_thread_T_for_legacy_path "
          f"(ub n_feasible={[r['n_feasible'] for r in ub_rows]}, "
          f"lr n_feasible={[r['n_feasible'] for r in lr_rows]})")


# =============================================================================
# 11. 非正平衡は solver_failure ではなく infeasible（DEFECT 2 回帰）
# =============================================================================

def test_equilibrium_non_positive_is_infeasible_not_solver_failure():
    """equilibrium_lrp で無漁獲平衡が非正のとき、classification は "solver_failure" ではなく
    "infeasible" になり、非正平衡の reason を surface すること（DEFECT 2）。

    equilibrium 系は ODE を積分しない（線形平衡ソルバ）ので、yield 未定義は
    「積分が壊れた」のではなく「平衡が正でないため収量が定義できない」ことに起因する。
    以前は n_success==0 を一律 "solver_failure" と誤ラベルしていた。
    """
    # 無漁獲平衡の x1 成分が負になる結合系（数値確認済み: B_eq_norm[0]<0）。
    p = _params(r_x1=1.0, r_x2=1.0, r_y1=1.0, r_y2=1.0,
                L11=1.0, L12=0.5, L21=0.5, L22=1.0,
                C1=5.0, D1=4.0, C2=2.0, D2=1.0)
    means = np.ones(4)
    X0 = np.ones(4)

    cfg = copy.deepcopy(sus.DEFAULT_SUSTAINABILITY)
    cfg["mode"] = "equilibrium_lrp"
    cfg["lrp_ratio"] = 0.3

    # 前提: 無漁獲平衡は解けるが正でない
    unfished = sus.compute_equilibrium(p, np.zeros(4), cfg)
    assert unfished["solvable"] is True, "この例は平衡が解ける（特異でない）はず"
    assert unfished["positive"] is False, \
        f"無漁獲平衡が正になってしまった（テスト設計を見直す必要あり）: {unfished['B_eq_norm']}"

    grid = sus.grid_search_general(p, means, X0, [0.5, 0.5, 0.5, 0.5],
                                    "equilibrium_lrp", cfg, n_grid=3)

    assert grid["n_feasible"] == 0, "この例は feasible 候補ゼロのはず"
    assert grid["n_success"] == 0, "非正平衡なので全候補で yield 未定義（n_success=0）のはず"

    cm = grid["constrained_maximum"]
    assert cm["classification"] == "infeasible", \
        f'classification が "infeasible" でない（solver_failure と誤ラベルの疑い）: {cm["classification"]}'
    assert cm["classification"] != "solver_failure"
    # 非正平衡の reason が surface されていること
    assert cm["reason"] == "no_positive_unfished_equilibrium", \
        f"非正平衡の reason が surface されていない: {cm.get('reason')}"
    assert cm["is_interior_optimum"] is False
    assert cm["is_bound_limited"] is False

    print(f"[OK] test_equilibrium_non_positive_is_infeasible_not_solver_failure "
          f"(classification={cm['classification']}, reason={cm['reason']}, n_success={grid['n_success']})")


# =============================================================================
def main():
    tests = [
        test_legacy_compatibility,
        test_phase_dependence_vs_equilibrium_invariance,
        test_equilibrium_analytical_two_species,
        test_non_positive_equilibrium_infeasible,
        test_upper_bound_binding,
        test_lrp_boundary_binding,
        test_near_optimal_safe_solution,
        test_time_average_yield_identity,
        test_solver_failure_never_high_yield,
        test_sensitivity_wrappers_thread_T_for_legacy_path,
        test_equilibrium_non_positive_is_infeasible_not_solver_failure,
    ]
    failed = []
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed.append((t.__name__, str(e)))
            print(f"[FAIL] {t.__name__}: {e}")
        except Exception as e:
            failed.append((t.__name__, repr(e)))
            print(f"[ERROR] {t.__name__}: {e!r}")

    if failed:
        print(f"\n{len(failed)}/{len(tests)} tests FAILED")
        sys.exit(1)
    else:
        print(f"\nAll {len(tests)} tests passed")
        sys.exit(0)


if __name__ == "__main__":
    main()
