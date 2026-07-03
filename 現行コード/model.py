"""
黒潮レジーム下の4種捕食被食モデル（capacity_ry: 12変数）と推定エンジン。

モデル: capacity_ry（12変数）
  r_x1, r_x2         : 被食者の自然増殖率
  r_y1, r_y2         : 捕食者の自然死亡率（推定）
  L11, L12, L21, L22 : 捕食圧（相互作用係数）
  C1, D1, C2, D2     : 捕食→捕食者への資源変換効率
  密度依存項（種内競争α）は含まない。

推定精度・頑健性のための仕組み:
  (A) マルチスタート最適化 ... 多数の初期値から least_squares を回し局所解を回避
  (B) 頑健な数値積分       ... LSODA + 許容誤差指定、失敗時も滑らかなペナルティ
  (C) 軽い正則化           ... 相互作用パラメータに L2 罰則をかけ過剰適合・非識別を抑制
  (D) 適合度の定量化       ... 実スケール(千トン)での種別 R^2 / NRMSE を返す

モデルは正規化空間（各種を全期間平均で割り平均1.0）で解き、対数誤差で評価する。

  被食者(x): マイワシ x1, カタクチイワシ x2
  捕食者(y): ブリ y1, サワラ y2
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp


# ----------------------------------------------------------------------
# モデル定義: capacity_ry（12変数）
# ----------------------------------------------------------------------
MODELS = {
    "capacity_ry": {
        "names":  ["r_x1", "r_x2", "r_y1", "r_y2", "L11", "L12", "L21", "L22",
                   "C1", "D1", "C2", "D2"],
        "guess":  [1.5, 1.5, 0.3, 0.4, 0.1, 0.1, 0.1, 0.1, 0.15, 0.15, 0.15, 0.15],
        "lower":  [0.1, 0.1, 0.01, 0.01, 1e-4, 1e-4, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3, 1e-3],
        "upper":  [5.0, 5.0, 2.0, 2.0, 5.0, 5.0, 5.0, 5.0, 10.0, 10.0, 10.0, 10.0],
        "reg_idx": [4, 5, 6, 7, 8, 9, 10, 11],
    },
}


def make_ode(fx1_i, fx2_i, fy1_i, fy2_i):
    """正規化空間の capacity_ry ODE 右辺を返す。"""
    def ode(t, state, p):
        x1, x2, y1, y2 = state
        r_x1, r_x2, r_y1, r_y2, L11, L12, L21, L22, C1, D1, C2, D2 = p
        x1 = max(1e-5, x1); x2 = max(1e-5, x2)
        y1 = max(1e-5, y1); y2 = max(1e-5, y2)
        dx1 = (r_x1 - fx1_i(t)) * x1 - L11 * x1 * y1 - L12 * x1 * y2
        dx2 = (r_x2 - fx2_i(t)) * x2 - L21 * x2 * y1 - L22 * x2 * y2
        dy1 = (-r_y1 - fy1_i(t)) * y1 + C1 * L11 * x1 * y1 + D1 * L21 * x2 * y1
        dy2 = (-r_y2 - fy2_i(t)) * y2 + C2 * L12 * x1 * y2 + D2 * L22 * x2 * y2
        return [dx1, dx2, dy1, dy2]
    return ode


def simulate(params, ode, t_rel, init):
    """頑健な積分。失敗時は None。"""
    try:
        sol = solve_ivp(ode, [t_rel[0], t_rel[-1]], init, t_eval=t_rel,
                        args=(params,), method="LSODA", rtol=1e-7, atol=1e-9)
    except Exception:
        return None
    if sol.status != 0 or sol.y.shape[1] != len(t_rel):
        return None
    if not np.all(np.isfinite(sol.y)):
        return None
    return sol.y


def estimate(series_slice, n_starts=32, reg_lambda=0.0, seed=0, verbose=False):
    """
    1レジーム分を capacity_ry（12変数）で推定する。

    n_starts   : マルチスタート回数（1なら単一スタート）
    reg_lambda : 相互作用パラメータへの L2 正則化強度（0で無効）

    返り値 dict(params_norm, params_abs, trajectory_abs, metrics, cost,
                means, names, at_bounds)
    """
    cfg = MODELS["capacity_ry"]
    names = cfg["names"]
    lower = np.array(cfg["lower"]); upper = np.array(cfg["upper"])
    guess0 = np.array(cfg["guess"])
    reg_idx = np.array(cfg["reg_idx"])

    years = series_slice["years"]
    t_rel = (years - years.min()).astype(float)
    obs_abs = [series_slice["x1"], series_slice["x2"],
               series_slice["y1"], series_slice["y2"]]
    means = np.array([np.mean(o) for o in obs_abs])
    obs_norm = [obs_abs[i] / means[i] for i in range(4)]
    init = [obs_norm[i][0] for i in range(4)]

    fx1_i = interp1d(t_rel, series_slice["fx1"], kind="linear", fill_value="extrapolate")
    fx2_i = interp1d(t_rel, series_slice["fx2"], kind="linear", fill_value="extrapolate")
    fy1_i = interp1d(t_rel, series_slice["fy1"], kind="linear", fill_value="extrapolate")
    fy2_i = interp1d(t_rel, series_slice["fy2"], kind="linear", fill_value="extrapolate")
    ode = make_ode(fx1_i, fx2_i, fy1_i, fy2_i)

    log_obs = [np.log(np.clip(o, 1e-5, None)) for o in obs_norm]
    n_pts = len(t_rel)

    def residuals(params):
        y = simulate(params, ode, t_rel, init)
        if y is None:
            base = np.ones(n_pts * 4) * 1e3
        else:
            log_y = np.log(np.clip(y, 1e-5, None))
            base = np.concatenate([log_y[i] - log_obs[i] for i in range(4)])
        if reg_lambda > 0:
            reg = np.sqrt(reg_lambda) * params[reg_idx]
            return np.concatenate([base, reg])
        return base

    rng = np.random.default_rng(seed)
    starts = [guess0.copy()]
    for _ in range(max(0, n_starts - 1)):
        g = np.empty(len(guess0))
        for k in range(len(guess0)):
            lo, hi = lower[k], upper[k]
            if lo > 0 and hi / lo > 50:
                g[k] = 10 ** rng.uniform(np.log10(lo), np.log10(hi))
            else:
                g[k] = rng.uniform(lo, hi)
        starts.append(g)

    best = None
    for s0 in starts:
        try:
            res = least_squares(residuals, s0, bounds=(lower, upper),
                                method="trf", max_nfev=4000,
                                verbose=1 if verbose else 0)
        except Exception:
            continue
        if best is None or res.cost < best.cost:
            best = res

    p = best.x
    traj_norm = simulate(p, ode, t_rel, init)
    traj_abs = np.vstack([traj_norm[i] * means[i] for i in range(4)])
    metrics = compute_metrics(traj_abs, obs_abs)

    at_bounds = []
    for k, nm in enumerate(names):
        if np.isclose(p[k], lower[k], rtol=1e-3) or np.isclose(p[k], upper[k], rtol=1e-3):
            at_bounds.append(nm)

    return {
        "params_norm": p,
        "params_abs":  _to_absolute(p, means),
        "trajectory_abs": traj_abs,
        "metrics": metrics,
        "cost": best.cost,
        "means": means,
        "names": names,
        "at_bounds": at_bounds,
    }


def _to_absolute(p, means):
    """正規化パラメータを元の物理スケールに換算。"""
    mx1, mx2, my1, my2 = means
    r_x1, r_x2, r_y1, r_y2, L11, L12, L21, L22, C1, D1, C2, D2 = p
    return {
        "r_x1": r_x1, "r_x2": r_x2, "r_y1": r_y1, "r_y2": r_y2,
        "l11": L11 / my1, "l12": L12 / my2, "l21": L21 / my1, "l22": L22 / my2,
        "c1": C1 * my1 / mx1, "d1": D1 * my1 / mx2,
        "c2": C2 * my2 / mx1, "d2": D2 * my2 / mx2,
    }


def compute_metrics(traj_abs, obs_abs):
    """実スケール(千トン)での種別 R^2 / RMSE / NRMSE。"""
    labels = ["x1", "x2", "y1", "y2"]
    out = {}
    r2s, nrmses = [], []
    for i, lab in enumerate(labels):
        o, pr = obs_abs[i], traj_abs[i]
        ss_res = np.sum((o - pr) ** 2)
        ss_tot = np.sum((o - np.mean(o)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        rmse = np.sqrt(np.mean((o - pr) ** 2))
        nrmse = rmse / (np.mean(o) if np.mean(o) != 0 else 1.0)
        out[lab] = {"R2": r2, "RMSE": rmse, "NRMSE": nrmse}
        r2s.append(r2); nrmses.append(nrmse)
    out["overall"] = {"mean_R2": float(np.mean(r2s)),
                      "mean_NRMSE": float(np.mean(nrmses))}
    return out
