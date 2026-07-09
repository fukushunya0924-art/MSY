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
import os
import functools
import multiprocessing
from typing import Callable, Optional

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

# 状態変数のアンダーフロー・ゼロ割り回避用フロア（正規化空間, 平均1.0スケール）
_STATE_FLOOR = 1e-5
# 対数誤差評価時に log(0) を避けるためのクリップ下限（正規化空間）
_LOG_CLIP_MIN = 1e-5
# 積分失敗（solve_ivp が非収束/例外）時に least_squares へ返す一律ペナルティ残差
_INTEGRATION_FAILURE_PENALTY = 1e3
# パラメータが探索範囲の上下限に「一致」とみなす相対許容誤差（at_bounds 判定用）
_BOUNDS_RTOL = 1e-3


def make_ode(fx1_i: Callable[[float], float], fx2_i: Callable[[float], float],
             fy1_i: Callable[[float], float], fy2_i: Callable[[float], float]):
    """正規化空間の capacity_ry ODE 右辺を返す。

    fx1_i, fx2_i, fy1_i, fy2_i : 各種の漁獲圧 f(t) を返す補間関数
        （interp1d 等。年次観測値を線形補間して連続時間の ODE に注入する）。
    """
    def ode(t, state, p):
        x1, x2, y1, y2 = state
        r_x1, r_x2, r_y1, r_y2, L11, L12, L21, L22, C1, D1, C2, D2 = p
        # ゼロ割り・負値化を防ぐフロア（生物量は物理的に正のはず）
        x1 = max(_STATE_FLOOR, x1); x2 = max(_STATE_FLOOR, x2)
        y1 = max(_STATE_FLOOR, y1); y2 = max(_STATE_FLOOR, y2)
        dx1 = (r_x1 - fx1_i(t)) * x1 - L11 * x1 * y1 - L12 * x1 * y2
        dx2 = (r_x2 - fx2_i(t)) * x2 - L21 * x2 * y1 - L22 * x2 * y2
        dy1 = (-r_y1 - fy1_i(t)) * y1 + C1 * L11 * x1 * y1 + D1 * L21 * x2 * y1
        dy2 = (-r_y2 - fy2_i(t)) * y2 + C2 * L12 * x1 * y2 + D2 * L22 * x2 * y2
        return [dx1, dx2, dy1, dy2]
    return ode


def simulate(params: np.ndarray, ode: Callable, t_rel: np.ndarray,
             init: list) -> Optional[np.ndarray]:
    """正規化空間で ODE を積分する。失敗時（非収束・例外・非有限値）は None を返す。"""
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


# マルチスタートの初期値サンプリングで、対数一様分布に切り替える
# 探索範囲の比（upper/lower）の閾値。桁がこれを超えて広いパラメータ
# （L11..L22, C1..D2 など）は対数空間で、そうでなければ線形空間で一様サンプルする。
_LOG_UNIFORM_RATIO_THRESHOLD = 50
# least_squares(method="trf") の最大関数評価回数
_MAX_NFEV = 4000


def estimate(series_slice, n_starts=32, reg_lambda=0.0, seed=0, verbose=False):
    """
    1レジーム分を capacity_ry（12変数）で推定する。

    n_starts   : マルチスタート回数（1なら単一スタート）
    reg_lambda : 相互作用パラメータへの L2 正則化強度（0で無効）

    返り値 dict(params_norm, params_abs, trajectory_abs, metrics, cost,
                means, names, at_bounds)

    注意: 全マルチスタートで least_squares が例外を投げた場合（極めて稀）、
    best は None のままとなり後続の best.x で AttributeError になる。
    データが有限で MODELS の初期値・境界が妥当なら通常発生しない。
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

    log_obs = [np.log(np.clip(o, _LOG_CLIP_MIN, None)) for o in obs_norm]
    n_pts = len(t_rel)

    def residuals(params):
        """4種×n_pts点の対数誤差残差（+ 正則化項）。simulate失敗時は一律ペナルティ。"""
        y = simulate(params, ode, t_rel, init)
        if y is None:
            base = np.full(n_pts * 4, _INTEGRATION_FAILURE_PENALTY)
        else:
            log_y = np.log(np.clip(y, _LOG_CLIP_MIN, None))
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
            if lo > 0 and hi / lo > _LOG_UNIFORM_RATIO_THRESHOLD:
                g[k] = 10 ** rng.uniform(np.log10(lo), np.log10(hi))
            else:
                g[k] = rng.uniform(lo, hi)
        starts.append(g)

    best = None
    for s0 in starts:
        try:
            res = least_squares(residuals, s0, bounds=(lower, upper),
                                method="trf", max_nfev=_MAX_NFEV,
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
        if (np.isclose(p[k], lower[k], rtol=_BOUNDS_RTOL)
                or np.isclose(p[k], upper[k], rtol=_BOUNDS_RTOL)):
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


def _estimate_worker(seed: int, series_slice: dict, n_starts: int,
                      reg_lambda: float, verbose: bool) -> Optional[dict]:
    """multiprocessing ワーカー: 1 seed 分の estimate() を実行する（モジュールトップレベル、pickle可能）。

    例外・異常時は None を返す（呼び出し側でスキップされる）。
    """
    try:
        res = estimate(series_slice, n_starts=n_starts, reg_lambda=reg_lambda,
                       seed=seed, verbose=verbose)
    except Exception:
        return None
    if res is None:
        return None
    res["best_seed"] = seed
    if verbose:
        print(f"[seed={seed}] cost={res['cost']:.6f}")
    return res


def estimate_robust(series_slice, n_starts=64, reg_lambda=0.0, n_seeds=12, seed0=0, verbose=False):
    """
    複数シードで estimate を並列実行し総コスト最小解を採る。単一シードでは局所解に落ちるため。

    seed0 から seed0+n_seeds-1 まで seed を変えて estimate() を n_seeds 回、
    multiprocessing.Pool で並列に呼び、res['cost'] が最小の結果を返す。
    返り値は estimate() と同一形式の dict に加え、最良だったシード番号を
    res['best_seed'] に格納する（デバッグ用）。

    seed 間は完全独立なため、ワーカー数 min(n_seeds, cpu_count) で並列化する。
    macOS は spawn 方式のためワーカー関数はモジュールトップレベルの _estimate_worker。
    あるseedが例外/失敗しても他のseedの結果で継続する（全滅時のみ best=None）。
    """
    seeds = list(range(seed0, seed0 + n_seeds))
    n_workers = max(1, min(n_seeds, os.cpu_count() or 1))

    worker = functools.partial(_estimate_worker, series_slice=series_slice,
                               n_starts=n_starts, reg_lambda=reg_lambda,
                               verbose=verbose)

    with multiprocessing.Pool(processes=n_workers) as pool:
        results = pool.map(worker, seeds)

    best = None
    for res in results:
        if res is None:
            continue
        if best is None or res["cost"] < best["cost"]:
            best = res

    if best is None:
        raise RuntimeError("estimate_robust: 全seedでestimateが失敗しました")

    if verbose:
        print(f"[estimate_robust] best_seed={best['best_seed']} cost={best['cost']:.6f}")

    return best


def _to_absolute(p: np.ndarray, means: np.ndarray) -> dict:
    """正規化パラメータを元の物理スケール（千トン, 1/年）の物理パラメータへ換算。

    正規化: x_abs = mean_x * x_norm （各種を全期間平均 means で除した空間）。
    捕食項 L_ij * x_norm * y_norm を絶対空間へ戻すと mean が非対称に効くため、
    l_ij = L_ij / mean_y（yのみで除す）, c_i/d_i = C_i・mean_y / mean_x
    （x側の平均で割り、y側の平均を掛ける）という非対称の換算式になる。
    r_x, r_y は無次元の率（1/年）なのでスケール換算不要でそのまま。
    """
    mx1, mx2, my1, my2 = means
    r_x1, r_x2, r_y1, r_y2, L11, L12, L21, L22, C1, D1, C2, D2 = p
    return {
        "r_x1": r_x1, "r_x2": r_x2, "r_y1": r_y1, "r_y2": r_y2,
        "l11": L11 / my1, "l12": L12 / my2, "l21": L21 / my1, "l22": L22 / my2,
        "c1": C1 * my1 / mx1, "d1": D1 * my1 / mx2,
        "c2": C2 * my2 / mx1, "d2": D2 * my2 / mx2,
    }


def compute_metrics(traj_abs: np.ndarray, obs_abs: list) -> dict:
    """実スケール(千トン)での種別 R^2 / RMSE / NRMSE と種横断平均を返す。"""
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
