"""
capacity_ry（12変数）のうち r_x1, r_x2, c1+d1, c2+d2 を Catch-MSY 外挿値に固定し、
残り8自由変数だけを推定する制約付きモデル。

ODE右辺・積分・適合度計算は model.py のものをそのまま再利用し、ここでは
再パラメータ化（8自由変数 -> 12変数への復元）と、それに伴う推定エンジンの
組み替えのみを行う。ODE右辺や積分ロジックの再実装はしない。

固定する4値（fixed_params.py が単一の真実の源）:
  r_x1, r_x2 : 被食者の自然増殖率（無次元の率なので正規化/絶対スケールで同値）
  S1 = c1+d1 : ブリの捕食→変換効率の和（絶対スケール）
  S2 = c2+d2 : サワラの捕食→変換効率の和（絶対スケール）

自由変数（8次元）:
  q = [r_y1, r_y2, L11, L12, L21, L22, theta1, theta2]

  r_y1, r_y2, L11, L12, L21, L22 は model.py の capacity_ry と同じ意味・スケール
  （正規化空間）。theta1, theta2 in [0,1] は S1, S2 を c/d へ配分する比で、
    c1 = theta1*S1,       d1 = (1-theta1)*S1
    c2 = theta2*S2,       d2 = (1-theta2)*S2
  として絶対スケールの c1,d1,c2,d2 を決め、それを正規化空間の C1,D1,C2,D2 へ
  逆算する（reconstruct_params 参照）。
"""
import os
import functools
import multiprocessing
from typing import Optional

import numpy as np
from scipy.optimize import least_squares
from scipy.interpolate import interp1d

import model
import fixed_params


# ----------------------------------------------------------------------
# 自由変数（8次元）の定義
# ----------------------------------------------------------------------
FREE_NAMES = ["r_y1", "r_y2", "L11", "L12", "L21", "L22", "theta1", "theta2"]
FREE_GUESS = [0.3, 0.4, 0.1, 0.1, 0.1, 0.1, 0.5, 0.5]
FREE_LOWER = [0.01, 0.01, 1e-4, 1e-4, 1e-4, 1e-4, 1e-6, 1e-6]
FREE_UPPER = [2.0, 2.0, 5.0, 5.0, 5.0, 5.0, 1.0 - 1e-6, 1.0 - 1e-6]

# 正則化の対象（q の先頭6要素: r_y1, r_y2, L11, L12, L21, L22）。
# theta1, theta2 (index 6, 7) には罰則をかけない
# （かけると配分比が 0.5 に引っ張られ「固定値 S を分配する」という意味が崩れるため）。
_REG_SLICE = slice(0, 6)

# マルチスタートのサンプリング方式をパラメータごとに明示指定する。
# r_y1, r_y2（比 2.0/0.01=200）と L11..L22（比 5.0/1e-4=50000）は
# model.py と同じ閾値ルールでも対数一様が選ばれるため "log" のままでよい。
# theta1, theta2 は bounds の比が 1e6 近くにあり、閾値ルールに素直に従うと
# 対数一様になって 0 近傍に極端に偏ってしまう（配分比としては不自然）ため、
# 必ず線形一様 "linear" を強制する。
_FREE_SAMPLE_MODES = ["log", "log", "log", "log", "log", "log", "linear", "linear"]


def reconstruct_params(q: np.ndarray, means: np.ndarray, fixed: dict) -> np.ndarray:
    """8自由変数 q と固定値 fixed から、capacity_ry の12次元 params_norm を復元する。

    q      : [r_y1, r_y2, L11, L12, L21, L22, theta1, theta2]
    means  : [mx1, mx2, my1, my2]（そのレジームの各種全期間平均、正規化に使った値）
    fixed  : dict(r_x1, r_x2, S1, S2)（fixed_params.get_point() 形式）

    返り値は model.MODELS["capacity_ry"]["names"] と完全に同じ並び
    [r_x1, r_x2, r_y1, r_y2, L11, L12, L21, L22, C1, D1, C2, D2] の ndarray。

    換算式（model._to_absolute の逆演算）:
      絶対スケールで c1+d1=S1, c2+d2=S2 を厳密に満たすように
        c1 = theta1*S1,  d1 = (1-theta1)*S1
        c2 = theta2*S2,  d2 = (1-theta2)*S2
      を作り、それを正規化空間へ戻す:
        C1 = c1 * mx1 / my1
        D1 = d1 * mx2 / my1
        C2 = c2 * mx1 / my2
        D2 = d2 * mx2 / my2
      （model._to_absolute の c1 = C1*my1/mx1 等の逆関数になっている）
    """
    mx1, mx2, my1, my2 = means
    r_y1, r_y2, L11, L12, L21, L22, theta1, theta2 = q

    r_x1 = fixed["r_x1"]
    r_x2 = fixed["r_x2"]
    S1 = fixed["S1"]
    S2 = fixed["S2"]

    c1 = theta1 * S1
    d1 = (1.0 - theta1) * S1
    c2 = theta2 * S2
    d2 = (1.0 - theta2) * S2

    C1 = c1 * mx1 / my1
    D1 = d1 * mx2 / my1
    C2 = c2 * mx1 / my2
    D2 = d2 * mx2 / my2

    return np.array([r_x1, r_x2, r_y1, r_y2, L11, L12, L21, L22, C1, D1, C2, D2])


def _sample_starts(rng: np.random.Generator, n_starts: int,
                    lower: np.ndarray, upper: np.ndarray,
                    guess0: np.ndarray, sample_modes: list) -> list:
    """マルチスタートの初期値集合を作る。1個目は guess0、残りは乱数サンプル。

    sample_modes[k] == "log"    : 10**uniform(log10(lo), log10(hi))
    sample_modes[k] == "linear" : uniform(lo, hi)
    """
    starts = [guess0.copy()]
    for _ in range(max(0, n_starts - 1)):
        g = np.empty(len(guess0))
        for k in range(len(guess0)):
            lo, hi = lower[k], upper[k]
            if sample_modes[k] == "log":
                g[k] = 10 ** rng.uniform(np.log10(lo), np.log10(hi))
            else:
                g[k] = rng.uniform(lo, hi)
        starts.append(g)
    return starts


def estimate_constrained(series_slice, n_starts=32, reg_lambda=0.0, seed=0,
                          fixed=None, verbose=False):
    """
    1レジーム分を、r_x1/r_x2/c1+d1/c2+d2 を固定した8自由変数で推定する。

    model.estimate() と同じ骨格（データ整形→残差関数→マルチスタート
    least_squares→絶対スケール換算）だが、最適化変数が8次元 q である点、
    残差関数の中で reconstruct_params により12次元 params_norm を
    組み立ててから simulate に渡す点が異なる。

    n_starts   : マルチスタート回数（1なら単一スタート）
    reg_lambda : q の先頭6要素（r_y1,r_y2,L11,L12,L21,L22）への L2 正則化強度
                 （0で無効。theta1, theta2 には掛けない）
    fixed      : dict(r_x1, r_x2, S1, S2)。None なら fixed_params.get_point()

    返り値 dict は model.estimate() と同一形状
    （params_norm, params_abs, trajectory_abs, metrics, cost, means, names,
    at_bounds）に加えて fixed, free_names, params_free を含む。

    注意: model.estimate() 同様、全マルチスタートで least_squares が例外を
    投げた場合（極めて稀）は best が None のままとなり後続で例外になる。
    """
    if fixed is None:
        fixed = fixed_params.get_point()

    lower = np.array(FREE_LOWER)
    upper = np.array(FREE_UPPER)
    guess0 = np.array(FREE_GUESS)

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
    ode = model.make_ode(fx1_i, fx2_i, fy1_i, fy2_i)

    log_obs = [np.log(np.clip(o, model._LOG_CLIP_MIN, None)) for o in obs_norm]
    n_pts = len(t_rel)

    def residuals(q):
        """4種×n_pts点の対数誤差残差（+ 正則化項）。simulate失敗時は一律ペナルティ。"""
        params_norm = reconstruct_params(q, means, fixed)
        y = model.simulate(params_norm, ode, t_rel, init)
        if y is None:
            base = np.full(n_pts * 4, model._INTEGRATION_FAILURE_PENALTY)
        else:
            log_y = np.log(np.clip(y, model._LOG_CLIP_MIN, None))
            base = np.concatenate([log_y[i] - log_obs[i] for i in range(4)])
        if reg_lambda > 0:
            reg = np.sqrt(reg_lambda) * q[_REG_SLICE]
            return np.concatenate([base, reg])
        return base

    rng = np.random.default_rng(seed)
    starts = _sample_starts(rng, n_starts, lower, upper, guess0, _FREE_SAMPLE_MODES)

    best = None
    for q0 in starts:
        try:
            res = least_squares(residuals, q0, bounds=(lower, upper),
                                method="trf", max_nfev=model._MAX_NFEV,
                                verbose=1 if verbose else 0)
        except Exception:
            continue
        if best is None or res.cost < best.cost:
            best = res

    q_best = best.x
    params_norm = reconstruct_params(q_best, means, fixed)
    traj_norm = model.simulate(params_norm, ode, t_rel, init)
    traj_abs = np.vstack([traj_norm[i] * means[i] for i in range(4)])
    metrics = model.compute_metrics(traj_abs, obs_abs)

    at_bounds = []
    for k, nm in enumerate(FREE_NAMES):
        if (np.isclose(q_best[k], lower[k], rtol=model._BOUNDS_RTOL)
                or np.isclose(q_best[k], upper[k], rtol=model._BOUNDS_RTOL)):
            at_bounds.append(nm)

    return {
        "params_norm": params_norm,
        "params_abs":  model._to_absolute(params_norm, means),
        "trajectory_abs": traj_abs,
        "metrics": metrics,
        "cost": best.cost,
        "means": means,
        "names": model.MODELS["capacity_ry"]["names"],
        "at_bounds": at_bounds,
        "fixed": dict(fixed),
        "free_names": list(FREE_NAMES),
        "params_free": q_best,
    }


def _estimate_constrained_worker(seed: int, series_slice: dict, n_starts: int,
                                  reg_lambda: float, fixed: dict,
                                  verbose: bool) -> Optional[dict]:
    """multiprocessing ワーカー: 1 seed 分の estimate_constrained() を実行する
    （モジュールトップレベル、pickle可能。macOS の spawn 方式に対応）。

    例外・異常時は None を返す（呼び出し側でスキップされる）。
    """
    try:
        res = estimate_constrained(series_slice, n_starts=n_starts,
                                   reg_lambda=reg_lambda, seed=seed,
                                   fixed=fixed, verbose=verbose)
    except Exception:
        return None
    if res is None:
        return None
    res["best_seed"] = seed
    if verbose:
        print(f"[seed={seed}] cost={res['cost']:.6f}")
    return res


def estimate_constrained_robust(series_slice, n_starts=64, reg_lambda=0.0,
                                 n_seeds=12, seed0=0, fixed=None, verbose=False):
    """
    複数シードで estimate_constrained を並列実行し総コスト最小解を採る。

    model.estimate_robust() と同じマルチシード並列構造。seed0 から
    seed0+n_seeds-1 まで seed を変えて estimate_constrained() を n_seeds 回、
    multiprocessing.Pool で並列に呼び、res['cost'] が最小の結果を返す。
    返り値は estimate_constrained() と同一形式の dict に加え、最良だった
    シード番号を res['best_seed'] に格納する（デバッグ用）。

    あるseedが例外/失敗しても他のseedの結果で継続する（全滅時のみ例外）。
    fixed=None のときは fixed_params.get_point() を既定値として使う。
    """
    if fixed is None:
        fixed = fixed_params.get_point()

    seeds = list(range(seed0, seed0 + n_seeds))
    n_workers = max(1, min(n_seeds, os.cpu_count() or 1))

    worker = functools.partial(_estimate_constrained_worker, series_slice=series_slice,
                               n_starts=n_starts, reg_lambda=reg_lambda,
                               fixed=fixed, verbose=verbose)

    with multiprocessing.Pool(processes=n_workers) as pool:
        results = pool.map(worker, seeds)

    best = None
    for res in results:
        if res is None:
            continue
        if best is None or res["cost"] < best["cost"]:
            best = res

    if best is None:
        raise RuntimeError("estimate_constrained_robust: 全seedでestimate_constrainedが失敗しました")

    if verbose:
        print(f"[estimate_constrained_robust] best_seed={best['best_seed']} cost={best['cost']:.6f}")

    return best
