"""
MSY（最大持続生産量）計算のコア関数群。

4種捕食被食ODEモデルに「定数漁獲圧」を注入し、一定積分期間 T の平均漁獲量を
最大化する漁獲率ベクトル f* を走査によって求める。

  被食者(x): マイワシ x1, カタクチイワシ x2
  捕食者(y): ヤリイカ y1, スルメイカ y2

基本方針
  - make_ode(model, fx1_i, fx2_i, fy1_i, fy2_i) に定数を返す関数 `lambda t: f_const`
    を渡すことで、既存 ODE 定義をそのまま再利用（新たな ODE は書かない）。
  - 正規化空間で ODE を解き、絶対スケール（千トン）に戻して漁獲量を計算。
  - 積分は台形則（np.trapz）/ T で時間平均。
  - 走査方式: ブラックボックス最適化はせず、グリッド全評価（列挙）で MSY 点を探す。

計算規模の定数（ファイル先頭でまとめて調整可能）
  N_EVAL_TRAJ : ODE 数値積分の t_eval 点数（細かいほど積分精度が上がるが時間増）
  N_COMMON    : 共通漁獲率スイープの点数
  N_GRID      : グリッド探索の各軸点数（N_GRID^4 = 6^4 = 1296 評価）
  N_SENS      : 種別感度スイープの点数
"""

import numpy as np
from scipy.integrate import solve_ivp

# =============================================================================
# 計算規模に関わる定数
# =============================================================================
N_EVAL_TRAJ = 200   # ODE 軌道の t_eval 点数（[0,T] を 200 分割）
N_COMMON    = 40    # 共通漁獲率スイープ: linspace(0, 0.95, 40)
N_GRID      = 6     # グリッド探索の各軸点数: 6^4 = 1296 評価
N_SENS      = 40    # 種別感度スイープ: linspace(0, 0.95, 40)

F_MAX       = 0.95  # data_loader と同じ漁獲圧の上限
F_MIN       = 0.0   # 漁獲圧の下限


# =============================================================================
# ヘルパ関数
# =============================================================================

def normalize_X0(obs_abs_at_t0, means):
    """
    初年観測資源量（絶対スケール, 4要素 ndarray）を正規化空間の初期値に変換する。

    Parameters
    ----------
    obs_abs_at_t0 : array-like, shape (4,)
        各種の初年資源量（千トン）。[x1, x2, y1, y2] の順。
    means : array-like, shape (4,)
        推定スライスの各種平均資源量（千トン）。estimate() 返り値の 'means' と同じ。

    Returns
    -------
    ndarray, shape (4,) : 正規化初期値（各種を means[i] で割った値）。
    """
    obs = np.asarray(obs_abs_at_t0, dtype=float)
    m   = np.asarray(means, dtype=float)
    return obs / m


def _const_f_ode(model_str, f_vec):
    """
    定数漁獲圧ベクトル f_vec を注入した ODE を返す。

    親モジュール model.py の make_ode() に「定数を返す callable」を渡すことで、
    既存の ODE 定義を一切書き換えずに再利用する。

    Parameters
    ----------
    model_str : str
        "capacity" または "capacity_ry"
    f_vec : array-like, shape (4,)
        定数漁獲圧 [f_x1, f_x2, f_y1, f_y2]

    Returns
    -------
    callable : ode(t, state, p) の形式で solve_ivp に渡せる ODE 右辺
    """
    # 遅延 import（rank1 等と同じパターンで親フォルダの model.py を使う）
    from model import make_ode

    f_x1, f_x2, f_y1, f_y2 = f_vec
    return make_ode(
        model_str,
        lambda t: f_x1,
        lambda t: f_x2,
        lambda t: f_y1,
        lambda t: f_y2,
    )


def average_yield(f_vec, params_norm, means, model_str, T, X0_norm, n_eval=N_EVAL_TRAJ):
    """
    定数漁獲圧 f_vec のもとで ODE を積分し、4種の平均漁獲量を計算する。

    漁獲量（絶対スケール, 千トン/年）:
      Y_i(t) = f_i * X_i(t)   [正規化軌道 × means で絶対スケールに換算後]

    時間平均漁獲量（スカラー）:
      mean_yield = (1/T) * ∫₀ᵀ Σᵢ Y_i(t) dt  ≈ trapz / T

    Parameters
    ----------
    f_vec : array-like, shape (4,)
        定数漁獲圧 [f_x1, f_x2, f_y1, f_y2] 。各要素 ∈ [0, 0.95]。
    params_norm : ndarray
        estimate() 返り値の 'params_norm'（正規化パラメータ）。
    means : ndarray, shape (4,)
        estimate() 返り値の 'means'（各種の全期間平均資源量, 千トン）。
    model_str : str
        "capacity" または "capacity_ry"
    T : float
        積分期間（年）。
    X0_norm : array-like, shape (4,)
        正規化初期値（normalize_X0() で作成）。
    n_eval : int
        t_eval 点数（デフォルト N_EVAL_TRAJ）。

    Returns
    -------
    dict with keys:
      'mean_yield'          : float  合計の時間平均漁獲量（千トン/年）。積分失敗時は -inf。
      'per_species_yield'   : ndarray shape (4,)  種別時間平均漁獲量（千トン/年）
      't'                   : ndarray  時間グリッド [0, T]
      'traj_abs'            : ndarray shape (4, n_eval)  絶対スケール軌道（千トン）。失敗時 None。
      'success'             : bool  ODE 積分が成功したか。
    """
    f_vec = np.asarray(f_vec, dtype=float)
    means = np.asarray(means, dtype=float)
    X0_norm = np.asarray(X0_norm, dtype=float)

    ode = _const_f_ode(model_str, f_vec)
    t_eval = np.linspace(0.0, T, n_eval)

    try:
        sol = solve_ivp(
            ode,
            [0.0, T],
            X0_norm.tolist(),
            args=(params_norm,),
            method="LSODA",
            rtol=1e-7,
            atol=1e-9,
            t_eval=t_eval,
        )
    except Exception:
        return {
            "mean_yield": -np.inf,
            "per_species_yield": np.full(4, -np.inf),
            "t": t_eval,
            "traj_abs": None,
            "success": False,
        }

    if sol.status != 0 or sol.y.shape[1] != len(t_eval) or not np.all(np.isfinite(sol.y)):
        return {
            "mean_yield": -np.inf,
            "per_species_yield": np.full(4, -np.inf),
            "t": t_eval,
            "traj_abs": None,
            "success": False,
        }

    # 絶対スケール軌道（千トン）: traj_norm[i] * means[i]
    traj_abs = sol.y * means[:, np.newaxis]   # shape (4, n_eval)

    # 種別漁獲量軌道 Y_i(t) = f_i * X_i(t)（千トン/年）
    yield_traj = f_vec[:, np.newaxis] * traj_abs  # shape (4, n_eval)

    # 時間平均（台形則 / T）
    per_species_yield = np.array([
        np.trapz(yield_traj[i], t_eval) / T
        for i in range(4)
    ])
    mean_yield = float(np.sum(per_species_yield))

    # 負の資源量や非物理的な軌道（任意の時点でも負値）は除外
    if np.any(traj_abs < 0):
        return {
            "mean_yield": -np.inf,
            "per_species_yield": np.full(4, -np.inf),
            "t": t_eval,
            "traj_abs": None,
            "success": False,
        }

    return {
        "mean_yield": mean_yield,
        "per_species_yield": per_species_yield,
        "t": t_eval,
        "traj_abs": traj_abs,
        "success": True,
    }


# =============================================================================
# スイープ 1: 共通漁獲率スイープ
# =============================================================================

def scan_common_rate(params_norm, means, model_str, T, X0_norm,
                     n_common=N_COMMON, n_eval=N_EVAL_TRAJ):
    """
    全4種に同じ定数漁獲率 f_common ∈ linspace(0, F_MAX, n_common) を与え、
    平均漁獲量 vs f_common の収量曲線を求める。

    これは「黒潮レジームごとの収量応答曲線」の概観を得るための一次スキャン。

    Parameters
    ----------
    params_norm : ndarray
        estimate() 返り値の 'params_norm'
    means : ndarray, shape (4,)
        estimate() 返り値の 'means'
    model_str : str
        "capacity" または "capacity_ry"
    T : float
        積分期間（年）
    X0_norm : array-like, shape (4,)
        正規化初期値
    n_common : int
        スイープ点数（デフォルト N_COMMON=40）
    n_eval : int
        ODE 積分の t_eval 点数

    Returns
    -------
    dict with keys:
      'f_common'      : ndarray, shape (n_common,)  漁獲率グリッド
      'mean_yield'    : ndarray, shape (n_common,)  対応する平均漁獲量（千トン/年）
      'per_species'   : ndarray, shape (4, n_common)  種別平均漁獲量
      'best_f'        : float  最大収量を達成した f_common
      'best_yield'    : float  最大収量値（千トン/年）
    """
    f_grid = np.linspace(F_MIN, F_MAX, n_common)
    mean_yields   = np.full(n_common, np.nan)
    per_species   = np.full((4, n_common), np.nan)

    for i, fc in enumerate(f_grid):
        f_vec = np.full(4, fc)
        res = average_yield(f_vec, params_norm, means, model_str, T, X0_norm, n_eval=n_eval)
        if res["success"]:
            mean_yields[i] = res["mean_yield"]
            per_species[:, i] = res["per_species_yield"]

    # NaN を除いて最大を探す
    valid = np.isfinite(mean_yields)
    if valid.any():
        best_idx = int(np.nanargmax(mean_yields))
        best_f   = float(f_grid[best_idx])
        best_yield = float(mean_yields[best_idx])
    else:
        best_f = float("nan")
        best_yield = float("nan")

    return {
        "f_common":   f_grid,
        "mean_yield": mean_yields,
        "per_species": per_species,
        "best_f":     best_f,
        "best_yield": best_yield,
    }


# =============================================================================
# スイープ 2: 4次元粗グリッド探索
# =============================================================================

def grid_search_msy(params_norm, means, model_str, T, X0_norm,
                    n_grid=N_GRID, n_eval=N_EVAL_TRAJ):
    """
    各 fᵢ ∈ linspace(0, F_MAX, n_grid) の直積グリッドを全列挙し、
    平均漁獲量を最大化する漁獲率ベクトル f* と MSY 値を求める。

    計算量: n_grid^4 = 6^4 = 1296 評価（各評価は ODE 1回 + 台形則）
             n_grid=8 なら 4096 評価。

    Parameters
    ----------
    params_norm : ndarray
    means : ndarray, shape (4,)
    model_str : str
    T : float
    X0_norm : array-like, shape (4,)
    n_grid : int
        各軸の点数（デフォルト N_GRID=6、6^4=1296 評価）
    n_eval : int
        ODE t_eval 点数

    Returns
    -------
    dict with keys:
      'f_star'             : ndarray, shape (4,)  MSY を達成する最適漁獲率
      'msy'                : float  最大平均漁獲量（千トン/年）
      'per_species_at_msy' : ndarray, shape (4,)  f* での種別漁獲量（千トン/年）
      'n_evaluated'        : int  評価したグリッド点数
      'n_success'          : int  ODE が成功した評価数
      'all_f'              : ndarray, shape (n_evaluated, 4)  全グリッド点
      'all_yield'          : ndarray, shape (n_evaluated,)   全グリッド点の漁獲量
    """
    f_axis = np.linspace(F_MIN, F_MAX, n_grid)

    # 全組み合わせを ndarray として生成 (n_grid^4, 4)
    g = np.array(np.meshgrid(f_axis, f_axis, f_axis, f_axis, indexing="ij"))
    all_f = g.reshape(4, -1).T   # shape (n_grid^4, 4)

    n_total = len(all_f)
    all_yield   = np.full(n_total, np.nan)
    per_species_best = None
    best_yield = -np.inf
    best_idx   = -1
    n_success  = 0

    for idx, f_vec in enumerate(all_f):
        res = average_yield(f_vec, params_norm, means, model_str, T, X0_norm, n_eval=n_eval)
        if res["success"]:
            all_yield[idx] = res["mean_yield"]
            n_success += 1
            if res["mean_yield"] > best_yield:
                best_yield = res["mean_yield"]
                best_idx   = idx
                per_species_best = res["per_species_yield"].copy()

    if best_idx >= 0:
        f_star = all_f[best_idx].copy()
        msy    = float(best_yield)
        per_sp = per_species_best
    else:
        f_star = np.full(4, float("nan"))
        msy    = float("nan")
        per_sp = np.full(4, float("nan"))

    return {
        "f_star":             f_star,
        "msy":                msy,
        "per_species_at_msy": per_sp,
        "n_evaluated":        n_total,
        "n_success":          n_success,
        "all_f":              all_f,
        "all_yield":          all_yield,
    }


# =============================================================================
# スイープ 3: 種別 1 次元感度
# =============================================================================

def species_sensitivity(f_star, params_norm, means, model_str, T, X0_norm,
                        n_sens=N_SENS, n_eval=N_EVAL_TRAJ):
    """
    f* を基準に、1種ずつ fᵢ を [0, F_MAX] で動かし他は f* 固定にした
    収量曲線（4本）を求める。

    これにより「MSY 点付近での各魚種漁獲率の感度」が分かり、
    どの種の漁獲圧を変えると収量が大きく変化するかを把握できる。

    Parameters
    ----------
    f_star : array-like, shape (4,)
        グリッド探索で得た最適漁獲率（基準点）
    params_norm, means, model_str, T, X0_norm : estimate() の結果と同じ
    n_sens : int
        1 次元スイープ点数（デフォルト N_SENS=40）
    n_eval : int
        ODE t_eval 点数

    Returns
    -------
    list of 4 dicts, 各要素は dict with keys:
      'species_idx'  : int  変化させた種のインデックス (0=x1,1=x2,2=y1,3=y2)
      'f_sweep'      : ndarray, shape (n_sens,)
      'mean_yield'   : ndarray, shape (n_sens,)  合計平均漁獲量
      'per_species'  : ndarray, shape (4, n_sens)  種別平均漁獲量
    """
    f_star = np.asarray(f_star, dtype=float)
    f_sweep = np.linspace(F_MIN, F_MAX, n_sens)
    results = []

    for species_idx in range(4):
        mean_yields = np.full(n_sens, np.nan)
        per_species = np.full((4, n_sens), np.nan)

        for j, fi in enumerate(f_sweep):
            f_vec = f_star.copy()
            f_vec[species_idx] = fi
            res = average_yield(f_vec, params_norm, means, model_str, T, X0_norm, n_eval=n_eval)
            if res["success"]:
                mean_yields[j] = res["mean_yield"]
                per_species[:, j] = res["per_species_yield"]

        results.append({
            "species_idx": species_idx,
            "f_sweep":    f_sweep,
            "mean_yield": mean_yields,
            "per_species": per_species,
        })

    return results


# =============================================================================
# 1 年ごとの戦術的 MSY
# =============================================================================

def tactical_msy_per_year(series_slice, params_norm, means, model_str,
                          n_grid=N_GRID, n_eval=N_EVAL_TRAJ):
    """
    レジーム内の各年について T=1 の戦術的 MSY を計算する。

    各年の初期値 = その年の観測資源量（正規化）、パラメータはそのレジームの推定値を使う。
    各年で grid_search_msy を実行し、最適 f* と MSY 値を返す。

    Parameters
    ----------
    series_slice : dict
        slice_series() で切り出したレジームの時系列（years, x1, x2, y1, y2 を含む）。
    params_norm : ndarray
        そのレジームの推定 params_norm
    means : ndarray, shape (4,)
        そのレジームの means
    model_str : str
    n_grid : int
        グリッド点数（n_grid^4 評価 × 年数）
    n_eval : int
        ODE t_eval 点数

    Returns
    -------
    list of dicts, len = (レジームの年数). 各要素は dict with keys:
      'year'               : int  対象年
      'f_star'             : ndarray, shape (4,)
      'msy'                : float  （千トン/年）
      'per_species_at_msy' : ndarray, shape (4,)
      'X0_norm'            : ndarray, shape (4,)  その年の正規化初期値
    """
    years   = series_slice["years"].astype(float)
    obs_abs = np.vstack([series_slice["x1"],
                         series_slice["x2"],
                         series_slice["y1"],
                         series_slice["y2"]])   # shape (4, n_years)

    results = []
    for t_idx, year in enumerate(years):
        X0_abs  = obs_abs[:, t_idx]
        X0_norm = normalize_X0(X0_abs, means)

        res = grid_search_msy(
            params_norm=params_norm,
            means=means,
            model_str=model_str,
            T=1.0,                    # 1 年積分
            X0_norm=X0_norm,
            n_grid=n_grid,
            n_eval=n_eval,
        )
        results.append({
            "year":               int(year),
            "f_star":             res["f_star"],
            "msy":                res["msy"],
            "per_species_at_msy": res["per_species_at_msy"],
            "X0_norm":            X0_norm,
        })

    return results
