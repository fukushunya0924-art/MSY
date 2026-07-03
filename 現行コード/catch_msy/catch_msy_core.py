"""
連続時間 Catch-MSY の計算エンジン。

古典的な Catch-MSY (Martell & Froese 2013) は差分方程式
    B_{t+1} = B_t + r·B_t·(1 − B_t/K) − C_t
を使うが、本実装はユーザー要望により連続時間 ODE
    dB/dt = r·B·(1 − B/K) − C(t)
を scipy.integrate.solve_ivp で解く。C(t) は年次漁獲量の線形補間。

リサンプリング法（連続時間版）:
  1. 事前分布 (r, K, B0) を対数一様サンプル
  2. 各ペアで ODE を積分
  3. 生存条件でフィルタ:
       - 全期間で B(t) > 0（崩壊しない）
       - 終端枯渇度 B(T)/K が指定レンジに収まる
     （連続 Schaefer では B0≤K なら B(t)≤K は自動満足。B=K で dB/dt=−C(t)≤0
       のため上から K を超えられない。よって棄却に効くのは B≤0 のみ。）
  4. 生存 (r, K) から r の幾何平均・分位点、MSY = r·K/4 を得る

速度設計（崩壊の早期終了 → ベクトル化）:
  N 万ペアを 1 ペアずつ solve_ivp で回すと Python オーバーヘッドが支配的で
  30,000 ペアに ~15 分かかる。そこで N 成分を「独立だが連立した 1 本の ODE 系」
  として **1 回の solve_ivp でまとめて積分**（integrate_batch）。~0.5 秒で済む。
    - 崩壊の扱い: 右辺で B を max(B,0) にクランプ（吸収境界）。崩壊したら
      dB/dt=−C(t)<0 で負に留まり回復しない → min_t B(t) ≤ 0 で崩壊判定できる。
      これは 1 ペア版の solve_ivp(events, terminal=True) による早期棄却と
      **同じ棄却セマンティクス**を、成分ごとにベクトル化したもの。
  参考実装として events 版の 1 ペア積分器 integrate_one も残す（検証用）。
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


# -----------------------------------------------------------------------
# 事前分布のプリセット（Froese & Martell 2013 準拠）
# -----------------------------------------------------------------------
# resilience → r 事前レンジ（log-uniform）
RESILIENCE_R = {
    "high":     (0.6, 1.5),
    "medium":   (0.2, 1.0),
    "low":      (0.05, 0.5),
    "very_low": (0.015, 0.1),
}

# 魚種ごとの resilience（run_catch_msy.py と共有）
SPECIES_RESILIENCE = {
    "sardine": "high",     # マイワシ
    "anchovy": "high",     # カタクチイワシ
    "buri":    "medium",   # ブリ
    "sawara":  "medium",   # サワラ
    "mackerel": "medium",  # サバ類
    "squid":    "high",    # スルメイカ
}

# K 事前レンジ = [K_MULT_LO × max(C), K_MULT_HI × max(C)]（log-uniform）
K_MULT_LO = 1.0
K_MULT_HI = 60.0

# 初期枯渇度 B0/K 事前レンジ（log-uniform）
B0_FRAC_RANGE = (0.5, 0.9)

# 既定サンプル数・終端判定レンジ
N_SAMPLES = 30_000
DEFAULT_FINAL_RANGE = (0.2, 0.6)

# 積分の時間解像度（年あたり評価点数）。崩壊検知の安全マージン用。
N_PER_YEAR = 6


def _loguniform(lo, hi, n, rng):
    """[lo, hi] の対数一様サンプル。"""
    return np.exp(rng.uniform(np.log(lo), np.log(hi), size=n))


def make_catch_interp(years, catch):
    """
    年次漁獲量を線形補間した連続関数 C(t) を返す。
    t は「最初の年を 0 とした経過年」。範囲外は端点で一定（外挿しない）。
    """
    t = years - years[0]
    return interp1d(t, catch, kind="linear",
                    bounds_error=False, fill_value=(catch[0], catch[-1]))


def integrate_one(r, K, B0, C_func, T, rtol=1e-6, atol=1e-8):
    """
    1 ペアぶんの連続 Schaefer を積分。
    戻り値 (survived, B_end):
      survived=False, B_end=None … B≤0 で崩壊（events 早期終了）
      survived=True,  B_end=B(T) … 完走
    """
    def rhs(t, B):
        b = B[0]
        return [r * b * (1.0 - b / K) - C_func(t)]

    def crash(t, B):
        # B が 0 に到達したら terminal 停止
        return B[0]
    crash.terminal = True
    crash.direction = -1  # 減少方向に 0 を横切るときだけ

    sol = solve_ivp(rhs, (0.0, T), [B0], method="RK45",
                    events=crash, rtol=rtol, atol=atol, dense_output=False)

    if sol.t_events[0].size > 0:
        # 崩壊（早期終了）
        return False, None
    if sol.status != 0:
        return False, None
    B_end = sol.y[0, -1]
    if B_end <= 0:
        return False, None
    return True, B_end


def integrate_batch(r, K, B0, years, catch, n_per_year=N_PER_YEAR,
                    rtol=1e-6, atol=1e-6):
    """
    N 成分を 1 回の solve_ivp でまとめて積分（ベクトル化）。

    各成分 i は独立な連続 Schaefer:
        dB_i/dt = r_i·B_i·(1 − B_i/K_i) − C(t)
    右辺で B を max(B,0) にクランプ（吸収境界）することで、崩壊した成分が
    負に発散せず、min_t B_i(t) ≤ 0 で崩壊判定できる。

    戻り値 (B_min, B_end):
      B_min : 各成分の全期間最小値（≤0 なら崩壊）
      B_end : 各成分の終端値 B_i(T)
    """
    r = np.asarray(r, dtype=float)
    K = np.asarray(K, dtype=float)
    B0 = np.asarray(B0, dtype=float)
    tgrid = (years - years[0]).astype(float)
    T = float(tgrid[-1])

    def rhs(t, B):
        Bc = np.maximum(B, 0.0)
        return r * Bc * (1.0 - Bc / K) - np.interp(t, tgrid, catch)

    t_eval = np.linspace(0.0, T, int(round(T)) * n_per_year + 1)
    sol = solve_ivp(rhs, (0.0, T), B0, method="RK45", t_eval=t_eval,
                    rtol=rtol, atol=atol)
    Y = sol.y  # (N, len(t_eval))
    return Y.min(axis=1), Y[:, -1]


def run_catch_msy(years, catch, resilience,
                  final_range=DEFAULT_FINAL_RANGE,
                  n_samples=N_SAMPLES,
                  k_mult=(K_MULT_LO, K_MULT_HI),
                  b0_frac_range=B0_FRAC_RANGE,
                  seed=0):
    """
    連続時間 Catch-MSY を1魚種に対して実行。

    引数:
      years, catch : 年次時系列（catch は千トン）
      resilience   : "high"/"medium"/"low"/"very_low"
      final_range  : 終端枯渇度 B(T)/K の許容レンジ (lo, hi)
      n_samples    : (r,K) サンプル数

    戻り値 dict:
      r_prior, k_prior         : 事前レンジ
      n_viable                 : 生存ペア数
      r_geomean, r_lo, r_hi    : 生存 r の幾何平均・25/75%点（無ければ NaN）
      k_geomean                : 生存 K の幾何平均
      msy_geomean, msy_lo, msy_hi : MSY=rK/4 の幾何平均・25/75%点
      r_viable, k_viable, msy_viable : 生存ペアの配列（後段の描画用）
    """
    rng = np.random.default_rng(seed)
    r_lo, r_hi = RESILIENCE_R[resilience]
    Cmax = float(np.max(catch))
    K_lo, K_hi = k_mult[0] * Cmax, k_mult[1] * Cmax
    T = float(years[-1] - years[0])

    # 事前サンプル
    r_s = _loguniform(r_lo, r_hi, n_samples, rng)
    K_s = _loguniform(K_lo, K_hi, n_samples, rng)
    b0f = _loguniform(b0_frac_range[0], b0_frac_range[1], n_samples, rng)
    B0_s = b0f * K_s

    # ベクトル化バッチ積分（1 回の solve_ivp で全ペア）
    B_min, B_end = integrate_batch(r_s, K_s, B0_s, years, catch)

    fmin, fmax = final_range
    survived = B_min > 0.0            # 崩壊しなかった
    depl = B_end / K_s
    viable = survived & (depl >= fmin) & (depl <= fmax)

    r_ok = r_s[viable]
    k_ok = K_s[viable]
    msy_ok = r_ok * k_ok / 4.0

    def _gmean(a):
        return float(np.exp(np.mean(np.log(a)))) if a.size else float("nan")

    def _q(a, p):
        return float(np.quantile(a, p)) if a.size else float("nan")

    return {
        "r_prior": (r_lo, r_hi),
        "k_prior": (K_lo, K_hi),
        "final_range": final_range,
        "T": T,
        "n_samples": n_samples,
        "n_viable": int(r_ok.size),
        "r_geomean": _gmean(r_ok),
        "r_lo": _q(r_ok, 0.25),
        "r_hi": _q(r_ok, 0.75),
        "k_geomean": _gmean(k_ok),
        "msy_geomean": _gmean(msy_ok),
        "msy_lo": _q(msy_ok, 0.25),
        "msy_hi": _q(msy_ok, 0.75),
        "r_viable": r_ok,
        "k_viable": k_ok,
        "msy_viable": msy_ok,
    }
