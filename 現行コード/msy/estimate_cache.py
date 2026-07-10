"""
estimate_robust の探索設定とその結果キャッシュを msy/ 配下で共有するユーティリティ。

run_msy.py・plot_fit_smooth.py・diagnose_iwashi.py はいずれも同一設定
（n_starts / n_seeds / reg_lambda）で estimate_robust を呼ぶ。推定は数時間かかるため、
一度計算した結果を pickle に保存し、設定が一致する限り再利用する。

役割分担:
  - run_msy.py（生産者）: 常に新規推定 → save_estimates で保存（MSY確定値は最新データ反映）
  - plot_fit_smooth.py / diagnose_iwashi.py（消費者）: load_estimates を試し、無ければ新規推定

⚠ キャッシュ整合性は探索設定（n_starts/n_seeds/reg_lambda）のみで判定し、入力データの
  変更は検知しない。データ・モデルを変えた場合は run_msy.py を実行してキャッシュを更新すること。
"""
import os
import pickle

from model import estimate_robust

# -----------------------------------------------------------------------
# 探索設定（3スクリプト共通の唯一の定義）
# -----------------------------------------------------------------------
# n_starts=64 × n_seeds=12 の総コスト最小解を採用する（Phase 7d）。
# 単一シード・n_starts=40 では局所解に落ちることが診断で判明したため。
N_STARTS = 64
N_SEEDS = 12
# レジーム別正則化強度:
#   NLM は 11 点・12 変数で識別性が保てるため正則化不要（Phase4 で λ>0 だと当てはまり悪化）
#   LM  は  8 点・12 変数で識別性が弱いため安定化
REG_LAMBDA = {"NLM": 0.0, "LM": 0.005}

# 推定結果キャッシュファイル（このモジュールと同じ msy/ 配下）
CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "estimates_capacity_ry.pkl")


def _config_signature():
    """キャッシュ整合性判定に使う探索設定の署名。"""
    return {"n_starts": N_STARTS, "n_seeds": N_SEEDS, "reg_lambda": REG_LAMBDA}


def estimate_regime(sl, regime_name):
    """1レジームを共通設定で推定する（キャッシュを介さない単発推定）。"""
    return estimate_robust(sl, n_starts=N_STARTS, reg_lambda=REG_LAMBDA[regime_name],
                           n_seeds=N_SEEDS, seed0=0)


def save_estimates(est_results):
    """est_results（{regime: estimate_robust の返り値}）を設定署名付きで保存する。"""
    payload = {"est_results": est_results, **_config_signature()}
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(payload, f)
    return CACHE_FILE


def load_estimates():
    """
    探索設定が現在の N_STARTS/N_SEEDS/REG_LAMBDA と一致するキャッシュを返す。
    ファイルが無い、または設定が不一致なら None（呼び出し側は新規推定にフォールバック）。
    """
    if not os.path.exists(CACHE_FILE):
        return None
    with open(CACHE_FILE, "rb") as f:
        cache = pickle.load(f)
    sig = _config_signature()
    if any(cache.get(k) != v for k, v in sig.items()):
        return None
    return cache["est_results"]
