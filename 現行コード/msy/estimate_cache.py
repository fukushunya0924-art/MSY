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
import model_constrained
import fixed_params

# -----------------------------------------------------------------------
# 探索設定（3スクリプト共通の唯一の定義）
# -----------------------------------------------------------------------
# マルチスタート×マルチシードはレジーム別に設定する（2026-07-12 の予算感度実験）:
#   NLM は良い解(NRMSE≈0.146)を持つ basin が稀（ランダムスタート数百回に1回未満）で、
#        予算を削ると NRMSE 0.22〜0.25 の浅い局所解に張り付く。よってフル 64×12 を維持。
#   LM  は 32×8=256 回で床(NRMSE≈0.0775)に到達し、フル 64×12(768回, 0.0792)と同等
#        （実測でむしろ僅かに良い）。3倍高速なので採用。実測カーブ:
#        8×4→0.103, 16×8→0.086, 32×8→0.0775, 32×12→0.0775（32×8 以降は頭打ち）。
#        n_starts を 32 まで増やすのが決め手（16 だと 0.086 止まり）。n_seeds は
#        4点とも best_seed=3 で頭打ちだが、単一seed依存を避け冗長性のため 8 とする。
N_STARTS = {"NLM": 64, "LM": 32}
N_SEEDS = {"NLM": 12, "LM": 8}
# レジーム別正則化強度:
#   NLM は 11 点・12 変数で識別性が保てるため正則化不要（Phase4 で λ>0 だと当てはまり悪化）
#   LM  は  8 点・12 変数で識別性が弱いため安定化
REG_LAMBDA = {"NLM": 0.0, "LM": 0.005}

# 推定結果キャッシュファイル（このモジュールと同じ msy/ 配下）
CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "estimates_capacity_ry.pkl")

# -----------------------------------------------------------------------
# 制約推定（10自由変数, Catch-MSY確定値 r_x1/r_x2 のみ固定）の探索設定
# -----------------------------------------------------------------------
# run_msy.py --constrained が使う。model_constrained.estimate_constrained_robust に渡す。
# 制約推定は自由推定の約30倍重い（Phase 8: least_squares 1回 3秒→89秒）ため、まずは
# 診断予算 16×8（Phase 8 実測で NLM+LM 合計 約11分）で配線を検証する。本番予算へ
# 上げる場合はここの数値だけを編集すればよい（キャッシュは署名不一致で自動再計算）。
N_STARTS_C = {"NLM": 16, "LM": 16}
N_SEEDS_C = {"NLM": 8, "LM": 8}
# 制約推定の正則化強度（θ を除く6変数に適用。適用対象の選定は model_constrained 側）。
# 自由版と同じ考え方: NLM は 11点・8自由で識別性が保てるため 0、LM は 8点で弱いため安定化。
REG_LAMBDA_C = {"NLM": 0.0, "LM": 0.005}

def _config_signature():
    """キャッシュ整合性判定に使う探索設定の署名。"""
    return {"n_starts": N_STARTS, "n_seeds": N_SEEDS, "reg_lambda": REG_LAMBDA}


# 制約モデルの版数タグ。自由変数の設計を変えたら必ず上げる（旧キャッシュを自動無効化）。
#   v1: 8自由変数（r_x1,r_x2,S1,S2 固定 + theta 配分）… 廃止
#   v2: 10自由変数（r_x1,r_x2 のみ固定, C1,D1,C2,D2 を自由推定）… 現行（2026-07-14）
_CONSTRAINED_MODEL_VERSION = "v2_rx_only_10free"

# 制約推定の結果キャッシュ（自由版 CACHE_FILE とは別ファイル。相互に上書きしない）。
# ファイル名に _CONSTRAINED_MODEL_VERSION を含めることで、モデル版数を上げた際に
# 旧版のキャッシュファイルが新版で上書きされて物理パラメータが消失する事故を防ぐ
# （2026-07-14の v1→v2 移行で旧8変数版の推定結果が失われた反省, docs/research_log.md Phase 13）。
CACHE_FILE_CONSTRAINED = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    f"estimates_capacity_ry_constrained_{_CONSTRAINED_MODEL_VERSION}.pkl")


def _config_signature_constrained():
    """制約版キャッシュの署名。探索設定に加え、固定値（r_x1,r_x2）とモデル版数を含める。

    自由版 _config_signature は探索設定のみで入力データ変更を検知できないが、
    制約版は fixed_params.py の r_x1,r_x2 を変えたら結果が変わるので署名に入れる。
    S1,S2 は現行の10変数モデルでは使わないため署名に含めない。
    さらにモデル版数タグを含めることで、旧 v1（8変数）キャッシュとは必ず
    署名不一致となり自動的に再推定される。
    → fixed_params.py（r_x1/r_x2）を書き換えると load_estimates_constrained が
      自動で None を返し、run_msy.py --constrained が再推定する。
    """
    fx = fixed_params.get_point()
    return {"n_starts_c": N_STARTS_C, "n_seeds_c": N_SEEDS_C,
            "reg_lambda_c": REG_LAMBDA_C,
            "fixed_rx": {"r_x1": fx["r_x1"], "r_x2": fx["r_x2"]},
            "model_version": _CONSTRAINED_MODEL_VERSION}


def estimate_regime(sl, regime_name):
    """1レジームをレジーム別設定で推定する（キャッシュを介さない単発推定）。"""
    return estimate_robust(sl, n_starts=N_STARTS[regime_name],
                           reg_lambda=REG_LAMBDA[regime_name],
                           n_seeds=N_SEEDS[regime_name], seed0=0)


def estimate_regime_constrained(sl, regime_name):
    """1レジームを制約付き（8自由変数, Catch-MSY確定値固定）で推定する単発推定。

    固定値は fixed=None として model_constrained 側の既定（fixed_params.get_point()）を
    使わせるので、_config_signature_constrained と同じ単一の真実の源から取られる。
    返り値は model.estimate() と同一形状の dict（＋ fixed, free_names, params_free）。
    """
    return model_constrained.estimate_constrained_robust(
        sl, n_starts=N_STARTS_C[regime_name],
        reg_lambda=REG_LAMBDA_C[regime_name],
        n_seeds=N_SEEDS_C[regime_name], seed0=0)


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


def save_estimates_constrained(est_results):
    """制約推定結果を制約版の署名（探索設定＋固定値4つ）付きで保存する。"""
    payload = {"est_results": est_results, **_config_signature_constrained()}
    with open(CACHE_FILE_CONSTRAINED, "wb") as f:
        pickle.dump(payload, f)
    return CACHE_FILE_CONSTRAINED


def load_estimates_constrained():
    """
    探索設定（N_STARTS_C/N_SEEDS_C/REG_LAMBDA_C）と固定値4つ（fixed_params.get_point()）が
    現在と一致する制約版キャッシュを返す。ファイルが無い/署名不一致なら None
    （呼び出し側は新規推定にフォールバック）。fixed_params.py を書き換えると
    署名の "fixed" が食い違うので自動で None になり、再推定される。
    """
    if not os.path.exists(CACHE_FILE_CONSTRAINED):
        return None
    with open(CACHE_FILE_CONSTRAINED, "rb") as f:
        cache = pickle.load(f)
    sig = _config_signature_constrained()
    if any(cache.get(k) != v for k, v in sig.items()):
        return None
    return cache["est_results"]
