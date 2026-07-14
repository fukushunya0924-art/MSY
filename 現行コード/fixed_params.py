"""
Catch-MSY で確定した外挿パラメータの単一の真実の源。

capacity_ry モデル（12変数）のうち、マアジ・ウルメイワシの自然増殖率 r_x と
ブリ・サワラの捕食→捕食者変換効率の和 S(=c+d) の4値は、ODE推定とは別手法の
Catch-MSY で NLM/LM 共通の値としてあらかじめ推定済みである。
`model_constrained.py` はこれらを固定値として外挿し、残り8自由変数
（r_y1, r_y2, L11, L12, L21, L22, theta1, theta2）だけを ODE 推定する。

データ源:
  - マアジ r_x1  : FRA資源評価（太平洋系群）catch 1982-2024（Phase 11,
                  2026-07-14）。ODE推定（msy/data_loader.py）と同一の
                  biomass/catchデータ源のため「Catch-MSYとODEで漁獲量データが
                  異なる」という問題が最初から生じない。標準レンジ[0.01,0.4]
                  で解ける（終端/max=0.29, n=22）ので、旧マイワシで必要だった
                  レンジ例外[0.6,0.95]は不要になった。
  - ブリ   S1    : FRA資源評価 catch 1994-2024（Phase 10, catch源をODEと整合）
  - サワラ  S2    : FRA資源評価 catch 1987-2024（Phase 10, catch源をODEと整合）
  - ウルメ  r_x2   : e-stat 太平洋12県 catch（FRA に絶対 catch が無く、ODE側も
                      ウルメの catch は e-stat 12県を使うため、両手法で整合済み）

キー対応（capacity_ry の names と対応させると r_x1, r_x2 はそのまま、
S1, S2 は c1+d1, c2+d2 に相当）:
  r_x1  : マアジの自然増殖率 (1/年)
  r_x2  : ウルメイワシの自然増殖率 (1/年)
  S1    : ブリの c1+d1（マアジ・ウルメイワシ由来の変換効率の和）
  S2    : サワラの c2+d2（同上）

これら4値はレジーム（NLM/LM）に依らず共通の値として扱う。
"""
from typing import Dict, Tuple

# 点推定値（1/年）。キーは model_constrained.reconstruct_params の fixed 引数と対応。
_POINT: Dict[str, float] = {
    "r_x1": 0.228,  # マアジ。FRA catch 1982-2024。終端レンジ[0.01,0.4]（標準ルール通り）
    "r_x2": 0.739,  # ウルメイワシ。e-stat太平洋12県 catch（ODEと整合）。終端レンジ[0.01,0.4]（標準ルール通り）
    "S1":   0.395,  # ブリ c1+d1。FRA catch 1994-2024。終端レンジ[0.3,0.7]（標準ルール通り）
    "S2":   0.260,  # サワラ c2+d2。FRA catch 1987-2024。終端レンジ[0.01,0.4]（標準ルール通り）
}

# 信頼区間 (lo, hi)。Catch-MSY 生存ペアの 25-75% 点。
_CI: Dict[str, Tuple[float, float]] = {
    "r_x1": (0.206, 0.246),
    "r_x2": (0.642, 0.824),
    "S1":   (0.268, 0.569),
    "S2":   (0.220, 0.295),
}


def get_point() -> Dict[str, float]:
    """固定パラメータの点推定値を dict で返す（キー: r_x1, r_x2, S1, S2）。"""
    return dict(_POINT)


def get_ci(name: str) -> Tuple[float, float]:
    """指定パラメータ名（r_x1, r_x2, S1, S2 のいずれか）の信頼区間 (lo, hi) を返す。"""
    return _CI[name]
