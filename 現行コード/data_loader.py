"""
データ読み込みと前処理（旧構成: マイワシ+カタクチイワシ / ヤリイカ+スルメイカ）。

4魚種の資源評価CSVを「千トン単位」に統一して返す。
  被食者(x): マイワシ x1, カタクチイワシ x2
  捕食者(y): ヤリイカ y1, スルメイカ y2

- data/ をファイル位置から自動探索（実行ディレクトリ非依存）
- 数値列の強制数値化（"#DIV/0!" などを NaN 化）
を追加して落ちにくくしている。

注意: 現行の種構成（マイワシ+ウルメイワシ / ブリ+サワラ, Phase 7d）を使う
`現行コード/msy/` の各スクリプトは、この階層の data_loader.py ではなく
`現行コード/msy/data_loader.py`（ブリ/サワラ・ウルメイワシ版）を
sys.path 優先で読み込む。このファイルは旧版（`旧版/legacy_multi_model/`）
互換のために残っている。
"""
import os
import numpy as np
import pandas as pd

# 漁獲圧 f = catch/biomass の上限クリップ値。
# 生物量に対する漁獲量比が1.0に近い/超えると ODE の増殖項が破綻するための安全弁。
F_CLIP_MAX = 0.95

# 万トン→千トン、トン→千トンのスケール換算係数
_MAN_TON_TO_KILO_TON = 10.0
_TON_TO_KILO_TON = 1.0 / 1000.0


def _find_data_dir() -> str:
    """data/ ディレクトリを実行位置に依存せず探索する。"""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "..", "data"),
        os.path.join(here, "data"),
        os.path.join(os.getcwd(), "data"),
        os.path.join(os.getcwd(), "..", "data"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return os.path.abspath(c)
    raise FileNotFoundError("data/ フォルダが見つかりません: " + repr(candidates))


_FILES = {
    "sardine": "マイワシ時系列データ_資源量・漁獲量・漁獲係数 - マイワシ時系列データ_資源量・漁獲量・漁獲係数.csv",
    "anchovy": "カタクチイワシ太平洋時系列データ - カタクチイワシ太平洋時系列データ.csv",
    "yariika": "ヤリイカ太平洋時系列データ - ヤリイカ太平洋時系列データ.csv",
    "squid":   "スルメイカ秋季 資源量・漁獲量時系列データ - スルメイカ秋季 資源量・漁獲量時系列データ.csv",
}


def _to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """「年」「黒潮大蛇行の有無」以外の列を数値化する（"#DIV/0!" 等は NaN 化）。"""
    for col in df.columns:
        if col not in ("年", "黒潮大蛇行の有無"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "年" in df.columns:
        df["年"] = pd.to_numeric(df["年"], errors="coerce")
    return df


def load_clean_dataframe() -> pd.DataFrame:
    """4魚種のCSVを読み込み、年で内部結合してNaN行を除去したDataFrameを返す。"""
    data_dir = _find_data_dir()

    def rd(key):
        return pd.read_csv(os.path.join(data_dir, _FILES[key]))

    df_sardine = _to_numeric(rd("sardine"))
    df_anchovy = _to_numeric(rd("anchovy"))
    df_yariika = _to_numeric(rd("yariika"))
    df_squid = _to_numeric(rd("squid").rename(columns={"漁期年": "年"}))

    df_merged = (
        df_sardine[["年", "資源量（万トン）", "漁獲量（万トン）"]]
        .merge(
            df_anchovy[["年", "資源量（千トン）", "漁獲量（トン）"]].rename(
                columns={"資源量（千トン）": "資源量_anchovy", "漁獲量（トン）": "漁獲量_anchovy"}
            ), on="年")
        .merge(
            df_yariika[["年", "資源量（トン）", "漁獲量（トン）"]].rename(
                columns={"資源量（トン）": "資源量_yariika", "漁獲量（トン）": "漁獲量_yariika"}
            ), on="年")
        .merge(
            df_squid[["年", "資源量（千トン）", "漁獲量（千トン）"]].rename(
                columns={"資源量（千トン）": "資源量_squid", "漁獲量（千トン）": "漁獲量_squid"}
            ), on="年")
    )
    return df_merged.dropna().sort_values("年").reset_index(drop=True)


def get_series(df_clean: pd.DataFrame) -> dict:
    """資源量(千トン)・漁獲圧を辞書で返す。元コードと同じ換算。

    漁獲圧 f_i = catch_i / biomass_i を [0, F_CLIP_MAX] にクリップする
    （生物量に対し漁獲量が過大な年でも ODE の自然増殖項が負に潰れないための安全弁）。
    """
    x1 = df_clean["資源量（万トン）"].values.astype(float) * _MAN_TON_TO_KILO_TON  # マイワシ
    x2 = df_clean["資源量_anchovy"].values.astype(float)                          # カタクチ（千トン）
    y1 = df_clean["資源量_yariika"].values.astype(float) * _TON_TO_KILO_TON        # ヤリイカ（トン→千トン）
    y2 = df_clean["資源量_squid"].values.astype(float)                            # スルメイカ（千トン）

    cx1 = df_clean["漁獲量（万トン）"].values.astype(float) * _MAN_TON_TO_KILO_TON
    cx2 = df_clean["漁獲量_anchovy"].values.astype(float) * _TON_TO_KILO_TON
    cy1 = df_clean["漁獲量_yariika"].values.astype(float) * _TON_TO_KILO_TON
    cy2 = df_clean["漁獲量_squid"].values.astype(float)

    fx1 = np.clip(cx1 / x1, 0.0, F_CLIP_MAX)
    fx2 = np.clip(cx2 / x2, 0.0, F_CLIP_MAX)
    fy1 = np.clip(cy1 / y1, 0.0, F_CLIP_MAX)
    fy2 = np.clip(cy2 / y2, 0.0, F_CLIP_MAX)

    return {
        "years": df_clean["年"].values.astype(float),
        "x1": x1, "x2": x2, "y1": y1, "y2": y2,
        "fx1": fx1, "fx2": fx2, "fy1": fy1, "fy2": fy2,
    }


SPECIES_LABELS = ["マイワシ (x1)", "カタクチイワシ (x2)", "ヤリイカ (y1)", "スルメイカ (y2)"]
KEYS = ["x1", "x2", "y1", "y2"]


if __name__ == "__main__":
    df = load_clean_dataframe()
    s = get_series(df)
    print("マージ後の年数:", len(df), " 年範囲:", int(s["years"].min()), "-", int(s["years"].max()))
    for k, lab in zip(KEYS, SPECIES_LABELS):
        print(f"  {lab:18s}: {s[k].min():9.1f} 〜 {s[k].max():9.1f}  (平均 {s[k].mean():9.1f})")
