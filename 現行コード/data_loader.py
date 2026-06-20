"""
データ読み込みと前処理（6.8 モデル用）。

6.8/ の2スクリプトが扱う4魚種を「千トン単位」に統一して返す。
  被食者(x): マイワシ x1, カタクチイワシ x2
  捕食者(y): ヤリイカ y1, スルメイカ y2

元コードと同じスケール換算・漁獲圧計算を踏襲しつつ、
  - data/ をファイル位置から自動探索（実行ディレクトリ非依存）
  - 数値列の強制数値化（"#DIV/0!" などを NaN 化）
を追加して落ちにくくしている。
"""
import os
import numpy as np
import pandas as pd


def _find_data_dir():
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


def _to_numeric(df):
    for col in df.columns:
        if col not in ("年", "黒潮大蛇行の有無"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "年" in df.columns:
        df["年"] = pd.to_numeric(df["年"], errors="coerce")
    return df


def load_clean_dataframe():
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


def get_series(df_clean):
    """資源量(千トン)・漁獲圧を辞書で返す。元コードと同じ換算。"""
    x1 = df_clean["資源量（万トン）"].values.astype(float) * 10.0   # マイワシ
    x2 = df_clean["資源量_anchovy"].values.astype(float)           # カタクチ（千トン）
    y1 = df_clean["資源量_yariika"].values.astype(float) / 1000.0  # ヤリイカ（トン→千トン）
    y2 = df_clean["資源量_squid"].values.astype(float)             # スルメイカ（千トン）

    cx1 = df_clean["漁獲量（万トン）"].values.astype(float) * 10.0
    cx2 = df_clean["漁獲量_anchovy"].values.astype(float) / 1000.0
    cy1 = df_clean["漁獲量_yariika"].values.astype(float) / 1000.0
    cy2 = df_clean["漁獲量_squid"].values.astype(float)

    fx1 = np.clip(cx1 / x1, 0.0, 0.95)
    fx2 = np.clip(cx2 / x2, 0.0, 0.95)
    fy1 = np.clip(cy1 / y1, 0.0, 0.95)
    fy2 = np.clip(cy2 / y2, 0.0, 0.95)

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
