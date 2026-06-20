"""
データ読み込み（1位の組み合わせ用）。

スキャン結果で最も ODE 適合しやすい（平均CV最小）組み合わせ:
  被食者(x): ヤリイカ x1, スルメイカ x2    （MID: 中型捕食者＝イカ類）
  捕食者(y): ブリ y1, サワラ y2            （HIGH: 大型魚食魚）

栄養段階の制約を満たす: x は同一段階(MID)、y は同一段階(HIGH)、y は x より高次。
生態: 大型魚(ブリ・サワラ)がイカ類(ヤリイカ・スルメイカ)を捕食。

親フォルダ improved/ の model.estimate と互換の series dict を返す。
"""
import os
import numpy as np
import pandas as pd


def _data_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    for c in [os.path.join(here, "..", "..", "data"),
              os.path.join(here, "..", "data"),
              os.path.join(os.getcwd(), "data")]:
        if os.path.isdir(c):
            return os.path.abspath(c)
    raise FileNotFoundError("data/ が見つかりません")


# 魚種レジストリ: ファイル名・列・千トン換算スケール
_REG = {
    "ヤリイカ": dict(
        file="ヤリイカ太平洋時系列データ - ヤリイカ太平洋時系列データ.csv",
        yr="年", bio="資源量（トン）", catch="漁獲量（トン）",
        bio_scale=1/1000, catch_scale=1/1000),
    "スルメイカ": dict(
        file="スルメイカ秋季 資源量・漁獲量時系列データ - スルメイカ秋季 資源量・漁獲量時系列データ.csv",
        yr="漁期年", bio="資源量（千トン）", catch="漁獲量（千トン）",
        bio_scale=1.0, catch_scale=1.0),
    "ブリ": dict(
        file="ブリ時系列データ_資源量・漁獲量・漁獲係数 - ブリ時系列データ_資源量・漁獲量・漁獲係数.csv",
        yr="年", bio="資源量（トン）", catch="漁獲量（トン）",
        bio_scale=1/1000, catch_scale=1/1000),
    "サワラ": dict(
        file="サワラ時系列データ_資源量・漁獲量・漁獲係数 - サワラ時系列データ_資源量・漁獲量・漁獲係数.csv",
        yr="年", bio="資源量（トン）", catch="漁獲量（トン）",
        bio_scale=1/1000, catch_scale=1/1000),
}

# この組み合わせの割り当て
ASSIGN = {"x1": "ヤリイカ", "x2": "スルメイカ", "y1": "ブリ", "y2": "サワラ"}
SPECIES_LABELS = ["ヤリイカ (x1)", "スルメイカ (x2)", "ブリ (y1)", "サワラ (y2)"]
KEYS = ["x1", "x2", "y1", "y2"]


def _load_one(name):
    cfg = _REG[name]
    df = pd.read_csv(os.path.join(_data_dir(), cfg["file"]))
    for c in df.columns:
        if c not in ("年", "漁期年", "黒潮大蛇行の有無"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.rename(columns={cfg["yr"]: "年"})
    df["年"] = pd.to_numeric(df["年"], errors="coerce")
    bio = df[cfg["bio"]].astype(float) * cfg["bio_scale"]
    catch = df[cfg["catch"]].astype(float) * cfg["catch_scale"]
    out = pd.DataFrame({"年": df["年"], f"bio_{name}": bio, f"catch_{name}": catch})
    return out.dropna()


def load_clean_dataframe():
    dfs = [_load_one(ASSIGN[k]) for k in KEYS]
    merged = dfs[0]
    for d in dfs[1:]:
        merged = merged.merge(d, on="年")
    return merged.dropna().sort_values("年").reset_index(drop=True)


def get_series(df_clean):
    s = {"years": df_clean["年"].values.astype(float)}
    for k in KEYS:
        name = ASSIGN[k]
        bio = df_clean[f"bio_{name}"].values.astype(float)
        catch = df_clean[f"catch_{name}"].values.astype(float)
        s[k] = bio
        s["f" + k] = np.clip(catch / bio, 0.0, 0.95)
    return s


if __name__ == "__main__":
    df = load_clean_dataframe()
    s = get_series(df)
    print("マージ後年数:", len(df), " 年範囲:", int(s["years"].min()), "-", int(s["years"].max()))
    for k, lab in zip(KEYS, SPECIES_LABELS):
        print(f"  {lab:16s}: {s[k].min():9.1f} 〜 {s[k].max():9.1f}  (平均 {s[k].mean():9.1f})")
