"""
データ読み込み（2026-06-23: y_2をサワラ→スケトウダラに変更）。

組み合わせ:
  被食者(x): マイワシ x1, カタクチイワシ x2  （LOW: 小型浮魚）
  捕食者(y): ブリ y1, スケトウダラ y2        （HIGH: 大型魚食魚）

4位組み合わせ（マイワシ+カタクチ/ブリ+サワラ）から y2 だけをスケトウダラに換えた版。
スケトウダラのデータは data/スケトウダラ時系列データ_資源量・漁獲量・漁獲係数.csv
（単位: 万トン）を使用。
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


_REG = {
    "マイワシ": dict(
        file="マイワシ時系列データ_資源量・漁獲量・漁獲係数 - マイワシ時系列データ_資源量・漁獲量・漁獲係数.csv",
        yr="年", bio="資源量（万トン）", catch="漁獲量（万トン）",
        bio_scale=10.0, catch_scale=10.0),
    "カタクチイワシ": dict(
        file="カタクチイワシ太平洋時系列データ - カタクチイワシ太平洋時系列データ.csv",
        yr="年", bio="資源量（千トン）", catch="漁獲量（トン）",
        bio_scale=1.0, catch_scale=1/1000),
    "ブリ": dict(
        file="ブリ時系列データ_資源量・漁獲量・漁獲係数 - ブリ時系列データ_資源量・漁獲量・漁獲係数.csv",
        yr="年", bio="資源量（トン）", catch="漁獲量（トン）",
        bio_scale=1/1000, catch_scale=1/1000),
    "スケトウダラ": dict(
        file="スケトウダラ時系列データ_資源量・漁獲量・漁獲係数.csv",
        yr="漁期年", bio="資源量（万トン）", catch="漁獲量（万トン）",
        bio_scale=10.0, catch_scale=10.0),
}

ASSIGN = {"x1": "マイワシ", "x2": "カタクチイワシ", "y1": "ブリ", "y2": "スケトウダラ"}
SPECIES_LABELS = ["マイワシ (x1)", "カタクチイワシ (x2)", "ブリ (y1)", "スケトウダラ (y2)"]
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
        print(f"  {lab:25s}: {s[k].min():9.1f} 〜 {s[k].max():9.1f}  (平均 {s[k].mean():9.1f} 千トン)")
