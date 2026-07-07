"""
データ読み込み（4位の組み合わせ用）。

スキャン結果4位（平均CV=0.356）:
  被食者(x): マイワシ x1, カタクチイワシ x2  （LOW: 小型浮魚）
  捕食者(y): ブリ y1, サワラ y2              （HIGH: 大型魚食魚）

1位（ヤリイカ+スルメイカ/ブリ+サワラ）との違いは被食者側のみ。
捕食者(ブリ・サワラ)を共通にすることで「被食者の選択効果」を切り出せる。
また被食者(マイワシ・カタクチ)の生物量はブリ・サワラより大きく、
生物量の順位逆転（捕食者>被食者）が起きないため c1 異常値も解消される見込み。
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
    "サワラ": dict(
        file="サワラ時系列データ_資源量・漁獲量・漁獲係数 - サワラ時系列データ_資源量・漁獲量・漁獲係数.csv",
        yr="年", bio="資源量（トン）", catch="漁獲量（トン）",
        bio_scale=1/1000, catch_scale=1/1000),
}

# ウルメイワシは資源評価CSV（絶対資源量）が存在しないため、
# 別の2ファイルを合成して bio / catch を作る（_load_urume 参照）。
#   資源量指標値（相対値, 平均1） × URUME_MEAN_BIOMASS → 絶対資源量（千トン）
#   URUME_MEAN_BIOMASS = 172.0 千トン
#     FRA余剰生産モデルの K×bkfrac / 指標値(1979) を3モデル平均した値。
#     相対指標値を絶対資源量スケールへ換算するための係数。
URUME_MEAN_BIOMASS = 172.0
_URUME_INDEX_FILE = "ウルメイワシ資源量指標値_FRA資源評価2025.csv"
_URUME_CATCH_FILE = "estat_海面漁業魚種別漁獲量_太平洋12県_1956-2023.csv"
_URUME_CATCH_COL = "ウルメイワシ"

ASSIGN = {"x1": "マイワシ", "x2": "ウルメイワシ", "y1": "ブリ", "y2": "サワラ"}
SPECIES_LABELS = ["マイワシ (x1)", "ウルメイワシ (x2)", "ブリ (y1)", "サワラ (y2)"]
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


def _load_urume():
    """
    ウルメイワシ専用ロード: 資源量指標値CSV × 漁獲量CSV(e-stat太平洋12県) を合成。

    bio_ウルメイワシ（千トン） = 資源量指標値 × URUME_MEAN_BIOMASS   （年 1979-2024）
    catch_ウルメイワシ（千トン） = e-stat太平洋12県「ウルメイワシ」列 ÷1000  （年 1956-2023）

    e-statは2023年までのため、2024年の漁獲量は2023年の値を持ち越す（端点保持）。
    """
    data_dir = _data_dir()

    idx = pd.read_csv(os.path.join(data_dir, _URUME_INDEX_FILE))
    idx["年"] = pd.to_numeric(idx["年"], errors="coerce")
    idx["資源量指標値"] = pd.to_numeric(idx["資源量指標値"], errors="coerce")
    idx = idx.dropna()
    bio = idx[["年"]].copy()
    bio["bio_ウルメイワシ"] = idx["資源量指標値"].astype(float) * URUME_MEAN_BIOMASS

    catch_df = pd.read_csv(os.path.join(data_dir, _URUME_CATCH_FILE), encoding="utf-8-sig")
    catch_df = catch_df.rename(columns={catch_df.columns[0]: "年"})
    catch_df["年"] = pd.to_numeric(catch_df["年"], errors="coerce")
    catch_df[_URUME_CATCH_COL] = pd.to_numeric(catch_df[_URUME_CATCH_COL], errors="coerce")
    catch_df = catch_df.dropna(subset=["年", _URUME_CATCH_COL])
    catch = catch_df[["年"]].copy()
    catch["catch_ウルメイワシ"] = catch_df[_URUME_CATCH_COL].astype(float) / 1000.0

    # 2024年の漁獲量欠測 → 2023年の値を持ち越す（端点保持）
    last_year = int(catch["年"].max())
    if bio["年"].max() > last_year:
        carry = catch.loc[catch["年"] == last_year, "catch_ウルメイワシ"].iloc[0]
        for y in bio.loc[bio["年"] > last_year, "年"]:
            catch = pd.concat(
                [catch, pd.DataFrame({"年": [y], "catch_ウルメイワシ": [carry]})],
                ignore_index=True,
            )

    out = bio.merge(catch, on="年")
    return out.dropna().sort_values("年").reset_index(drop=True)


def load_clean_dataframe():
    dfs = []
    for k in KEYS:
        name = ASSIGN[k]
        dfs.append(_load_urume() if name == "ウルメイワシ" else _load_one(name))
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
        print(f"  {lab:20s}: {s[k].min():9.1f} 〜 {s[k].max():9.1f}  (平均 {s[k].mean():9.1f})")
