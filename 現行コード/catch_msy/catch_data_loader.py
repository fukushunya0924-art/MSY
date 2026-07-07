"""
Catch-MSY 用データローダ（漁獲量のみ・単一種）。

e-stat 海面漁業魚種別漁獲量累年統計（全国, 1956-2024）の整形済 CSV を読み、
魚種ごとに「年 → 漁獲量（千トン）」の時系列を返す。

Catch-MSY は資源量を使わず漁獲量のみを入力とするため、既存の
data_loader.py（資源量・漁獲圧まで扱う推定用）とは切り離した軽量ローダにする。

元 CSV の単位はトン。ODE の数値スケールを既存コード（千トン）に揃えるため
÷1000 して千トンで返す。
"""
import os
import csv

import numpy as np


# 4魚種（被食者2・捕食者2）＋教授候補（サバ類・スルメイカ・スケトウダラ）
# ＋カタクチ代替候補（ウルメイワシ）も読める
SPECIES = {
    "sardine": "マイワシ",
    "anchovy": "カタクチイワシ",
    "buri":    "ブリ",
    "sawara":  "サワラ",
    "mackerel": "サバ類",
    "squid":    "スルメイカ",
    "pollock":  "スケトウダラ",
    "urume":    "ウルメイワシ",
}

# 表示ラベル（図・表用）
SPECIES_LABELS = {
    "sardine": "マイワシ",
    "anchovy": "カタクチイワシ",
    "buri":    "ブリ",
    "sawara":  "サワラ",
    "mackerel": "サバ類",
    "squid":    "スルメイカ",
    "pollock":  "スケトウダラ",
    "urume":    "ウルメイワシ",
}

# 主対象（run_catch_msy.py が既定で回す4種）
# 2026-07-07 確定: 被食者はマイワシ・ウルメイワシ（カタクチ→ウルメ置換）、
# 捕食者はブリ・サワラ。
MAIN_KEYS = ["sardine", "urume", "buri", "sawara"]

# データソース（2026-07-04決定: 太平洋12県版を標準にする）
#   系群混在を避けるため、太平洋沿岸12県（岩手・宮城・福島・茨城・千葉・静岡・
#   愛知・三重・和歌山・徳島・高知・宮崎）合算を既定とする。
#   1956-2015 は表5（都道府県別長期累年）、2016-2023 は年次別2-2表（大海区
#   都道府県振興局別 魚種別漁獲量）を接続。全国行が表3と完全一致・2015境界も
#   連続することを検証済。2024は確報未公開（速報のみ）のため未収録。
#   全国版（1956-2024）も比較用に選択可。
_CSV_PACIFIC = "estat_海面漁業魚種別漁獲量_太平洋12県_1956-2023.csv"
_CSV_NATIONAL = "estat_海面漁業魚種別漁獲量_全国_1956-2024.csv"
_CSV_NAME = _CSV_PACIFIC  # 既定


def _find_data_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "..", "..", "data"),
        os.path.join(here, "..", "data"),
        os.path.join(here, "data"),
        os.path.join(os.getcwd(), "data"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return os.path.abspath(c)
    raise FileNotFoundError("data/ フォルダが見つかりません: " + repr(candidates))


def load_catch_table(csv_name=_CSV_NAME):
    """
    estat CSV を読み、(header, rows) を返す。
      header : 列名リスト（先頭は "year"）
      rows   : dict のリスト。year=int、魚種列=float（千トン, 欠損は None）
    元 CSV の単位トン → ÷1000 で千トンに換算。
    csv_name で全国版(_CSV_NATIONAL)にも切替可（既定は太平洋12県版）。
    """
    path = os.path.join(_find_data_dir(), csv_name)
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        header[0] = "year"
        rows = []
        for raw in reader:
            if not raw or raw[0].strip() == "":
                continue
            rec = {"year": int(float(raw[0]))}
            for col, val in zip(header[1:], raw[1:]):
                val = val.strip()
                rec[col] = (float(val) / 1000.0) if val not in ("", "-") else None
            rows.append(rec)
    return header, rows


def get_catch_series(key, csv_name=_CSV_NAME):
    """
    魚種 key の (years, catch) を返す。
      years : np.ndarray[int]   （欠損年を除いた昇順）
      catch : np.ndarray[float] （千トン）
    csv_name で全国版(_CSV_NATIONAL)にも切替可（既定は太平洋12県版）。
    """
    if key not in SPECIES:
        raise KeyError(f"未知の魚種 key: {key}（{list(SPECIES)}）")
    col = SPECIES[key]
    _, rows = load_catch_table(csv_name)
    years, catch = [], []
    for rec in rows:
        if rec.get(col) is not None:
            years.append(rec["year"])
            catch.append(rec[col])
    order = np.argsort(years)
    years = np.asarray(years, dtype=int)[order]
    catch = np.asarray(catch, dtype=float)[order]
    return years, catch


if __name__ == "__main__":
    _, rows = load_catch_table()
    yrs = [r["year"] for r in rows]
    print(f"期間: {min(yrs)}–{max(yrs)}  ({len(rows)} 年)")
    for k in MAIN_KEYS:
        y, c = get_catch_series(k)
        print(f"{SPECIES_LABELS[k]:8s}  n={len(c):3d}  "
              f"max={c.max():8.1f}  min={c.min():7.1f}  "
              f"直近={c[-1]:7.1f} 千トン (千トン単位)")
