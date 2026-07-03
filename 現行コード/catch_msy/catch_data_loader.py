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


# 4魚種（被食者2・捕食者2）＋教授候補（サバ類・スルメイカ）も読める
SPECIES = {
    "sardine": "マイワシ",
    "anchovy": "カタクチイワシ",
    "buri":    "ブリ",
    "sawara":  "サワラ",
    "mackerel": "サバ類",
    "squid":    "スルメイカ",
}

# 表示ラベル（図・表用）
SPECIES_LABELS = {
    "sardine": "マイワシ",
    "anchovy": "カタクチイワシ",
    "buri":    "ブリ",
    "sawara":  "サワラ",
    "mackerel": "サバ類",
    "squid":    "スルメイカ",
}

# 主対象（run_catch_msy.py が既定で回す4種）
MAIN_KEYS = ["sardine", "anchovy", "buri", "sawara"]

_CSV_NAME = "estat_海面漁業魚種別漁獲量_全国_1956-2024.csv"


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


def load_catch_table():
    """
    estat CSV を読み、(header, rows) を返す。
      header : 列名リスト（先頭は "year"）
      rows   : dict のリスト。year=int、魚種列=float（千トン, 欠損は None）
    元 CSV の単位トン → ÷1000 で千トンに換算。
    """
    path = os.path.join(_find_data_dir(), _CSV_NAME)
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


def get_catch_series(key):
    """
    魚種 key の (years, catch) を返す。
      years : np.ndarray[int]   （欠損年を除いた昇順）
      catch : np.ndarray[float] （千トン）
    """
    if key not in SPECIES:
        raise KeyError(f"未知の魚種 key: {key}（{list(SPECIES)}）")
    col = SPECIES[key]
    _, rows = load_catch_table()
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
