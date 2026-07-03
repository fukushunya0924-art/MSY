"""
利用可能な魚種の全組み合わせをスキャンし、
「同一栄養段階グループ × LM/NLM期のODE適合しやすさ」で
最良の4種組み合わせをランキングする。

栄養段階グループ定義:
  低次 (TL ~2.8-3.2): マイワシ、カタクチイワシ  ← 小型浮魚、植食寄り
  中次 (TL ~3.5-3.7): マサバ、ヤリイカ、スルメイカ ← 中型捕食者
  高次 (TL ~4.1-4.5): ブリ、サワラ           ← 大型捕食者

制約: x(被食者)グループ2種・y(捕食者)グループ2種が
      それぞれ同じ栄養段階グループに収まること。
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from itertools import combinations

# =========================================================
# 1. 全魚種の読み込み定義
# =========================================================
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

def load(filename, rename_yr=None, bio_col=None, catch_col=None, scale=1.0, catch_scale=1.0):
    path = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(path)
    for c in df.columns:
        if c not in ('年', '漁期年', '黒潮大蛇行の有無'):
            df[c] = pd.to_numeric(df[c], errors='coerce')
    if rename_yr:
        df = df.rename(columns={rename_yr: '年'})
    df['年'] = pd.to_numeric(df['年'], errors='coerce')
    df = df.dropna(subset=['年']).copy()
    df['bio_kt'] = df[bio_col].values * scale       # 千トン換算
    df['catch_kt'] = df[catch_col].values * catch_scale
    df['f'] = np.clip(df['catch_kt'] / df['bio_kt'], 0.0, 0.95)
    return df[['年', 'bio_kt', 'catch_kt', 'f']].dropna()

SPECIES = {
    # name: (trophic_group, df)
    'マイワシ': (
        'LOW',
        load('マイワシ時系列データ_資源量・漁獲量・漁獲係数 - マイワシ時系列データ_資源量・漁獲量・漁獲係数.csv',
             bio_col='資源量（万トン）', catch_col='漁獲量（万トン）', scale=10.0, catch_scale=10.0)
    ),
    'カタクチイワシ': (
        'LOW',
        load('カタクチイワシ太平洋時系列データ - カタクチイワシ太平洋時系列データ.csv',
             bio_col='資源量（千トン）', catch_col='漁獲量（トン）', scale=1.0, catch_scale=1/1000)
    ),
    'マサバ': (
        'MID',
        load('マサバ時系列データ_資源量・漁獲量・漁獲係数 - マサバ時系列データ_資源量・漁獲量・漁獲係数.csv',
             rename_yr='漁期年',
             bio_col='資源量（万トン）', catch_col='漁獲量（万トン）', scale=10.0, catch_scale=10.0)
    ),
    'ヤリイカ': (
        'MID',
        load('ヤリイカ太平洋時系列データ - ヤリイカ太平洋時系列データ.csv',
             bio_col='資源量（トン）', catch_col='漁獲量（トン）', scale=1/1000, catch_scale=1/1000)
    ),
    'スルメイカ': (
        'MID',
        load('スルメイカ秋季 資源量・漁獲量時系列データ - スルメイカ秋季 資源量・漁獲量時系列データ.csv',
             rename_yr='漁期年',
             bio_col='資源量（千トン）', catch_col='漁獲量（千トン）', scale=1.0, catch_scale=1.0)
    ),
    'ブリ': (
        'HIGH',
        load('ブリ時系列データ_資源量・漁獲量・漁獲係数 - ブリ時系列データ_資源量・漁獲量・漁獲係数.csv',
             bio_col='資源量（トン）', catch_col='漁獲量（トン）', scale=1/1000, catch_scale=1/1000)
    ),
    'サワラ': (
        'HIGH',
        load('サワラ時系列データ_資源量・漁獲量・漁獲係数 - サワラ時系列データ_資源量・漁獲量・漁獲係数.csv',
             bio_col='資源量（トン）', catch_col='漁獲量（トン）', scale=1/1000, catch_scale=1/1000)
    ),
}

NLM_RANGE = (2006, 2016)
LM_RANGE  = (2017, 2024)
MIN_POINTS = 6  # 各レジームの最低データ点数


def regime_stats(df, yr_range):
    mask = (df['年'] >= yr_range[0]) & (df['年'] <= yr_range[1])
    sub = df[mask]
    if len(sub) < MIN_POINTS:
        return None
    b = sub['bio_kt'].values
    return {
        'n':    len(sub),
        'cv':   b.std() / b.mean() if b.mean() > 0 else 99,
        'mean': b.mean(),
    }


def score_combo(x1, x2, y1, y2):
    """4種の組み合わせのスコアを返す。低いほど良い。"""
    results = {}
    for regime_name, yr_range in [('NLM', NLM_RANGE), ('LM', LM_RANGE)]:
        stats = {}
        for label, name in [('x1',x1),('x2',x2),('y1',y1),('y2',y2)]:
            _, df = SPECIES[name]
            s = regime_stats(df, yr_range)
            if s is None:
                return None  # データ不足でこの組み合わせは無効
            stats[label] = s
        results[regime_name] = stats

    # スコア: 全種・全レジームのCV平均（低ほどODE適合しやすい）
    cvs = []
    for regime_stats_dict in results.values():
        for s in regime_stats_dict.values():
            cvs.append(s['cv'])
    return float(np.mean(cvs)), results


# =========================================================
# 2. 全有効組み合わせを列挙
# =========================================================
# 同一栄養段階グループで2種ずつ選ぶ
groups = {'LOW': [], 'MID': [], 'HIGH': []}
for name, (grp, _) in SPECIES.items():
    groups[grp].append(name)

print("栄養段階グループ:")
for g, members in groups.items():
    print(f"  {g}: {members}")
print()

# x グループ候補: LOW または MID から2種
# y グループ候補: MID または HIGH から2種
# ただし x と y のグループは異なること（y は x より栄養段階が高い）
TROPHIC_ORDER = {'LOW': 0, 'MID': 1, 'HIGH': 2}

x_candidates = []
for g in ['LOW', 'MID']:
    for pair in combinations(groups[g], 2):
        x_candidates.append((g, list(pair)))

y_candidates = []
for g in ['MID', 'HIGH']:
    for pair in combinations(groups[g], 2):
        y_candidates.append((g, list(pair)))

rows = []
for xg, (xa, xb) in x_candidates:
    for yg, (ya, yb) in y_candidates:
        if TROPHIC_ORDER[xg] >= TROPHIC_ORDER[yg]:
            continue  # y は x より高栄養段階
        if len({xa, xb, ya, yb}) < 4:
            continue  # 重複排除
        result = score_combo(xa, xb, ya, yb)
        if result is None:
            continue
        mean_cv, regime_data = result

        # 各種のCV詳細
        detail = {}
        for regime_name, stats in regime_data.items():
            for sp_label, s in stats.items():
                detail[f"{regime_name}_{sp_label}_cv"] = s['cv']
                detail[f"{regime_name}_{sp_label}_n"]  = s['n']

        rows.append({
            'x1': xa, 'x2': xb, 'y1': ya, 'y2': yb,
            'x_group': xg, 'y_group': yg,
            'mean_cv': mean_cv,
            **detail
        })

df_rank = pd.DataFrame(rows).sort_values('mean_cv').reset_index(drop=True)

# =========================================================
# 3. 結果表示
# =========================================================
print(f"有効な組み合わせ数: {len(df_rank)}")
print()
print("=== 上位10組み合わせ（平均CV昇順 = ODE適合しやすさ順） ===")
print(f"{'Rank':4}  {'x1':12} {'x2':12} {'y1':12} {'y2':12} "
      f"{'xGrp':5} {'yGrp':5} {'平均CV':7}  "
      f"NLM_x1 NLM_x2 NLM_y1 NLM_y2  LM_x1  LM_x2  LM_y1  LM_y2")
for i, row in df_rank.head(10).iterrows():
    cvs = [row.get(f'{r}_{k}_cv', 99) for r in ['NLM','LM'] for k in ['x1','x2','y1','y2']]
    cv_str = ' '.join(f'{v:.2f}' for v in cvs)
    print(f"{i+1:4}  {row['x1']:12} {row['x2']:12} {row['y1']:12} {row['y2']:12} "
          f"{row['x_group']:5} {row['y_group']:5} {row['mean_cv']:.3f}    {cv_str}")

print()
print("=== 現在の 6.8 モデル（参考） ===")
cur = score_combo('マイワシ', 'カタクチイワシ', 'ヤリイカ', 'スルメイカ')
if cur:
    mean_cv, _ = cur
    print(f"  x: マイワシ + カタクチイワシ  y: ヤリイカ + スルメイカ  平均CV={mean_cv:.3f}")
