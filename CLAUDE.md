# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 概要

黒潮大蛇行（LM/NLM）レジームが日本近海4魚種の資源量に与える影響を、捕食被食ODEモデルとパラメータ推定で定量化する研究コード。

- **被食者 (x)**: マイワシ (x1), カタクチイワシ (x2)
- **捕食者 (y)**: ブリ (y1), サワラ (y2)
- **レジーム分割**: NLM 2006–2016 / LM 2017–2024

## 実行コマンド

```bash
# 標準推定（現行コード/ から実行、capacity_ry 12変数）
cd 現行コード && python3 data_loader.py             # データ確認

# MSY 計算（現行コード/msy/ から実行）
cd 現行コード/msy && python3 run_msy.py             # capacity_ry（デフォルト）
cd 現行コード/msy && python3 diagnose_iwashi.py     # マイワシ終端挙動の診断
cd 現行コード/msy && python3 plot_fit_smooth.py     # 滑らかな推定フィット図

# Catch-MSY（現行コード/catch_msy/ から実行）
cd 現行コード/catch_msy && python3 run_catch_msy.py # 4種・既定レンジ(0.2,0.6)
cd 現行コード/catch_msy && python3 sensitivity.py   # 終端レンジ感度・箱ひげ
```

出力 PNG は各スクリプトディレクトリ配下の `outputs/`（種構成・実装・制約種別を明記した日本語ファイル名）に保存される。置換前種の参考資料は `catch_msy/outputs/legacy/` に隔離。

## アーキテクチャ

### ディレクトリ構成

```
data/          CSVデータ（魚種別 資源量・漁獲量時系列、e-stat漁獲量）
現行コード/
  data_loader.py   資源評価データ読み込み・前処理・スケール統一
  model.py         ODE定義・推定エンジン（estimate, capacity_ry 12変数）
  msy/             MSY計算（msy_core.py, run_msy.py, diagnose_iwashi.py, plot_fit_smooth.py）
  catch_msy/       連続時間Catch-MSY（catch_data_loader.py, catch_msy_core.py, run_catch_msy.py, sensitivity.py）
outputs/       推定結果の PNG
旧版/           旧バージョン・試行版
報告書_MSY計算と持続性制約.md   Phase 5 の成果報告書
```

### ODE推定の流れ（`現行コード/`）

1. `data_loader.load_clean_dataframe()` → 4種CSVをマージ、NaN除去
2. `data_loader.get_series()` → 全種を千トン統一、漁獲圧 `f = catch/biomass`（上限0.95）を計算
3. NLM/LMの年次マスクで時系列を分割
4. `model.estimate()` → 各レジームをODE推定
   - 正規化空間（各種を全期間平均で除した平均1.0の空間）でODEを解く
   - `solve_ivp(method="LSODA")` 積分 → `least_squares(method="trf")` で対数誤差最小化
   - マルチスタート（`n_starts`）で局所解回避、相互作用パラメータに L2 正則化（`reg_lambda`）
5. 推定後に `model._to_absolute()` で元スケールの物理パラメータへ換算

### モデル定義（`model.py`, capacity_ry 12変数）

正規化空間の ODE 右辺（`make_ode`）:
```
dx1 = (r_x1 − f_x1)·x1 − L11·x1·y1 − L12·x1·y2
dx2 = (r_x2 − f_x2)·x2 − L21·x2·y1 − L22·x2·y2
dy1 = (−r_y1 − f_y1)·y1 + C1·L11·x1·y1 + D1·L21·x2·y1
dy2 = (−r_y2 − f_y2)·y2 + C2·L12·x1·y2 + D2·L22·x2·y2
```
- `r_x1, r_x2`: 被食者の自然増殖率 / `r_y1, r_y2`: 捕食者の自然死亡率
- `L11..L22`: 捕食圧（相互作用係数） / `C1,D1,C2,D2`: 捕食→捕食者への変換効率
- **密度依存項（種内競争 α）は含まない**

> **モデル方針（2026-06-13決定）**: 環境収容力なし（種内競争項 α なし）モデルのみで進める。
> `capacity_ry`（12変数, r_y自由化）を主力、`capacity`（10変数, r_y固定）も使用可。
> `full`（16変数, 種内競争項あり）は LM 期8点に対し過剰パラメータで識別性崩壊・非物理的パラメータ（c1=41等）が生じるため**使用しない**。

### 適合度指標

- **RMSE** `sqrt(mean((obs−pred)²))` — 千トン、絶対誤差
- **NRMSE** `RMSE / mean(obs)` — 無次元、魚種横断で比較可能
- **R²** `1 − SS_res/SS_tot` — 平らなデータ（分散極小）で負に暴れるアーティファクトに注意

→ **評価はNRMSE主体、R²は補助**。

---

## データ方針（重要）

### 使用データ = e-stat 太平洋12県版（2026-07-04決定）

**今後の全魚種で、e-stat「海面漁業魚種別漁獲量累年統計（都道府県別, 表5）」の太平洋沿岸12県合算を標準データとする。**

- **12県**: 岩手・宮城・福島・茨城・千葉・静岡・愛知・三重・和歌山・徳島・高知・宮崎
- **保存**: `data/estat_海面漁業魚種別漁獲量_太平洋12県_1956-2023.csv`（整形済, 単位トン, 8魚種）
- **ローダ**: `catch_msy/catch_data_loader.py` が既定でこれを読む（`_CSV_NAME=_CSV_PACIFIC`）。全国版は `get_catch_series(key, csv_name=_CSV_NATIONAL)` で比較選択可。
- **理由**: 全国合算は系群混在（太平洋系群＋日本海側の別系群等）で変動が多峰化し、Catch-MSY の定常性前提をさらに崩す（下記 Phase 7c 参照）。現行ODE推定の資源評価データも「太平洋系群」限定なので、海域整合もとれる。
- **データ接続（2026-07-05実施）**: 1956-2015 は表5（都道府県別長期累年）、**2016-2023 は年次別「2-2 大海区都道府県振興局別統計 魚種別漁獲量」**（各年確報, xls/xlsx混在）を県ごとに抽出・合算して接続。**検証済**: 各年の全国行が表3（全国長期累年）と完全一致（不一致0）、2015境界も連続。
- **制約（未解決）**: **2024は確報未公開**（速報のみ・県別2-2表なし）で欠測。終端年は2023。確報公開後に1年追加する。
- **県選択の粗さ**: 12県は北海道太平洋側・青森・神奈川・鹿児島等を含まず、県境と系群境界も不一致。将来精緻化の余地あり。

参考: 全国版 `data/estat_海面漁業魚種別漁獲量_全国_1956-2024.csv`（表3, 1956-2024, 69年欠損なし）と原本xlsx も残置。

### 単位・スケール

- e-stat CSV は単位トン → ローダで ÷1000 して千トンで扱う。
- 資源評価CSV（ODE推定用）は魚種で単位が異なる（マイワシ万トン×10、カタクチ千トン等）ため `data_loader.get_series()` で千トンに統一。

---

## 現在の状態（2026-07-11時点）

**種構成確定**: 被食者=マイワシ(x1)＋ウルメイワシ(x2, カタクチイワシから置換) ／ 捕食者=ブリ(y1)＋サワラ(y2)。

**確定パラメータ**（太平洋12県, 1956-2023, Catch-MSYで推定）:

| パラメータ | 魚種 | 値(1/年) | 終端レンジ | 状態 |
|---|---|---|---|---|
| r_x1 | マイワシ | 0.90 [0.71,1.12] | [0.6,0.95]（標準ルール外の例外採用、要相談） | 暫定 |
| r_x2 | ウルメイワシ | 0.78 [0.66,0.86] | [0.01,0.4]（標準ルール通り） | ✅ 確定 |
| c1+d1 | ブリ | 0.40 [0.27,0.58] | [0.3,0.7]（標準ルール通り） | ✅ 確定 |
| c2+d2 | サワラ | 0.37 [0.26,0.50] | [0.01,0.4]（標準ルール通り） | ✅ 確定 |

**推定の安定化（完了）**: `model.py` の `estimate_robust()`（マルチシード×マルチスタート並列探索）で局所解問題を解消。`run_msy.py`・`plot_fit_smooth.py`・`diagnose_iwashi.py` は切替済み。到達フィット: NLM平均NRMSE=0.146 / LM平均NRMSE=0.079。

**次のステップ**:
1. 本番 `run_msy.py` フルランでPhase 5〜5bのMSY結果を再検証(局所解時代との差分確認)。→ 完了(2026-07-10)。NLM制約MSY=691.6千トン/年、LM制約MSY=9127.2千トン/年(詳細はdocs/research_log.md Phase 8以前)。
2. ウルメx2 NLMの当てはまりの悪さ(R²=−2.63)の原因切り分け(モデル vs 指標値ノイズ)。→ 未着手。
3. 確定パラメータ(r_x1, r_x2, c1+d1, c2+d2)を固定 → 残りパラメータ推定 → MSY導出のパイプライン実装。→ **試作したが保留中**(Phase 8, docs/research_log.md)。`現行コード/fixed_params.py`・`model_constrained.py`は実装・検証済みだが、固定すると当てはまりがNLM2.5倍/LM1.5倍悪化し、モデル構造(C×Lの非識別性)・教授の導出前提(飽和関数形)・Catch-MSYの定常性仮定(2006年以降で漁獲量が1.7〜2.4倍に加速)の3点で矛盾が見つかった。`estimate_cache.py`・`run_msy.py`への配線は未着手。
4. マイワシの r_x1=0.90(標準ルール外の高レンジ採用)の妥当性を教授と相談。→ 上記3.の矛盾とあわせて相談予定。

詳細な実験経緯・判断根拠(Phase 1〜7d)は `docs/research_log.md` を参照。
