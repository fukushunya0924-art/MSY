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

## 現在の状態（2026-07-14時点, Phase 12反映）

**種構成確定**: 被食者=マアジ(x1, マイワシから置換)＋ウルメイワシ(x2, カタクチイワシから置換) ／ 捕食者=ブリ(y1)＋サワラ(y2)。

**制約推定の設計変更（Phase 12, 2026-07-14）**: 制約 ODE で**固定するのは r_x1, r_x2 のみ**に変更した（旧: r_x1,r_x2,S1,S2 の4値を固定＋theta 配分の8自由変数 → 新: **r_x1,r_x2 のみ固定の10自由変数** [r_y1,r_y2,L11,L12,L21,L22,C1,D1,C2,D2]）。S1(=c1+d1), S2(=c2+d2) は Catch-MSY の生成物として下表に残るが、**制約 ODE 推定では固定に使わず C1,D1,C2,D2 を自由推定**する（S固定が適合度悪化の主因という仮説の検証結果、下記）。

**Catch-MSY 由来パラメータ**（下表。r_x1,r_x2 は制約 ODE で固定、S1,S2 は Phase 12 以降**制約 ODE では未使用**）:

| パラメータ | 魚種 | 値(1/年) | 終端レンジ | catch源 | 制約ODEでの扱い |
|---|---|---|---|---|---|
| r_x1 | マアジ | 0.228 [0.206,0.246] | [0.01,0.4]（標準ルール通り） | FRA資源評価 1982-2024 | ✅ 固定 |
| r_x2 | ウルメイワシ | 0.739 [0.642,0.824] | [0.01,0.4]（標準ルール通り） | e-stat 太平洋12県 | ✅ 固定 |
| c1+d1 | ブリ | 0.395 [0.268,0.569] | [0.3,0.7]（標準ルール通り） | FRA資源評価 1994-2024 | ⛔ 未使用（C1,D1を自由推定） |
| c2+d2 | サワラ | 0.260 [0.220,0.295] | [0.01,0.4]（標準ルール通り） | FRA資源評価 1987-2024 | ⛔ 未使用（C2,D2を自由推定） |

> **マイワシ→マアジ載せ替え（2026-07-14, Phase 11）**: マイワシはCatch-MSYが標準ルール
> [0.01,0.4]で解けず[0.6,0.95]の例外採用が必要だった問題を、被食者x1をマアジに差し替える
> ことで解消。マアジ太平洋系群のbiomass/catchはFRA「令和7年度マアジ太平洋系群の資源評価」
> （表3-1, 1982-2024）から取得し `data/マアジ時系列データ_資源量・漁獲量・漁獲係数_FRA資源評価2025.csv`
> に保存。ODE推定(`msy/data_loader.py`)とCatch-MSY(`catch_msy/catch_data_loader.py`)の両方で
> 同一のFRAデータを使うため、catch源の不整合問題も最初から生じない。r_x2/S1/S2はx1変更と
> 無関係のため据置。詳細は `docs/research_log.md` Phase 11。
> なお Catch-MSY の事前分布は原論文（Martell & Froese 2013）整合済み（Phase 9a, K上限100×max(catch)・B0/K uniform）。

**推定の安定化（完了）**: `model.py` の `estimate_robust()`（マルチシード×マルチスタート並列探索）で局所解問題を解消。`run_msy.py`・`plot_fit_smooth.py`・`diagnose_iwashi.py` は切替済み。到達フィット（マアジ版, 自由推定, 2026-07-14再計測）: NLM平均NRMSE=0.099（R²=+0.71） / LM平均NRMSE=0.065（R²=+0.34）。マイワシ版(0.146/0.079)より改善。

**fit図は `run_msy.py`（自由版・`--constrained`版とも）が Step4 で毎回自動保存する**（`plot_fit`関数, 2026-07-14追加）。手動で `plot_fit_smooth.py`/`_plot_fit_constrained.py` を個別実行しなくても最新のfit図が `msy/outputs/fit_..._capacity_ry.png` / `fit_制約_..._constrained.png` に生成される。

**制約推定の到達フィット（Phase 12, 10変数, r_x のみ固定）**: NLM平均NRMSE=**0.293** / LM平均NRMSE=**0.170**。旧S固定版(8変数, 0.452/0.338)から明確に改善（NLM −35%, LM −50%）＝**「S固定が適合度悪化の主因」仮説は支持**。ただし自由版(0.099/0.065)には未達。魚種別NRMSE（NLM/LM）: マアジ 0.485/0.452, ウルメ 0.246/0.089, ブリ 0.308/0.075, サワラ 0.131/0.065。NLMで C1=9.92(上限付近)・物理c1=44.7 と非物理的（C×L非識別性残存）。※サンドボックス制約で本番予算16×8を一括完走できず、4×4バッチをNLM24/LM16シード積み上げたbest-of（両レジーム収束済）。詳細 `docs/research_log.md` Phase 12。

**マアジ不適合の構造診断（Phase 12b）**: 4種の中で**マアジだけ「固定 r_x1(0.228) < 漁獲圧 f_x1(≈0.41)」**。f=漁獲量/資源量（data_loader.py:141、推定パラメータではなくデータ由来の既知強制項）。マアジは毎年資源量の約4割を漁獲される高漁獲圧種で、被食者式の実効内因成長 (r_x1−f_x1)=−0.18/年 が捕食項の前から負→モデルが構造的にマアジを暴落させ横ばい実データに合わない。対照的にウルメは f≈0.11 と低漁獲圧で r_x2=0.739>>f と健全（LM NRMSE 0.089）。捕食者ブリ・サワラは r_y(自然死亡率)を自由推定するため固定 vs f の衝突なし。**根本原因**: Catch-MSYのr（漁獲を暗黙に含む余剰生産の内因成長）を、f を明示減算する捕食被食ODEの純内因成長 r_x にそのまま代入した不整合（f>r となるマアジで破綻）。

**持続性制約・上限張り付き改修（Phase 13, 2026-07-16）**: ODE側MSYの持続性判定を設定切替式に再設計し、漁獲率上限への張り付きを自動診断できるようにした。新規 `msy/sustainability.py`（平衡点計算・4モード持続判定・境界診断・感度分析・95%安全側解）＋診断ドライバ `msy/run_sustainability_diagnostics.py`＋テスト `tests/test_sustainability.py`（11群, pytest不使用）。`msy_core.py` は `np.trapz`→`np.trapezoid` 互換シムのみ追加（**numpy 2.x で現行 `run_msy.py` が `average_yield` で落ちていた既存バグを解消**）、現行 `check_sustainability`（legacy）は無改変で保持。
> - **現行90%制約の正体**: `run_msy.py` の `SUSTAIN_CFG` は関数既定の `mode="path"` ではなく **`mode="endpoint"`** で上書きされており、判定は「全4種で **B_i(T) ≥ 0.9·B0_i**（終端値 vs 初期値）」。B0は観測初年資源量＝**位相依存**。
> - **4モード**: `legacy_path`（現行互換・警告付き）/ `equilibrium_lrp`（無漁獲平衡比 `B_eq_fished ≥ lrp·B_eq_unfished`, 主解析想定）/ `trajectory_floor`（長期軌道の資源下限）/ `time_average_lrp`。設定はPython定数（`DEFAULT_SUSTAINABILITY` 等, YAML不使用）。
> - **決定的知見（フィット4種で実走・独立検証済）**: モデルは一般化LV（自己制限項なし＝Aの対角0）で、**両レジームとも正の共存平衡が存在しない**（NLM: マアジx1=−50千トン, LM: ウルメx2=−130千トン）。→ `equilibrium_lrp` は全lrp比で n_feasible=0（適用不可）。**収量は漁獲率上限f=0.95に張り付く上限駆動**（無制約: サワラy2等が常時上限、収量は範囲内で単調増加）で**内部最大なし**。∴ 結果は **MSYではなく「資源下限制約下の最大収量（LRP-constrained maximum yield）」** と呼ぶべき。詳細 `docs/research_log.md` Phase 13。
> - 実行: `cd msy && python3 run_sustainability_diagnostics.py`（自由推定キャッシュを使用, 約25秒, CSVを `outputs/sustainability_sensitivity_{NLM,LM}.csv` に出力）。

**次のステップ**:
0. **上限駆動・平衡非存在への対応（Phase 13の含意）**: 現行の密度依存なしLVでは内部MSYが定義できない。持続的な内部最大が要るなら種内競争項（α, 環境収容力）の再導入か、漁獲率上限を生物学的根拠で設定する必要。教授相談事項に追加。
1. **マアジ r_x1 の固定を外す**（r_x2 のみ固定＝11自由変数）で再実行し、マアジのフィット回復と「マアジのCatch-MSY r が主因」を直接検証。あるいはマアジ Catch-MSY r のプライアレンジ再検討。
2. 戦略的MSY（持続性制約つき）は Phase 11 の8変数版で NLM 374.7 / LM 188.6千トン/年。10変数版でのMSY再計算は `run_msy.py --constrained`（本番予算）で通せば新キャッシュから算出可能。
3. ウルメx2 NLMの当てはまりの悪さの原因切り分け(モデル vs 指標値ノイズ)。→ 未着手。
4. 教授相談事項: **Catch-MSY r と ODE r_x の意味不整合（f>r で破綻）**を追加。既存のC×L非識別性・catch源整合の是非・サワラS2の弱点(相関0.44)も残存。

**一時ファイル（次クリーンアップ候補）**: `msy/_run_constrained_report.py`（バッチ実行ヘルパー）, `msy/_partial_NLM.pkl` / `_partial_LM.pkl`（best-of中間）。

詳細な実験経緯・判断根拠(Phase 1〜12)は `docs/research_log.md` を参照。
