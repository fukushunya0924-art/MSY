# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 概要

黒潮大蛇行（LM/NLM）レジームが日本近海4魚種の資源量に与える影響を、捕食被食ODEモデルとパラメータ推定で定量化する研究コード。

- **被食者 (x)**: マアジ (x1), ウルメイワシ (x2) ※旧構成はマイワシ・カタクチイワシ（Phase 7d/11 で置換）
- **捕食者 (y)**: ブリ (y1), サワラ (y2)
- **レジーム分割**: NLM 2006–2016 / LM 2017–2024

## 実行コマンド

```bash
# 標準推定（現行コード/ から実行、capacity_ry 12変数）
cd 現行コード && python3 data_loader.py             # データ確認

# MSY 計算（現行コード/msy/ から実行）
cd 現行コード/msy && python3 run_msy.py             # capacity_ry（デフォルト）
cd 現行コード/msy && python3 run_msy.py --constrained # 制約推定10変数版
cd 現行コード/msy && python3 diagnose_iwashi.py     # マイワシ終端挙動の診断
cd 現行コード/msy && python3 plot_fit_smooth.py     # 滑らかな推定フィット図

# 持続性診断（現行コード/msy/ から実行）※既定は n_grid=5 の粗い格子
cd 現行コード/msy && python3 run_sustainability_diagnostics.py
# MSY.md §6・§7 の数値は刻み0.05固定版。重いのでレジーム別に -u でリダイレクト
cd 現行コード/msy && python3 -u run_sustainability_diagnostics.py --regime NLM --f-step 0.05
cd 現行コード/msy && python3 -u run_sustainability_diagnostics_iwashi.py --regime NLM --f-step 0.05
cd 現行コード/msy && python3 positivity_region.py   # 漁獲下平衡が正になる f 領域（Phase 16）
cd 現行コード/msy && python3 -u _regen_grid_figs_step05.py  # グリッド系PNGを刻み0.05で再生成

# Catch-MSY（現行コード/catch_msy/ から実行）
cd 現行コード/catch_msy && python3 run_catch_msy.py # 4種・既定レンジ(0.2,0.6)
cd 現行コード/catch_msy && python3 sensitivity.py   # 終端レンジ感度・箱ひげ

# 学会発表スライド（発表/ から実行, 詳細は 発表/README.md）
cd 発表 && python3 make_figs.py                     # figs/*.png と figs/labels.json
cd 発表 && python3 build_deck.py                    # pptx を組む（figs/ が要る）
cd 発表 && python3 qa_deck.py 数理生物学会2026_発表スライド.pptx      # 余白・はみ出し点検
cd 発表 && python3 render_preview.py 数理生物学会2026_発表スライド.pptx # preview/ に見た目再現

# テスト（pytest不使用, 12群）
cd 現行コード && .venv/bin/python tests/test_sustainability.py
```

出力 PNG は各スクリプトディレクトリ配下の `outputs/`（種構成・実装・制約種別を明記した日本語ファイル名）に保存される。置換前種の参考資料は `catch_msy/outputs/legacy/` に隔離。

**出力ファイルの扱い（重要）**:
- `msy/outputs/` **直下**は `run_msy.py` が毎回上書きする作業用。**ドキュメントが参照する正本は
  `msy/outputs/{マアジ版,マイワシ版}/{自由推定,制約_*}/`** の整理済みアーカイブ。
- `sustainability_sensitivity_*.csv` は接尾辞なし（`n_grid=5` 等分割）と `_step0.05`（刻み固定）が
  **意図的に併存**している。**採用値は `_step0.05` 側**。接尾辞なしは粗い格子が生む人工物の対照用
  （`MSY.md` 付録）。
- `発表/figs/`・`発表/preview/` は `make_figs.py`／`render_preview.py` で再生成できるため git 管理外。

## アーキテクチャ

### ディレクトリ構成

```
data/          CSVデータ（魚種別 資源量・漁獲量時系列、e-stat漁獲量）。old/ は旧リビジョン
現行コード/
  data_loader.py       資源評価データ読み込み・前処理・スケール統一
  model.py             ODE定義・推定エンジン（estimate/estimate_robust, capacity_ry 12変数）
  model_constrained.py 制約推定（r_x のみ固定の10自由変数, Phase 12）
  fixed_params.py      Catch-MSY 由来の確定パラメータ
  msy/                 MSY計算・持続性診断
                       msy_core.py / run_msy.py / sustainability.py /
                       run_sustainability_diagnostics.py / positivity_region.py /
                       estimate_cache.py / diagnose_iwashi.py / plot_fit_smooth.py
                       ※ *_iwashi.py と `_` 始まりはマイワシ版・補助ドライバ
  catch_msy/           連続時間Catch-MSY（catch_data_loader.py, catch_msy_core.py, run_catch_msy.py, sensitivity.py）
  tests/               test_sustainability.py（12群）, test_model_constrained.py（pytest不使用）
docs/research_log.md   実験経緯・判断根拠の全記録（Phase 1〜17）
MSY.md                 持続性4モードの計算手法と診断結果（結果の正本）
行列式結果.md          無漁獲平衡の手計算検算（4パターン）
発表/                  学会発表スライド（build_deck.py で pptx を生成）＋ README.md
発表準備.md            スライド骨子・想定質疑・構造定理の完全な導出（§5）
旧版/                  旧バージョン・試行版
報告書_MSY計算と持続性制約.md   Phase 5 の成果報告書（旧種構成・内容は MSY.md に置換済み）
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

## 現在の状態（2026-08-20時点, Phase 17反映）

> **【2026-08-19 決定・未実施】発表のメイン種構成を マアジ → マイワシ に戻す。**
> 現在のコード・図・ドキュメントはすべてマアジがメインのままなので、次セッションで
> 入れ替えること（影響範囲と留意点は `docs/research_log.md`「種構成の方針変更」）。
> マイワシ版の数値は刻み0.05で算出済みなので**再計算は不要**。ただしマイワシ LM の
> 収量 10,986千トン/年は $r_{x1}=3.00$ という過剰適合パラメータの産物（`MSY.md` §7）で、
> メインに据えるなら本編で正面から扱う必要がある。**この決定の理由自体が未記録**なので、
> 次セッションで確認して research_log に補うこと。

**種構成（現状のコード）**: 被食者=マアジ(x1, マイワシから置換)＋ウルメイワシ(x2, カタクチイワシから置換) ／ 捕食者=ブリ(y1)＋サワラ(y2)。

### 推定の状態

- **自由推定（12変数, `estimates_capacity_ry.pkl`）**: NLM平均NRMSE=**0.099**（R²=+0.71） / LM平均NRMSE=**0.065**（R²=+0.34）。マイワシ版(0.146/0.079)より改善。
- **制約推定（10変数, r_x のみ固定）**: NLM平均NRMSE=**0.293** / LM平均NRMSE=**0.170**。旧S固定版(8変数, 0.452/0.338)から大幅改善＝**「S固定が適合度悪化の主因」仮説は支持**。ただし自由版には未達。NLMで C1=9.92(上限付近)・物理c1=44.7 と非物理的（**C×L非識別性は未解消**）。
- **局所解問題は解消済み**: `model.py` の `estimate_robust()`（マルチシード×マルチスタート並列）。`run_msy.py`・`plot_fit_smooth.py`・`diagnose_iwashi.py` は切替済み。
- **fit図は `run_msy.py`（自由版・`--constrained` 版とも）が Step4 で毎回自動保存する**。`plot_fit_smooth.py` / `_plot_fit_constrained.py` の個別実行は不要。

**制約推定の設計（Phase 12）**: 固定するのは **r_x1, r_x2 のみ**（10自由変数 [r_y1,r_y2,L11,L12,L21,L22,C1,D1,C2,D2]）。S1(=c1+d1), S2(=c2+d2) は Catch-MSY の生成物として下表に残るが**制約 ODE では未使用**（C1,D1,C2,D2 を自由推定）。

| パラメータ | 魚種 | 値(1/年) | 終端レンジ | catch源 | 制約ODEでの扱い |
|---|---|---|---|---|---|
| r_x1 | マアジ | 0.228 [0.206,0.246] | [0.01,0.4]（標準ルール通り） | FRA資源評価 1982-2024 | ✅ 固定 |
| r_x2 | ウルメイワシ | 0.739 [0.642,0.824] | [0.01,0.4]（標準ルール通り） | e-stat 太平洋12県 | ✅ 固定 |
| c1+d1 | ブリ | 0.395 [0.268,0.569] | [0.3,0.7]（標準ルール通り） | FRA資源評価 1994-2024 | ⛔ 未使用（C1,D1を自由推定） |
| c2+d2 | サワラ | 0.260 [0.220,0.295] | [0.01,0.4]（標準ルール通り） | FRA資源評価 1987-2024 | ⛔ 未使用（C2,D2を自由推定） |

> Catch-MSY の事前分布は原論文（Martell & Froese 2013）整合済み（Phase 9a, K上限100×max(catch)・B0/K uniform）。
> マアジのbiomass/catchはFRA「令和7年度マアジ太平洋系群の資源評価」（表3-1, 1982-2024）由来で、
> ODE推定と Catch-MSY が同一データを使うため catch源の不整合は生じない（Phase 11）。

**マアジ不適合の構造診断（Phase 12b）**: 4種の中で**マアジだけ「固定 r_x1(0.228) < 漁獲圧 f_x1(≈0.41)」**（f=漁獲量/資源量＝データ由来の既知強制項, `data_loader.py:141`）。実効内因成長 (r_x1−f_x1)=−0.18/年 が捕食項の前から負→モデルが構造的にマアジを暴落させ、横ばいの実データに合わない。ウルメは f≈0.11 と低漁獲圧で健全（LM NRMSE 0.089）。捕食者は r_y を自由推定するため衝突なし。**根本原因**: Catch-MSY の r（漁獲を暗黙に含む余剰生産の内因成長）を、f を明示減算する捕食被食ODEの純内因成長 r_x にそのまま代入した意味の不整合。

### MSY 側の到達点：内点MSYは構造的に存在しない

**結論**: 本モデル（密度依存なし一般化LV）が返す「MSY」は探索上限 $f_{max}$ の人工物であり、**「資源下限制約下の最大収量（LRP-constrained maximum yield）」と呼ぶべき**。経緯は Phase 13→13b→14→15〜17。

- **持続性判定は4モードの設定切替式**（`msy/sustainability.py` ＋ ドライバ `run_sustainability_diagnostics.py`）: `legacy_path`（現行互換）/ `equilibrium_lrp` / `trajectory_floor` / `time_average_lrp`。設定はPython定数（YAML不使用）。
  **現行90%制約の正体**は `mode="endpoint"`＝「全4種で $B_i(T) \geq 0.9 B_i(0)$」。B0は観測初年資源量＝**位相依存**。
- **構造定理（Phase 14, 条件を 07-30 に訂正）**: 効いているのは「$A$ の対角0」**ではなく** $A$ の**ブロック反対角構造**。必要なのは $\mathrm{diag}(A^{-1})=0$ で、これは $\mathrm{diag}(A)=0$ からは**一般には従わない**（反例: $A=[[0,1,1],[1,0,1],[1,1,0]]$ は $\mathrm{diag}(A^{-1})=-1/2$）。本モデルは $A=\begin{bmatrix}0&-L\\ C\circ L&0\end{bmatrix}$ ゆえ $A^{-1}$ の対角が消える。
  → 平衡は $L\,y_{eq}=r_x-f_x$ / $(C\circ L)\,x_{eq}=r_y+f_y$ に分解され、**$x_{eq}$ は $f_x$ に依存しない**。よって $Y_{eq}(f)=f^{\mathsf T}B_{eq}(0)+f^{\mathsf T}A^{-1}f$ から $f_i^2$ が消えて **f について多重線形**→ 箱 $[0,f_{max}]^4$ 上の最大は必ず頂点。**推定値に依存しない構造的結果**。
- **共通漁獲率MSYは検討済み・不採用（Phase 13b）**: NLMでは内部ピークが立つが（マアジ f=0.373）LMでは上限張り付きのままで、内部最大の有無が**レジームで反転**する。密度依存ではなく「全種同一f制約＋栄養段階カップリング」が生む見かけの最大で、種の重み付け（等重み＝恣意的）に依存。→ `MSY.md` §8 に補助解析として記載。
- **格子の刻みは0.05に固定（Phase 15）**: 旧 `n_grid=5` は上限を上げるほど刻みも粗くなり、「収量増は上限のせいか刻みのせいか」を分離できなかった。刻み固定でも単調増加＝上限駆動は不変だが、Mode 1 の値は**約2倍に改訂**（マアジ NLM 527.7 / LM 254.9千トン/年）。
- **「無漁獲平衡が負」の解釈は訂正済み（Phase 15b）**: Mode 3 の合否を説明するのは無漁獲平衡ではなく**漁獲下の平衡の正値性**。被食者自身を漁獲してもその平衡は動かず（構造定理）、**捕食者漁獲だけが被食者平衡を正に転じさせる**。
- **`_STATE_FLOOR` は撤去済み（Phase 15c）**: 「資源量が負」は実装上の副作用だった。撤去後も Mode 3 の合否（NLM=False / LM=True）は不変。
- **正領域は直積構造（Phase 16）**: $R=\{f: B^*(f)>0\}=R_x\times R_y$（独立な2つの2次元多角形）。4次元探索が2つの2次元線形不等式問題に厳密帰着。$Y_{eq}$ は $f_x,f_y$ について双線形なので $R$ 上の最大は頂点対の全評価だけで厳密に求まる（グリッド探索不要）。実装は `msy/positivity_region.py`（正式昇格済みの唯一の構造解析スクリプト）。
  > ⚠️ **Phase 14・17 の数値検証（∂x_eq/∂f_x=0, diag(A⁻¹)=0, 2階差分＝機械精度, $Y(f)$ の閉じた式の照合）は
  > いまも scratchpad のみで正式コードベースに昇格していない**。`positivity_region.py` は双線形性を
  > *根拠として使う*が、これらを*検証する*コードは含まない。→ `tests/` に diag(A⁻¹)=0 のテストを1本
  > 入れるのが残タスク（`発表準備.md` §4 タスク4, 質疑Q11の数値的裏付け）。
- **Schaefer との対応（Phase 17）**: 種ごとに Schaefer 式を立てると2次式ではなく**直線**になる（多重線形性の言い換え）。共通 f を課すと間接効果で2次式が復活し、NLM は凹（内点ピーク f*=0.115）／LM は凸。ただしピーク点でマアジが負のままの**集約artifact**。

> **未実施の最優先改善（`MSY.md` §8b）**: Mode 1 の探索に「漁獲下の平衡が全種正」という前段フィルタを足す。
> $2\times2$ 線形方程式を解くだけでODE積分が不要なので160,000点格子でも数秒。現行 Mode 1 は終端1点しか
> 見ておらず「10年はもつが長期に崩れる $f$」を拾ってしまう（NLM の 527.7 がまさにそれ）。

### 学会発表

数理生物学会 2026（**9/7–10, 口頭15分**）。要旨提出済み（「多種資源動態モデルを用いたレジーム別最大持続生産量(MSY)の推定」）。**主軸は上記の構造定理**で、Catch-MSY結合の障害（和 vs 積、r の意味不整合）はそこへ至る診断として前段に配置。

> **Volterra の原理は「主張」として出さない方針（2026-07-30決定）**: 「自種の漁獲が自種の平衡量を変えない」は古典的既知事実で、売りにすると「既知の再発見」に見える。**主張するのは「このモデルクラスでは平衡収量が f について多重線形 ⇒ 最適 f は常に探索箱の頂点 ⇒ 報告MSYは f_max の人工物」という計算枠組みへの帰結の方**。Volterra は脚注1行の出典表記に留め、質疑で問われたときのみ展開する。

スライド構成（本編16枚＋予備9枚＋マイワシ版5枚）・想定質疑・週次タスクは **`発表準備.md`** に集約（同 §5 に構造定理の完全な導出）。pptx のビルド手順と検算した数値は **`発表/README.md`**。

**次のステップ**:
1. **Mode 1 に「漁獲下平衡が全種正」フィルタを追加**（上記 `MSY.md` §8b）。最優先。
2. **発表のメイン種構成を マイワシ に戻す**（冒頭の未実施決定）。再計算は不要、`発表/` の差し替えが主。
3. **マアジ r_x1 の固定を外す**（r_x2 のみ固定＝11自由変数）で再実行し、「マアジのCatch-MSY r が主因」を直接検証。あるいは Catch-MSY r のプライアレンジ再検討。
4. ウルメx2 NLMの当てはまりの悪さの原因切り分け（モデル vs 指標値ノイズ）。→ 未着手。
5. 教授相談事項: **Catch-MSY r と ODE r_x の意味不整合（f>r で破綻）**。既存のC×L非識別性・catch源整合の是非・サワラS2の弱点(相関0.44)も残存。

詳細な実験経緯・判断根拠(Phase 1〜17)は `docs/research_log.md`、MSY計算の手法と結果は `MSY.md`、学会発表の設計は `発表準備.md` を参照。
