# 発表スライド（数理生物学会 2026, 口頭15分）

`数理生物学会2026_発表スライド.pptx` — 本編16枚＋予備8枚。構成の元は `../発表準備.md`。

## 作り直し方

```bash
cd 発表
python3 make_figs.py      # figs/*.png と figs/labels.json を作る
python3 build_deck.py     # pptx を作る（figs/ が要る）
python3 qa_deck.py 数理生物学会2026_発表スライド.pptx   # 余白・はみ出し・重なりの点検
python3 render_preview.py 数理生物学会2026_発表スライド.pptx   # preview/ に見た目を再現
```

`make_figs.py` は `現行コード/msy/estimates_capacity_ry.pkl`（12変数の自由推定）と
資源評価データを読む。pkl が無い場合は先に `cd 現行コード/msy && python3 run_msy.py`。

## ファイル

| ファイル | 中身 |
|---|---|
| `make_figs.py` | 図を作る。**画像に焼き込むのは数値だけ**で、日本語は位置だけ `figs/labels.json` に書き出す |
| `build_deck.py` | スライドを組む。labels.json を読んでテキストボックスとして図の上に置く |
| `omml.py` | PowerPoint の「挿入 → 数式」と同じ形式（OMML）を組み立てる |
| `mathtext.py` | 点検用に数式を平文へ直す |
| `qa_deck.py` | 座標ベースの点検（端の余白・文字のはみ出し・重なり） |
| `render_preview.py` | LibreOffice が無い環境向けに、座標から見た目を再現する |

`figs/` と `preview/` は作り直せるので git に入れていない。

## 決めごと

- 図の中の日本語は**すべてテキストボックス**。あとから文言だけ直せる。数値は画像に入れたまま。
- 数式は OMML（PowerPoint 上で数式として編集できる）。捕食圧の記号は `MSY O(1/T) 論法について.pdf`
  に合わせて λ を使う。
- 背景は白（表紙とまとめだけ紺＋白文字）。色は紺・橙・灰の3色。全ページに番号。
- 4枚目の食物網と5枚目の行列は画像ではなく図形で組んである。

## 検算した数値

- 上限感度（無制約, 千トン/年）: 大蛇行なし 206.7 → 380.6 → 607.2 → 801.4 → 1094.0 ／
  大蛇行 140.6 → 203.6 → 246.0 → 282.2 → 345.5。出典は
  `現行コード/msy/outputs/sustainability_sensitivity_{NLM,LM}.csv` の `upper_bound_unconstrained` 行の合計。
- 期間平均資源量（1994–2024, 千トン）: ブリ 257.4 ／ ウルメイワシ 196.6 ／ マアジ 84.9 ／ サワラ 4.35。
  → 桁が2つ小さいのは**サワラ**（`発表準備.md` の旧記述を訂正済み）。
- 構造の確認（`sustainability.build_A_rho` で再実行）: A の逆行列の対角＝0、
  とれる量の2階差分＝機械精度でゼロ、角16点の最大（2144／966）> 内側2万点の最大（1935／926）。
