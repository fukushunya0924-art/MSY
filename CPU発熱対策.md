# CPU発熱対策.md

2026-07-10、`現行コード/msy/run_msy.py` 実行中のPC発熱への対処記録。

## 発生した問題

`run_msy.py` → `model.estimate_robust()` が n_seeds=12 のマルチシード×マルチスタート探索を
`multiprocessing.Pool` で並列実行する際、`n_workers = min(n_seeds, os.cpu_count())` により
このMac（8コア）では **8プロセスが同時に90〜96%でフル稼働** し、PCが高温になった。

## 試した対策と結果

### 1. cpulimit（プロセス毎にCPU%上限） → 効果なし

```bash
cpulimit -l 50 -p <PID>
```

cpulimitプロセス自体は起動・監視していたが、対象プロセスのCPU%は下がらなかった。

**原因**: cpulimitはSIGSTOP/SIGCONTを間欠的に送ってCPU%を制御する仕組みだが、
**Apple Silicon（M系チップ）でこの仕組みが正しく機能しないという既知の問題**がある
（Intel Mac向け実装で、ARMアーキテクチャに完全対応していない）。

### 2. 低電力モード（システム全体） → 効果あり

```bash
sudo pmset -a lowpowermode 1   # 有効化
sudo pmset -a lowpowermode 0   # 計算終了後に解除
```

CPUの最大クロックそのものを下げるため、個別プロセスの権限問題に関係なく確実に効く。

### 3. taskpolicy -b（Apple Silicon純正QoS機構） → 一番効果的

```bash
for pid in 4671 4672 4673 4674 4675 4676 4677 4678; do
  sudo taskpolicy -b -p $pid
done
```

対象プロセスをE（省電力）コア専用に固定。CPU%表示が **95%前後 → 34%前後** まで低下し、
体感の発熱も明確に改善した。macOSのスケジューラが直接扱う仕組みなのでApple Siliconでも確実に効く。

※対象PIDは実行のたびに変わるため、都度 `ps aux | grep python3` で確認が必要。

## 次回実行時の暫定運用（コード修正前）

1. `run_msy.py` 実行後、`ps aux | grep python3` でワーカーPIDを確認
2. `sudo pmset -a lowpowermode 1` を実行
3. 各PIDに `sudo taskpolicy -b -p <PID>` を実行
4. 計算終了後 `sudo pmset -a lowpowermode 0` で解除

## 今後の恒久対策（未実施）

`現行コード/model.py` の `estimate_robust()`（229〜236行目付近）で並列数の上限
（`max_workers`）を明示的に設けるコード修正を予定。n_seeds=12は変えず並列度だけ下げることで、
総計算量は変えずに同時発熱のピークを抑える狙い。修正用プロンプトは別途用意済み（下記参照）。

---

## 次回Claude Codeに投げるプロンプト（並列数の上限化）

```
model.py の estimate_robust()（229〜236行目付近）を修正してください。

現状:
    n_workers = max(1, min(n_seeds, os.cpu_count() or 1))
    ...
    with multiprocessing.Pool(processes=n_workers) as pool:

この n_workers が cpu_count（このMacでは8）まで使われるため、n_seeds=12 のフルラン時に
8プロセス同時フル稼働となり、発熱が大きい問題が起きています。

やってほしいこと:
1. estimate_robust() に max_workers 引数を追加（デフォルト値は4程度を推奨、要相談）。
   n_workers = max(1, min(n_seeds, os.cpu_count() or 1, max_workers)) に変更する。
2. 呼び出し元（現行コード/msy/estimate_cache.py の estimate_regime 関数、
   および必要なら run_msy.py）からも max_workers を指定できるようにする
   （デフォルトのままでも動くよう後方互換を保つこと）。
3. n_seeds=12 は変えず、並列度だけ下げる（総計算量は変わらず、ウォールクロック時間は
   伸びる代わりに同時発熱を抑える、という意図をコメントに残す）。
4. 変更後、小さいn_starts/n_seedsで簡単な動作確認をし、estimate_cache.py の
   キャッシュ整合性判定（署名関数）に影響がないか確認してください。

背景: cpulimitはApple Siliconで機能せず、低電力モード＋taskpolicy -bで応急対応済み。
恒久対策としてコード側の並列数を絞りたい、という経緯です。
```
