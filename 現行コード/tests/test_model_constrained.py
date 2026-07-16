"""
model_constrained.py の往復テスト（pytest不使用の自己検証スクリプト）。

pytest がこの環境に入っていないため、直接実行できる形にしてある:
    python3 現行コード/tests/test_model_constrained.py

設計（2026-07-14, 10自由変数版）:
  固定 : r_x1, r_x2 のみ（fixed_params から参照。S1,S2 は使わない）
  自由 : [r_y1, r_y2, L11, L12, L21, L22, C1, D1, C2, D2]（正規化空間）

検証内容:
  1. reconstruct_params が r_x1, r_x2 を固定値どおり先頭に差し込み、
     残り10要素（r_y1..D2）をそのまま12次元へ連結すること。
  2. 復元された12次元ベクトルの並びが
     model.MODELS["capacity_ry"]["names"] と対応していること
     （index 0 が r_x1, index 8 が C1）。
  3. C1,D1,C2,D2 が自由推定である（S1,S2 で束縛されない）こと。
     -> params_norm の C1..D2 が q の値そのままであることを確認。
  4. FREE_NAMES / bounds が model.py capacity_ry の names[2:12] と一致すること。

失敗時は AssertionError を送出し、非ゼロ終了コードでプロセスを終える。
"""
import os
import sys

import numpy as np

# 現行コード/ を sys.path に追加（model.py, model_constrained.py, fixed_params.py を import するため）
_here = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_here)
sys.path.insert(0, _parent)

import model                     # noqa: E402
import model_constrained as mc   # noqa: E402
import fixed_params              # noqa: E402


def test_round_trip_only_rx_are_fixed():
    """ランダムな q に対し、r_x1/r_x2 のみが固定値に一致し、他はそのまま通ることを確認する。"""
    rng = np.random.default_rng(12345)
    fixed = fixed_params.get_point()
    lower = np.array(mc.FREE_LOWER)
    upper = np.array(mc.FREE_UPPER)

    n_trials = 500
    for _ in range(n_trials):
        q = rng.uniform(lower, upper)
        params_norm = mc.reconstruct_params(q, fixed)

        # r_x1, r_x2 は固定値どおり
        assert np.isclose(params_norm[0], fixed["r_x1"], rtol=1e-12), \
            f"r_x1 != fixed r_x1: {params_norm[0]} != {fixed['r_x1']}"
        assert np.isclose(params_norm[1], fixed["r_x2"], rtol=1e-12), \
            f"r_x2 != fixed r_x2: {params_norm[1]} != {fixed['r_x2']}"
        # 残り10要素は q そのまま
        assert np.allclose(params_norm[2:], q, rtol=1e-12), \
            f"params_norm[2:] が q と一致しない: {params_norm[2:]} vs {q}"

    print(f"[OK] test_round_trip_only_rx_are_fixed ({n_trials} trials)")


def test_params_norm_ordering_matches_capacity_ry_names():
    """reconstruct_params が返す12次元ベクトルの並びが capacity_ry の names と一致すること。"""
    names = model.MODELS["capacity_ry"]["names"]
    assert names == ["r_x1", "r_x2", "r_y1", "r_y2", "L11", "L12", "L21", "L22",
                      "C1", "D1", "C2", "D2"], "capacity_ry の names 定義が想定と異なる"
    assert names[0] == "r_x1"
    assert names[8] == "C1"

    fixed = {"r_x1": 0.9, "r_x2": 0.78}
    q = np.array([0.31, 0.42, 0.11, 0.12, 0.13, 0.14, 0.21, 0.22, 0.23, 0.24])
    r_y1, r_y2, L11, L12, L21, L22, C1, D1, C2, D2 = q

    params_norm = mc.reconstruct_params(q, fixed)

    assert np.isclose(params_norm[0], fixed["r_x1"])   # r_x1
    assert np.isclose(params_norm[1], fixed["r_x2"])   # r_x2
    assert np.isclose(params_norm[2], r_y1)
    assert np.isclose(params_norm[3], r_y2)
    assert np.isclose(params_norm[4], L11)
    assert np.isclose(params_norm[5], L12)
    assert np.isclose(params_norm[6], L21)
    assert np.isclose(params_norm[7], L22)
    # C1,D1,C2,D2 は自由推定（正規化空間の値をそのまま通す。theta/S 換算なし）
    assert np.isclose(params_norm[8], C1)
    assert np.isclose(params_norm[9], D1)
    assert np.isclose(params_norm[10], C2)
    assert np.isclose(params_norm[11], D2)

    print("[OK] test_params_norm_ordering_matches_capacity_ry_names")


def test_cd_are_free_not_bound_by_S():
    """C1,D1,C2,D2 が固定 S に束縛されず自由に取れること（絶対 c/d 換算も破綻しない）。"""
    fixed = {"r_x1": 0.5, "r_x2": 0.6}
    means = np.array([4.0, 6.0, 2.0, 3.0])  # mx1, mx2, my1, my2

    # 2つの異なる C/D 設定で、絶対 c1+d1 が別々の値になる（S 固定でない）ことを示す
    q_a = np.array([0.3, 0.4, 0.1, 0.1, 0.1, 0.1, 0.10, 0.10, 0.10, 0.10])
    q_b = np.array([0.3, 0.4, 0.1, 0.1, 0.1, 0.1, 0.50, 0.30, 0.20, 0.40])

    abs_a = model._to_absolute(mc.reconstruct_params(q_a, fixed), means)
    abs_b = model._to_absolute(mc.reconstruct_params(q_b, fixed), means)

    s_a = abs_a["c1"] + abs_a["d1"]
    s_b = abs_b["c1"] + abs_b["d1"]
    assert not np.isclose(s_a, s_b), \
        f"c1+d1 が2設定で同値になっている（S 固定の名残の疑い）: {s_a} vs {s_b}"
    # r_x は両方とも固定値どおり
    assert np.isclose(abs_a["r_x1"], fixed["r_x1"]) and np.isclose(abs_b["r_x2"], fixed["r_x2"])

    print("[OK] test_cd_are_free_not_bound_by_S")


def test_import_smoke():
    """model_constrained のトップレベル import と公開APIの存在確認。"""
    assert hasattr(mc, "reconstruct_params")
    assert hasattr(mc, "estimate_constrained")
    assert hasattr(mc, "estimate_constrained_robust")
    assert mc.FREE_NAMES == ["r_y1", "r_y2", "L11", "L12", "L21", "L22",
                             "C1", "D1", "C2", "D2"]
    assert len(mc.FREE_GUESS) == 10
    assert len(mc.FREE_LOWER) == 10
    assert len(mc.FREE_UPPER) == 10
    # bounds が model.py capacity_ry の names[2:12] 部分と一致すること
    cfg = model.MODELS["capacity_ry"]
    assert list(mc.FREE_LOWER) == list(np.array(cfg["lower"])[2:]), "FREE_LOWER が capacity_ry と不一致"
    assert list(mc.FREE_UPPER) == list(np.array(cfg["upper"])[2:]), "FREE_UPPER が capacity_ry と不一致"
    # theta/S 関連の残骸が無いこと
    assert not any("theta" in n for n in mc.FREE_NAMES)
    print("[OK] test_import_smoke")


def main():
    tests = [
        test_round_trip_only_rx_are_fixed,
        test_params_norm_ordering_matches_capacity_ry_names,
        test_cd_are_free_not_bound_by_S,
        test_import_smoke,
    ]
    failed = []
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed.append((t.__name__, str(e)))
            print(f"[FAIL] {t.__name__}: {e}")
        except Exception as e:
            failed.append((t.__name__, repr(e)))
            print(f"[ERROR] {t.__name__}: {e!r}")

    if failed:
        print(f"\n{len(failed)}/{len(tests)} tests FAILED")
        sys.exit(1)
    else:
        print(f"\nAll {len(tests)} tests passed")
        sys.exit(0)


if __name__ == "__main__":
    main()
