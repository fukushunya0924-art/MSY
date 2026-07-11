"""
model_constrained.py の往復テスト（pytest不使用の自己検証スクリプト）。

pytest がこの環境に入っていないため、直接実行できる形にしてある:
    python3 現行コード/tests/test_model_constrained.py

検証内容:
  1. ランダムな q(bounds内) とランダムな正の means を多数生成し、
     reconstruct_params -> model._to_absolute を通したとき、
     固定した c1+d1==S1, c2+d2==S2, r_x1==fixed["r_x1"], r_x2==fixed["r_x2"]
     が成立すること（制約が本当に効いているかの確認）。
  2. 復元された12次元ベクトルの並びが
     model.MODELS["capacity_ry"]["names"] と対応していること
     （index 0 が r_x1, index 8 が C1）。
  3. theta1=0 / theta1=1 の極端値で c1/d1（c2/d2）が正しく片側に寄ること。

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


def test_round_trip_fixed_values_are_respected():
    """ランダムな q, means の組み合わせで c1+d1==S1, c2+d2==S2, r_x1/r_x2==fixed値 を確認する。"""
    rng = np.random.default_rng(12345)
    fixed = fixed_params.get_point()
    lower = np.array(mc.FREE_LOWER)
    upper = np.array(mc.FREE_UPPER)

    n_trials = 500
    for _ in range(n_trials):
        q = rng.uniform(lower, upper)
        means = rng.uniform(0.1, 100.0, size=4)  # ランダムな正の means (mx1, mx2, my1, my2)

        params_norm = mc.reconstruct_params(q, means, fixed)
        abs_p = model._to_absolute(params_norm, means)

        assert np.isclose(abs_p["c1"] + abs_p["d1"], fixed["S1"], rtol=1e-9), \
            f"c1+d1 != S1: {abs_p['c1']} + {abs_p['d1']} != {fixed['S1']}"
        assert np.isclose(abs_p["c2"] + abs_p["d2"], fixed["S2"], rtol=1e-9), \
            f"c2+d2 != S2: {abs_p['c2']} + {abs_p['d2']} != {fixed['S2']}"
        assert np.isclose(abs_p["r_x1"], fixed["r_x1"], rtol=1e-9), \
            f"r_x1 != fixed r_x1: {abs_p['r_x1']} != {fixed['r_x1']}"
        assert np.isclose(abs_p["r_x2"], fixed["r_x2"], rtol=1e-9), \
            f"r_x2 != fixed r_x2: {abs_p['r_x2']} != {fixed['r_x2']}"

    print(f"[OK] test_round_trip_fixed_values_are_respected ({n_trials} trials)")


def test_params_norm_ordering_matches_capacity_ry_names():
    """reconstruct_params が返す12次元ベクトルの並びが capacity_ry の names と一致すること。"""
    names = model.MODELS["capacity_ry"]["names"]
    assert names == ["r_x1", "r_x2", "r_y1", "r_y2", "L11", "L12", "L21", "L22",
                      "C1", "D1", "C2", "D2"], "capacity_ry の names 定義が想定と異なる"
    assert names[0] == "r_x1"
    assert names[8] == "C1"

    fixed = {"r_x1": 0.9, "r_x2": 0.78, "S1": 0.40, "S2": 0.37}
    means = np.array([2.0, 3.0, 5.0, 7.0])  # mx1, mx2, my1, my2
    q = np.array([0.31, 0.42, 0.11, 0.12, 0.13, 0.14, 0.6, 0.3])
    r_y1, r_y2, L11, L12, L21, L22, theta1, theta2 = q
    mx1, mx2, my1, my2 = means

    params_norm = mc.reconstruct_params(q, means, fixed)

    # index 0: r_x1（固定値そのまま）
    assert np.isclose(params_norm[0], fixed["r_x1"])
    # index 1-3: r_x2, r_y1, r_y2
    assert np.isclose(params_norm[1], fixed["r_x2"])
    assert np.isclose(params_norm[2], r_y1)
    assert np.isclose(params_norm[3], r_y2)
    # index 4-7: L11, L12, L21, L22（qそのまま、正規化空間なので換算不要）
    assert np.isclose(params_norm[4], L11)
    assert np.isclose(params_norm[5], L12)
    assert np.isclose(params_norm[6], L21)
    assert np.isclose(params_norm[7], L22)
    # index 8: C1 = c1 * mx1/my1 = (theta1*S1) * mx1/my1
    expected_C1 = (theta1 * fixed["S1"]) * mx1 / my1
    assert np.isclose(params_norm[8], expected_C1)
    # index 9: D1 = d1 * mx2/my1 = ((1-theta1)*S1) * mx2/my1
    expected_D1 = ((1 - theta1) * fixed["S1"]) * mx2 / my1
    assert np.isclose(params_norm[9], expected_D1)
    # index 10: C2 = c2 * mx1/my2 = (theta2*S2) * mx1/my2
    expected_C2 = (theta2 * fixed["S2"]) * mx1 / my2
    assert np.isclose(params_norm[10], expected_C2)
    # index 11: D2 = d2 * mx2/my2 = ((1-theta2)*S2) * mx2/my2
    expected_D2 = ((1 - theta2) * fixed["S2"]) * mx2 / my2
    assert np.isclose(params_norm[11], expected_D2)

    print("[OK] test_params_norm_ordering_matches_capacity_ry_names")


def test_theta_extremes_allocate_entirely_to_one_side():
    """theta1=0/1, theta2=0/1 の極端値で c1/d1, c2/d2 が正しく片側に寄ることを確認する。"""
    fixed = fixed_params.get_point()
    means = np.array([4.0, 6.0, 2.0, 3.0])  # mx1, mx2, my1, my2

    eps = 1e-6  # bounds は 1e-6 <= theta <= 1-1e-6 なので端点近傍を使う

    # theta1=eps -> c1 ≈ 0, d1 ≈ S1 / theta2 は中間値に固定
    q_lo = np.array([0.3, 0.4, 0.1, 0.1, 0.1, 0.1, eps, 0.5])
    params_norm_lo = mc.reconstruct_params(q_lo, means, fixed)
    abs_lo = model._to_absolute(params_norm_lo, means)
    assert abs_lo["c1"] < 1e-3, f"theta1=eps なのに c1 が0に寄っていない: {abs_lo['c1']}"
    assert np.isclose(abs_lo["d1"], fixed["S1"], atol=1e-3), \
        f"theta1=eps なのに d1 が S1 に寄っていない: {abs_lo['d1']} vs {fixed['S1']}"

    # theta1=1-eps -> c1 ≈ S1, d1 ≈ 0
    q_hi = np.array([0.3, 0.4, 0.1, 0.1, 0.1, 0.1, 1 - eps, 0.5])
    params_norm_hi = mc.reconstruct_params(q_hi, means, fixed)
    abs_hi = model._to_absolute(params_norm_hi, means)
    assert np.isclose(abs_hi["c1"], fixed["S1"], atol=1e-3), \
        f"theta1=1-eps なのに c1 が S1 に寄っていない: {abs_hi['c1']} vs {fixed['S1']}"
    assert abs_hi["d1"] < 1e-3, f"theta1=1-eps なのに d1 が0に寄っていない: {abs_hi['d1']}"

    # theta2=eps -> c2 ≈ 0, d2 ≈ S2
    q_lo2 = np.array([0.3, 0.4, 0.1, 0.1, 0.1, 0.1, 0.5, eps])
    params_norm_lo2 = mc.reconstruct_params(q_lo2, means, fixed)
    abs_lo2 = model._to_absolute(params_norm_lo2, means)
    assert abs_lo2["c2"] < 1e-3, f"theta2=eps なのに c2 が0に寄っていない: {abs_lo2['c2']}"
    assert np.isclose(abs_lo2["d2"], fixed["S2"], atol=1e-3), \
        f"theta2=eps なのに d2 が S2 に寄っていない: {abs_lo2['d2']} vs {fixed['S2']}"

    # theta2=1-eps -> c2 ≈ S2, d2 ≈ 0
    q_hi2 = np.array([0.3, 0.4, 0.1, 0.1, 0.1, 0.1, 0.5, 1 - eps])
    params_norm_hi2 = mc.reconstruct_params(q_hi2, means, fixed)
    abs_hi2 = model._to_absolute(params_norm_hi2, means)
    assert np.isclose(abs_hi2["c2"], fixed["S2"], atol=1e-3), \
        f"theta2=1-eps なのに c2 が S2 に寄っていない: {abs_hi2['c2']} vs {fixed['S2']}"
    assert abs_hi2["d2"] < 1e-3, f"theta2=1-eps なのに d2 が0に寄っていない: {abs_hi2['d2']}"

    print("[OK] test_theta_extremes_allocate_entirely_to_one_side")


def test_import_smoke():
    """model_constrained のトップレベル import と公開APIの存在確認。"""
    assert hasattr(mc, "reconstruct_params")
    assert hasattr(mc, "estimate_constrained")
    assert hasattr(mc, "estimate_constrained_robust")
    assert mc.FREE_NAMES == ["r_y1", "r_y2", "L11", "L12", "L21", "L22", "theta1", "theta2"]
    print("[OK] test_import_smoke")


def main():
    tests = [
        test_round_trip_fixed_values_are_respected,
        test_params_norm_ordering_matches_capacity_ry_names,
        test_theta_extremes_allocate_entirely_to_one_side,
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
