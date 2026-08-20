"""
漁獲下平衡 B*(f) が全種正になる f の領域を、箱 [0,f_max]^4 全体で厳密に特定する。

sustainability.py（無改修・公開APIのみ使用）の docstring にある通り、平衡は
2つの独立な2x2線形系に分解できる:
    L @ y_eq = r_x - f_x        (y_eq は f_x=(fx1,fx2) だけに依存)
    (C∘L) @ x_eq = r_y + f_y    (x_eq は f_y=(fy1,fy2) だけに依存)

よって「全4種が正」という条件は、独立な2つの2次元問題
    R_x = {f_x in [0,f_max]^2 : y1(f_x)>0 and y2(f_x)>0}
    R_y = {f_y in [0,f_max]^2 : x1(f_y)>0 and x2(f_y)>0}
に分解でき、4次元の正領域は R = R_x × R_y という直積（Cartesian product）になる
（2026-08-20 Phase 16 で確認。従来は箱16頂点での離散チェックのみだった
 [発表準備.md §5.8.5, 2026-08-05] を、連続領域として厳密に確定させたもの）。

さらに平衡収量 Y_eq(f) = f・B*(f) は f_x と f_y の間で双線形（bilinear）
（x_eq は f_y の、y_eq は f_x のアフィン関数であるため）。双線形関数を
P×Q（P,Q は凸多角形）上で最大化すると最大は必ず頂点対 (P の頂点, Q の頂点) で
達成されるという双線形計画法の標準事実により、R 上の Y_eq の最大は
R_x の頂点と R_y の頂点の全組み合わせを評価するだけで厳密に求まる
（グリッド探索や非線形最適化は不要）。

実行:
    cd 現行コード/msy && python3 positivity_region.py

出力:
    コンソールへの詳細レポート（境界直線・頂点・面積比・R制約下のY_eq最大）
    outputs/positivity_region_summary.csv （レジーム別サマリ1行ずつ）

このスクリプトは sustainability.py を一切変更せず、公開関数
（build_A_rho, compute_equilibrium）のみを使う。多角形演算（半平面クリップ・
面積計算）はこのファイル内で完結させている。
"""
import csv as _csv
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_here)
if _here not in sys.path:
    sys.path.insert(0, _here)
if _parent not in sys.path:
    sys.path.append(_parent)

import numpy as np

import estimate_cache                          # noqa: E402
import sustainability as sus                    # noqa: E402

_out_dir = os.path.join(_here, "outputs")
os.makedirs(_out_dir, exist_ok=True)

F_MAX = 0.95
_EPS = 1e-9


# =============================================================================
# 多角形演算（Sutherland-Hodgman: 凸多角形を半平面 A*x+B*y<=rhs で切る）
# =============================================================================

def _clip_halfplane(poly, A, B, rhs, eps=_EPS):
    def inside(p):
        return A * p[0] + B * p[1] <= rhs + eps

    def intersect(p1, p2):
        d1 = A * p1[0] + B * p1[1] - rhs
        d2 = A * p2[0] + B * p2[1] - rhs
        t = d1 / (d1 - d2)
        return (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))

    out = []
    n = len(poly)
    for i in range(n):
        cur, prev = poly[i], poly[i - 1]
        cur_in, prev_in = inside(cur), inside(prev)
        if cur_in:
            if not prev_in:
                out.append(intersect(prev, cur))
            out.append(cur)
        elif prev_in:
            out.append(intersect(prev, cur))
    return out


def _polygon_area(poly):
    if len(poly) < 3:
        return 0.0
    s = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


# =============================================================================
# R_x, R_y の厳密な多角形頂点（境界直線を陽に導出してクリップ）
# =============================================================================

def region_polygons(params_norm, f_max=F_MAX):
    """params_norm（capacity_ry 12変数）から R_x, R_y の頂点リストを返す。

    導出（モジュールdocstring参照）:
      y1>0 <=> sign_y*((r_x1-fx1)*L22-(r_x2-fx2)*L12) > 0
      y2>0 <=> sign_y*((r_x2-fx2)*L11-(r_x1-fx1)*L21) > 0
      x1>0 <=> sign_x*((r_y1+fy1)*CL22-(r_y2+fy2)*CL12) > 0
      x2>0 <=> sign_x*((r_y2+fy2)*CL11-(r_y1+fy1)*CL21) > 0
    （sign_y=sign(det L), sign_x=sign(det C∘L)）を "A*f1+B*f2<=rhs" 形に変形し、
    箱 [0,f_max]^2 をこの4本の半平面で順にクリップする。
    """
    r_x1, r_x2, r_y1, r_y2, L11, L12, L21, L22, C1, D1, C2, D2 = params_norm
    CL11, CL12, CL21, CL22 = C1 * L11, D1 * L21, C2 * L12, D2 * L22
    Delta_L = L11 * L22 - L12 * L21
    Delta_CL = CL11 * CL22 - CL12 * CL21
    sign_y = 1.0 if Delta_L > 0 else -1.0
    sign_x = 1.0 if Delta_CL > 0 else -1.0

    A1, B1 = sign_y * L22, -sign_y * L12
    rhs1 = sign_y * (r_x1 * L22 - r_x2 * L12)
    A2, B2 = -sign_y * L21, sign_y * L11
    rhs2 = sign_y * (r_x2 * L11 - r_x1 * L21)
    poly_x = [(0.0, 0.0), (f_max, 0.0), (f_max, f_max), (0.0, f_max)]
    poly_x = _clip_halfplane(poly_x, A1, B1, rhs1)
    poly_x = _clip_halfplane(poly_x, A2, B2, rhs2)

    A3, B3 = -sign_x * CL22, sign_x * CL12
    rhs3 = sign_x * (r_y1 * CL22 - r_y2 * CL12)
    A4, B4 = sign_x * CL21, -sign_x * CL11
    rhs4 = sign_x * (r_y2 * CL11 - r_y1 * CL21)
    poly_y = [(0.0, 0.0), (f_max, 0.0), (f_max, f_max), (0.0, f_max)]
    poly_y = _clip_halfplane(poly_y, A3, B3, rhs3)
    poly_y = _clip_halfplane(poly_y, A4, B4, rhs4)

    return {
        "poly_fx": poly_x, "poly_fy": poly_y,
        "area_fx": _polygon_area(poly_x), "area_fy": _polygon_area(poly_y),
        "box_area": f_max ** 2,
    }


def r_constrained_yield_max(params_norm, means, poly_fx, poly_fy):
    """R=R_x×R_y 上の Y_eq(f)=f・B*(f) の最大を、頂点対の全評価で厳密に求める。

    Y_eq は f_x と f_y の間で双線形（x_eq は f_y だけの、y_eq は f_x だけの
    アフィン関数）なので、双線形計画法の標準事実により最大は必ず
    (R_x の頂点, R_y の頂点) の組で達成される（グリッド探索は不要）。
    多角形の頂点はしばしば境界線の交点（複数種が同時に0）にもなるため、
    compute_equilibrium の厳密な正値判定（eps=1e-10）を満たさない組は除外する。
    """
    cfg = sus.DEFAULT_SUSTAINABILITY
    best = None
    evaluated = []
    for fx in poly_fx:
        for fy in poly_fy:
            fvec = np.array([fx[0], fx[1], fy[0], fy[1]])
            res = sus.compute_equilibrium(params_norm, fvec, cfg, species_scope="all")
            if not res["positive"]:
                continue
            B_abs = res["B_eq_norm"] * means
            Y = float(np.dot(fvec, B_abs))
            evaluated.append((fvec.copy(), Y, B_abs.copy()))
            if best is None or Y > best[1]:
                best = (fvec.copy(), Y, B_abs.copy())
    return best, evaluated


# =============================================================================
# メイン
# =============================================================================

def _fmt_poly(poly):
    return "[" + ", ".join(f"({a:.4f},{b:.4f})" for a, b in poly) + "]"


def main():
    est = estimate_cache.load_estimates()
    if est is None:
        print("[ERROR] estimate_cache.load_estimates() が None。先に run_msy.py を実行してキャッシュを作成してください。")
        sys.exit(1)

    csv_rows = []
    print("=" * 78)
    print("正の共存平衡を持つ f の領域 R = R_x × R_y（Phase 16, 2026-08-20）")
    print("=" * 78)

    for rname in ["NLM", "LM"]:
        pn = est[rname]["params_norm"]
        mn = est[rname]["means"]
        reg = region_polygons(pn)
        frac_x = reg["area_fx"] / reg["box_area"]
        frac_y = reg["area_fy"] / reg["box_area"]
        vol_frac = frac_x * frac_y

        print(f"\n-- {rname} --")
        print(f"  R_x 頂点(f_x1,f_x2)（y1>0 & y2>0）: {_fmt_poly(reg['poly_fx'])}")
        print(f"  R_x 面積比 = {frac_x:.4f}")
        print(f"  R_y 頂点(f_y1,f_y2)（x1>0 & x2>0）: {_fmt_poly(reg['poly_fy'])}")
        print(f"  R_y 面積比 = {frac_y:.4f}")
        print(f"  R の4次元体積比 = R_x面積比 × R_y面積比 = {vol_frac:.4f}")

        best, evaluated = r_constrained_yield_max(pn, mn, reg["poly_fx"], reg["poly_fy"])
        print(f"  R上で厳密に正となる頂点対: {len(evaluated)}組")
        for fvec, Y, B_abs in evaluated:
            print(f"    f={np.round(fvec, 4)}  B*={np.round(B_abs, 2)}  Y_eq={Y:.2f}")
        if best is not None:
            f_opt, Y_opt, B_opt = best
            print(f"  ★ R制約下の Y_eq 最大 = {Y_opt:.2f} 千トン/年  "
                  f"f*={np.round(f_opt, 4)}  B*={np.round(B_opt, 2)}")
        else:
            f_opt, Y_opt, B_opt = np.full(4, np.nan), float("nan"), np.full(4, np.nan)
            print("  ★ R上で厳密に正となる頂点対が無い（数値的に要再確認）")

        csv_rows.append({
            "regime": rname,
            "Rx_area_fraction": f"{frac_x:.6f}",
            "Ry_area_fraction": f"{frac_y:.6f}",
            "R_volume_fraction_4d": f"{vol_frac:.6f}",
            "Rx_vertices": _fmt_poly(reg["poly_fx"]),
            "Ry_vertices": _fmt_poly(reg["poly_fy"]),
            "Yeq_max_on_R": f"{Y_opt:.4f}" if np.isfinite(Y_opt) else "",
            "f_opt_on_R": np.array2string(f_opt, precision=4) if np.all(np.isfinite(f_opt)) else "",
        })

    csv_path = os.path.join(_out_dir, "positivity_region_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = _csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nCSV出力: {csv_path}")


if __name__ == "__main__":
    main()
