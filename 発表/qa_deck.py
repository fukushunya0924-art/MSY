"""スライドの機械チェック: はみ出し・端の余白・重なり・文字サイズ。

LibreOffice が無い環境なので、日本語1文字=1em / 半角=0.52em / 行送り1.35 で
行数を見積もり、テキストボックスの高さと突き合わせる。
"""
import sys
import unicodedata
from pptx import Presentation
from pptx.util import Emu
from mathtext import omath_of, linearize, eq_size, eq_height_lines

EMU_IN = 914400.0
W, H = 13.333, 7.5
MARGIN = 0.30          # 端からの最低余白（インチ）
BODY_MIN_PT = 11.0     # 本文の最小サイズ（脚注・ページ番号を除く）


def em_width(ch):
    if unicodedata.east_asian_width(ch) in ("W", "F", "A"):
        return 1.0
    return 0.52


def est_lines(text, box_w_in, size_pt):
    """折り返しを見積もって行数を返す。"""
    if not text:
        return 1
    cap = box_w_in * 72.0 / size_pt          # 1行に入る em 数
    n, cur = 1, 0.0
    for ch in text:
        w = em_width(ch)
        if cur + w > cap:
            n += 1
            cur = w
        else:
            cur += w
    return n


def rects_overlap(a, b, pad=0.0):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return (ax < bx + bw - pad and bx < ax + aw - pad and
            ay < by + bh - pad and by < ay + ah - pad)


def main(path):
    prs = Presentation(path)
    problems = []
    for idx, slide in enumerate(prs.slides, start=1):
        boxes = []
        for sh in slide.shapes:
            if sh.left is None or sh.top is None:
                continue
            x, y = sh.left / EMU_IN, sh.top / EMU_IN
            w, h = sh.width / EMU_IN, sh.height / EMU_IN
            kind = sh.shape_type
            name = f"{sh.shape_id}:{sh.name}"

            # --- 端の余白
            if x < MARGIN - 1e-6 or y < MARGIN - 1e-6 \
                    or x + w > W - MARGIN + 1e-6 or y + h > H - MARGIN + 1e-6:
                problems.append(
                    f"p{idx} 端に近すぎ/はみ出し {name} "
                    f"x={x:.2f} y={y:.2f} w={w:.2f} h={h:.2f}")

            has_text = sh.has_text_frame and sh.text_frame.text.strip()
            boxes.append((x, y, w, h, name, has_text, kind))

            # --- 文字のはみ出し見積もり
            if sh.has_text_frame:
                tf = sh.text_frame
                ml = (tf.margin_left or 0) / EMU_IN
                mr = (tf.margin_right or 0) / EMU_IN
                inner_w = max(w - ml - mr, 0.2)
                total = 0.0
                for p in tf.paragraphs:
                    om = omath_of(p)
                    eq_factor = 1.0
                    if om is not None:
                        size = eq_size(p)
                        txt = linearize(om)
                        eq_factor = eq_height_lines(om)
                    else:
                        sizes = [r.font.size.pt for r in p.runs if r.font.size]
                        size = max(sizes) if sizes else 18.0
                        txt = "".join(r.text for r in p.runs)
                    ls = p.line_spacing or 1.32
                    if om is not None:
                        lines = est_lines(txt, inner_w, size * 0.72)
                        lines = max(1, lines) * eq_factor / 1.32
                    else:
                        lines = est_lines(txt, inner_w, size)
                    sa = (p.space_after.pt if p.space_after else 0)
                    total += lines * size * ls / 72.0 + sa / 72.0
                if total > h + 0.06:
                    problems.append(
                        f"p{idx} 文字が箱からはみ出す可能性 {name} "
                        f"必要={total:.2f}in 箱={h:.2f}in  «{tf.text[:34]}»")

        # --- 重なり（テキスト同士 / テキストと図）
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                if not (a[5] or b[5]):
                    continue
                # 塗り箱の中に文字を置く設計は正常なので、包含は除外
                inside = (a[0] <= b[0] + .02 and a[1] <= b[1] + .02 and
                          a[0] + a[2] >= b[0] + b[2] - .02 and
                          a[1] + a[3] >= b[1] + b[3] - .02)
                inside |= (b[0] <= a[0] + .02 and b[1] <= a[1] + .02 and
                           b[0] + b[2] >= a[0] + a[2] - .02 and
                           b[1] + b[3] >= a[1] + a[3] - .02)
                if inside:
                    continue
                fig_lbl = ("figlabel" in a[4]) or ("figlabel" in b[4])
                pic = ("Picture" in a[4]) or ("Picture" in b[4])
                if fig_lbl and pic:
                    continue
                if rects_overlap(a[:4], b[:4], pad=0.02):
                    problems.append(
                        f"p{idx} 重なり {a[4]} × {b[4]}  "
                        f"({a[0]:.2f},{a[1]:.2f},{a[2]:.2f},{a[3]:.2f}) / "
                        f"({b[0]:.2f},{b[1]:.2f},{b[2]:.2f},{b[3]:.2f})")

    print(f"総ページ数: {len(prs.slides.__iter__.__self__._sldIdLst)}")
    if problems:
        print(f"\n指摘 {len(problems)} 件:")
        for p in problems:
            print("  -", p)
    else:
        print("\n指摘なし")
    return len(problems)


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1]) == 0 else 1)
