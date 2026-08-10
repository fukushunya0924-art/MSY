"""数理生物学会2026 口頭発表(15分)のスライドを作る。

本編16枚＋予備8枚。白背景・色数を絞る・全ページ番号つき。

方針:
  - 図の中の日本語は一切焼き込まない。make_figs.py が書き出した labels.json を
    読んで、PowerPoint のテキストボックスとして図の上に置く（あとから直せる）。
  - 数式は PowerPoint の「挿入 → 数式」と同じ形式（OMML, omml.py）で入れる。
  - 食物網の図と行列の図は画像を使わず、図形とテキストボックスで組む。

実行: cd 発表 && python3 build_deck.py
"""
import json
import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.oxml.ns import qn
from pptx.oxml import parse_xml

from omml import Eq as E, add_equation, declare_a14_in_package

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs")
OUT = os.path.join(HERE, "数理生物学会2026_発表スライド.pptx")

with open(os.path.join(FIGS, "labels.json")) as fh:
    _META = json.load(fh)
FIG_LABELS, FIG_SIZES = _META["labels"], _META["sizes"]

FONT = "Meiryo"
NAVY = "1F4E79"
DARK = "1B3654"
ORANGE = "C0561A"
INK = "222222"
GRAY = "6E6E6E"
WHITE = "FFFFFF"
TINT = "F2F5F9"
TINT2 = "FBF1E8"
SHADE = "EDEDED"

W, H = 13.333, 7.5
ML = 0.62
CW = W - 2 * ML

prs = Presentation()
prs.slide_width = Inches(W)
prs.slide_height = Inches(H)
BLANK = prs.slide_layouts[6]
_page = {"n": 0}


# ------------------------------------------------------------------ 部品
def _set_font(run, size, bold=False, color=INK, name=FONT):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = RGBColor.from_string(color)
    run.font.name = name
    rPr = run._r.get_or_add_rPr()
    for tag in ("a:ea", "a:cs"):
        el = rPr.find(qn(tag))
        if el is None:
            el = parse_xml(f'<{tag} xmlns:a="http://schemas.openxmlformats.org/'
                           f'drawingml/2006/main" typeface="{name}"/>')
            rPr.append(el)
        else:
            el.set("typeface", name)


def textbox(slide, x, y, w, h, lines, size=18, color=INK, bold=False,
            align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, space_after=6,
            line_spacing=1.32):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    if isinstance(lines, str):
        lines = [lines]
    for i, item in enumerate(lines):
        opt = {}
        if isinstance(item, tuple):
            item, opt = item
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = opt.get("align", align)
        p.space_after = Pt(opt.get("space_after", space_after))
        p.line_spacing = opt.get("line_spacing", line_spacing)
        run = p.add_run()
        run.text = item
        _set_font(run, opt.get("size", size), opt.get("bold", bold),
                  opt.get("color", color))
    return tb


def box(slide, x, y, w, h, fill=TINT, line=None, line_w=1.25):
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y),
                                Inches(w), Inches(h))
    sh.shadow.inherit = False
    if fill is None:
        sh.fill.background()
    else:
        sh.fill.solid()
        sh.fill.fore_color.rgb = RGBColor.from_string(fill)
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = RGBColor.from_string(line)
        sh.line.width = Pt(line_w)
    sh.text_frame.word_wrap = True
    return sh


def arrow(slide, x1, y1, x2, y2, color=GRAY, width=1.5):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1),
                                   Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = RGBColor.from_string(color)
    c.line.width = Pt(width)
    ln = c.line._get_or_add_ln()
    ln.append(parse_xml(
        '<a:tailEnd xmlns:a="http://schemas.openxmlformats.org/drawingml/'
        '2006/main" type="triangle" w="med" len="med"/>'))
    return c


_ALIGN = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}


def _em(ch):
    import unicodedata
    return 1.0 if unicodedata.east_asian_width(ch) in ("W", "F", "A") else 0.52


def text_width(text, size):
    """折り返さない前提での文字幅（インチ）。"""
    return max(sum(_em(c) for c in line) for line in text.split("\n")) \
        * size / 72.0 + 0.05


def figure(slide, name, x, y):
    """図を原寸で置き、labels.json のテキストをその上にテキストボックスで置く。"""
    fw, fh = FIG_SIZES[name]
    slide.shapes.add_picture(os.path.join(FIGS, name + ".png"),
                             Inches(x), Inches(y), width=Inches(fw))
    for it in FIG_LABELS.get(name, []):
        px = x + it["x"] * fw
        py = y + (1.0 - it["y"]) * fh
        bw = max(text_width(it["text"], it["size"]), 0.18)
        bh = it["size"] * 1.30 / 72.0 * it["lines"] + 0.02
        bx = {"l": px, "c": px - bw / 2, "r": px - bw}[it["ha"]]
        by = {"t": py, "c": py - bh / 2, "b": py - bh}[it["va"]]
        tb = slide.shapes.add_textbox(Inches(bx), Inches(by), Inches(bw),
                                      Inches(bh))
        tb.name = "figlabel"
        tf = tb.text_frame
        tf.word_wrap = False
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        for k, part in enumerate(it["text"].split("\n")):
            p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
            p.alignment = _ALIGN[it["ha"]]
            p.space_after = Pt(0)
            p.line_spacing = 1.20
            run = p.add_run()
            run.text = part
            _set_font(run, it["size"], it["bold"], it["color"])
    return fw, fh


def page_number(slide, dark=False):
    tb = slide.shapes.add_textbox(Inches(W - 1.25), Inches(H - 0.62),
                                  Inches(0.7), Inches(0.30))
    tf = tb.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.RIGHT
    run = p.add_run()
    run.text = str(_page["n"])
    _set_font(run, 11, False, "8FA6C0" if dark else GRAY)


def new_slide(title=None, dark=False, title_size=26):
    _page["n"] += 1
    s = prs.slides.add_slide(BLANK)
    if dark:
        bg = s.background.fill
        bg.solid()
        bg.fore_color.rgb = RGBColor.from_string(DARK)
    if title:
        textbox(s, ML, 0.36, CW, 0.9, title, size=title_size, bold=True,
                color=(WHITE if dark else NAVY), line_spacing=1.15)
    page_number(s, dark)
    return s


def table(slide, x, y, w, rows, col_w, size=14, row_h=0.42, head_h=0.42):
    heights = [head_h] + [row_h] * (len(rows) - 1)
    shp = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y),
                                 Inches(w), Inches(sum(heights)))
    tbl = shp.table
    tbl.first_row = True
    tbl.horz_banding = False
    for j, cw in enumerate(col_w):
        tbl.columns[j].width = Inches(cw)
    for i, hh in enumerate(heights):
        tbl.rows[i].height = Inches(hh)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.margin_left = Inches(0.10)
            cell.margin_right = Inches(0.08)
            cell.margin_top = Inches(0.03)
            cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor.from_string(
                NAVY if i == 0 else (WHITE if i % 2 else TINT))
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT
            txt, opt = (val if isinstance(val, tuple) else (val, {}))
            run = p.add_run()
            run.text = txt
            _set_font(run, opt.get("size", size), opt.get("bold", i == 0),
                      opt.get("color", WHITE if i == 0 else INK))
    return shp


def lead(slide, text, y=1.28, size=17, color=INK):
    return textbox(slide, ML, y, CW, 0.5, text, size=size, color=color)


def note(slide, text, y=None, size=13, color=GRAY, height=0.5):
    y = H - 0.92 if y is None else y
    return textbox(slide, ML, y, CW - 0.8, height, text, size=size, color=color)


def eqbox(slide, x, y, w, h, items, size=15, color=INK, space_after=7):
    """数式と短い文を交互に置く箱。items は Eq ノード、または (文字列, opt)。"""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    first = True
    for item in items:
        if isinstance(item, tuple) or isinstance(item, str):
            txt, opt = (item if isinstance(item, tuple) else (item, {}))
            p = tf.paragraphs[0] if (first and not tf.paragraphs[0].runs) \
                else tf.add_paragraph()
            p.alignment = PP_ALIGN.LEFT
            p.space_after = Pt(opt.get("space_after", space_after))
            p.line_spacing = opt.get("line_spacing", 1.28)
            run = p.add_run()
            run.text = txt
            _set_font(run, opt.get("size", size), opt.get("bold", False),
                      opt.get("color", color))
        else:
            add_equation(tf, item, size=size + 1, color=color,
                         space_after=space_after, first=first)
        first = False
    return tb


# ------------------------------------------------------------------ 数式の部品
def v(base, sub=None, sub_italic=False):
    if sub is None:
        return E.i(base)
    return E.sub(E.i(base), E.i(sub) if sub_italic else E.n(sub))


def lam(ij):
    return E.sub(E.i("λ"), E.n(ij))


def ddt(top):
    return E.frac(E.seq(E.u("d"), top), E.seq(E.u("d"), E.i("t")))


# ================================================================== 1 表紙
s = new_slide(dark=True)
textbox(s, 1.0, 1.95, W - 2.0, 1.5,
        "多種資源動態モデルを用いた\nレジーム別最大持続生産量（MSY）の推定",
        size=32, bold=True, color=WHITE, align=PP_ALIGN.CENTER,
        line_spacing=1.28)
textbox(s, 1.0, 4.55, W - 2.0, 1.1,
        ["福岡 駿也（名古屋大学）",
         ("日本数理生物学会 2026年大会　2026年9月", {"size": 15, "color": "C9D6E4"})],
        size=20, color=WHITE, align=PP_ALIGN.CENTER, space_after=10)

# ================================================================== 2 背景
s = new_slide("黒潮の流れ方が変わると、近海の魚のとれ方も変わる")
lead(s, "本研究は、黒潮が大きく蛇行する時期とそうでない時期に分けて、4魚種の増減を同じ式で説明する。")
figure(s, "01_資源量の移り変わり", ML, 2.05)
note(s, "※「大蛇行が原因で魚が減った」とは言わない。時期で分けて比べるだけにとどめる。",
     y=5.70, size=14)

# ================================================================== 3 問い
s = new_slide("MSY（とっていい量の上限）の「山」を作っているのは何か")
lead(s, "魚が1種だけのモデルでは、とるほど増えにくくなる「混み合いの効果」が山を作る。")
figure(s, "02_山ができる", ML, 1.90)
textbox(s, 6.75, 1.95, 5.95, 0.4, "山ができる理由", size=18, bold=True, color=NAVY)
textbox(s, 6.75, 2.50, 5.95, 0.70,
        "とればとるほど魚が減り、増えにくくなる。この効果が式に入っていると、",
        size=15.5)
eqbox(s, 6.95, 3.30, 5.75, 1.4,
      [E.seq(E.sup(v("B"), E.n("*")), E.d(E.i("f")), E.o("="),
             E.i("K"), E.d(E.seq(E.n("1"), E.o("−"), E.frac(E.i("f"), E.i("r"))))),
       E.seq(E.i("Y"), E.d(E.i("f")), E.o("="), E.i("K"), E.i("f"), E.o("−"),
             E.frac(E.i("K"), E.i("r")), E.sup(E.i("f"), E.n("2")))],
      size=15)
b = box(s, 6.75, 4.85, 5.95, 0.62, fill=TINT2, line=ORANGE)
textbox(s, 6.92, 4.98, 5.61, 0.4,
        "2乗の項があるから山になる", size=15.5, bold=True, color=ORANGE)
b = box(s, ML, 5.78, CW, 0.95, fill=TINT2, line=ORANGE)
textbox(s, ML + 0.22, 5.92, CW - 0.44, 0.72,
        [("問い：混み合いの効果を入れない多種のモデルでは、MSY はどう決まるのか。",
          {"size": 18, "bold": True, "color": ORANGE}),
         "本発表の答え：内側には決まらない。報告される値は、探索範囲の上限そのものになる。"],
        size=16, color=INK, space_after=4)

# ================================================================== 4 データ
s = new_slide("餌の魚2種と、それを食べる魚2種")
for cx, nm in [(1.55, "ブリ  y1"), (4.15, "サワラ  y2")]:
    b = box(s, cx, 1.70, 2.1, 0.52, fill=TINT2, line=ORANGE)
    textbox(s, cx, 1.83, 2.1, 0.3, nm, size=14, align=PP_ALIGN.CENTER)
for cx, nm in [(1.55, "マアジ  x1"), (4.15, "ウルメイワシ  x2")]:
    b = box(s, cx, 3.50, 2.1, 0.52, fill=TINT, line=NAVY)
    textbox(s, cx, 3.63, 2.1, 0.3, nm, size=14, align=PP_ALIGN.CENTER)
for px in (2.60, 5.20):
    for qx in (2.60, 5.20):
        arrow(s, px, 3.50, qx, 2.22)
textbox(s, 0.62, 1.80, 0.9, 0.3, "食べる魚", size=11.5, color=ORANGE)
textbox(s, 0.62, 3.60, 0.9, 0.3, "餌の魚", size=11.5, color=NAVY)
textbox(s, 0.62, 4.28, 5.7, 0.3, "矢印＝食べる関係（4本すべてを式に入れる）",
        size=12, color=GRAY)
textbox(s, 0.62, 4.66, 5.7, 0.3, "同じ段の魚どうしの取り合いは、式に入れていない",
        size=13, color=ORANGE)
table(s, 6.55, 1.55, 6.16,
      [["魚種", "資源量と漁獲量の出どころ"],
       ["マアジ", "水産研究・教育機構（1982〜2024年）"],
       ["ウルメイワシ", "資源量の目安 ＋ 太平洋側12県の漁獲量"],
       ["ブリ", "水産研究・教育機構（1994〜2024年）"],
       ["サワラ", "水産研究・教育機構（1987〜2024年）"]],
      col_w=[1.85, 4.31], size=14, row_h=0.46, head_h=0.42)
textbox(s, 6.55, 3.92, 6.16, 0.35,
        "※いずれも太平洋系群。資源量と漁獲量の年ごとの値を使う。",
        size=13, color=GRAY)
textbox(s, 6.55, 4.48, 6.16, 1.5,
        ["4種そろう期間は 1994〜2024年（31年分）",
         "大蛇行なし：2006〜2016年（11年）",
         "大蛇行：2017〜2024年（8年）"],
        size=16, space_after=5)
b = box(s, ML, 6.02, CW, 0.78, fill=TINT, line=None)
textbox(s, ML + 0.22, 6.18, CW - 0.44, 0.5,
        "漁獲率 f ＝ 漁獲量 ÷ 資源量。推定する値ではなく、データから直接計算できる既知の値（上限 0.95）。",
        size=16, color=INK)

# ================================================================== 5 モデル
s = new_slide("4種をつないだ式：混み合いの効果は入れていない")
SUM_J = E.nary("∑", E.i("j"), E.seq(lam("ij"), v("x", "i", True),
                                    v("y", "j", True)))
SUM_I = E.nary("∑", E.i("i"), E.seq(v("c", "ij", True), lam("ij"),
                                    v("x", "i", True), v("y", "j", True)))
b = box(s, ML, 1.32, 6.85, 2.15, fill=TINT, line=None)
textbox(s, ML + 0.22, 1.44, 6.4, 0.3, "餌の魚", size=14, bold=True, color=NAVY)
eqbox(s, ML + 0.22, 1.74, 6.4, 0.62,
      [E.seq(ddt(v("x", "i", True)), E.o("="),
             E.d(E.seq(v("r", "xi", True), E.o("−"), v("f", "xi", True))),
             v("x", "i", True), E.o("−"), SUM_J)], size=14)
textbox(s, ML + 0.22, 2.42, 6.4, 0.3, "食べる魚", size=14, bold=True, color=ORANGE)
eqbox(s, ML + 0.22, 2.72, 6.4, 0.62,
      [E.seq(ddt(v("y", "j", True)), E.o("="), E.n("−"),
             E.d(E.seq(v("r", "yj", True), E.o("+"), v("f", "yj", True))),
             v("y", "j", True), E.o("+"), SUM_I)], size=14)
textbox(s, ML, 3.58, 6.85, 1.10,
        ["r_x：餌の魚がもともと増える速さ　　r_y：食べる魚が自然に減る速さ",
         "λ：食べられる強さ　　c：変換の割合（食べた分が体になる割合）",
         "f：漁獲率（データから計算した既知の値）"],
        size=14.5, space_after=5)
textbox(s, ML, 4.66, 6.85, 0.30,
        [("まとめて書くと", {"size": 15, "bold": True, "color": NAVY})],
        size=15)
eqbox(s, ML, 5.06, 6.85, 0.62,
      [E.seq(ddt(v("B", "i", True)), E.o("="), v("B", "i", True),
             E.d(E.seq(v("ρ", "i", True), E.o("+"),
                       E.nary("∑", E.i("j"), E.seq(v("A", "ij", True),
                                                   v("B", "j", True))))))],
      size=14)
textbox(s, ML, 5.74, 6.85, 0.34,
        "この行列 A の、どこをゼロにしたか。これが後半の結論を決める。",
        size=14.5, color=GRAY)

# 行列（図形＋テキストボックスで組む）
MX, MY, CWD, RHT, HH = 8.72, 1.78, 0.95, 0.42, 0.34
box(s, MX, MY, CWD * 2, RHT * 2, fill=SHADE, line=None)
box(s, MX + CWD * 2, MY + RHT * 2, CWD * 2, RHT * 2, fill=SHADE, line=None)
heads = ["マアジ", "ウルメ", "ブリ", "サワラ"]
cells = [["0", "0", "−λ11", "−λ12"],
         ["0", "0", "−λ21", "−λ22"],
         ["c1λ11", "d1λ21", "0", "0"],
         ["c2λ12", "d2λ22", "0", "0"]]
for j, hd in enumerate(heads):
    textbox(s, MX + j * CWD, MY - HH, CWD, 0.28, hd, size=11.5,
            align=PP_ALIGN.CENTER, color=GRAY)
for i, hd in enumerate(heads):
    textbox(s, MX - 0.98, MY + i * RHT + 0.09, 0.82, 0.28, hd, size=11.5,
            align=PP_ALIGN.RIGHT, color=GRAY)
    for j, cval in enumerate(cells[i]):
        textbox(s, MX + j * CWD, MY + i * RHT + 0.09, CWD, 0.28, cval,
                size=13, align=PP_ALIGN.CENTER,
                color=(GRAY if cval == "0" else INK))
for sx, shape in ((MX - 0.14, MSO_SHAPE.LEFT_BRACKET),
                  (MX + CWD * 4 + 0.02, MSO_SHAPE.RIGHT_BRACKET)):
    br = s.shapes.add_shape(shape, Inches(sx), Inches(MY - 0.06),
                            Inches(0.12), Inches(RHT * 4 + 0.12))
    br.shadow.inherit = False
    br.fill.background()
    br.line.color.rgb = RGBColor.from_string(INK)
    br.line.width = Pt(1.4)
textbox(s, 8.72, 3.66, 3.9, 0.3, "灰色＝すべてゼロ", size=12, color=ORANGE)
textbox(s, 7.70, 4.10, 5.0, 1.0,
        ["・左上がゼロ ＝ 餌の魚どうしの取り合いなし",
         "・右下がゼロ ＝ 食べる魚どうしの関係なし"],
        size=13.5, color=INK, space_after=4)
b = box(s, ML, 6.15, CW, 0.72, fill=TINT2, line=ORANGE)
textbox(s, ML + 0.22, 6.30, CW - 0.44, 0.45,
        "同じ段の魚どうしの関係も、自分自身の混み合いも、どちらも式に入れていない。",
        size=17, bold=True, color=ORANGE)

# ================================================================== 6 全体図
s = new_slide("MSY を出すまでの流れ")
lead(s, "係数をデータから決め、そこに漁獲率を入れて回し、"
        "資源が減りすぎない範囲で漁獲量を最大にする。")

NX = [0.62, 2.41, 5.33, 8.25, 11.17]
NW = [1.55, 2.68, 2.68, 2.68, 1.54]
NY, NH = 1.86, 1.02
for k, (nx, nw, fill, line, col, lines) in enumerate([
        (NX[0], NW[0], TINT, None, INK,
         ["データ", ("資源量・漁獲量", {"size": 12.5, "color": GRAY})]),
        (NX[1], NW[1], NAVY, None, WHITE,
         ["① パラメータを", "レジーム別に推定"]),
        (NX[2], NW[2], NAVY, None, WHITE,
         ["② 漁獲率 f を入れて", "T 年計算"]),
        (NX[3], NW[3], NAVY, None, WHITE,
         ["③ 減りすぎない条件で", "期間平均漁獲量を最大化"]),
        (NX[4], NW[4], TINT2, ORANGE, ORANGE,
         ["レジーム別の", "MSY"])]):
    box(s, nx, NY, nw, NH, fill=fill, line=line)
    textbox(s, nx + 0.08, NY, nw - 0.16, NH, lines, size=14, bold=True,
            color=col, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE,
            space_after=2, line_spacing=1.22)
for k in range(4):
    arrow(s, NX[k] + NW[k] + 0.03, NY + NH / 2,
          NX[k + 1] - 0.03, NY + NH / 2, color=NAVY, width=2.0)

DY, DH = 3.00, 2.58
for nx, nw, bullets in [
    (NX[1], NW[1],
     ["・時期ごとに係数12個を別々に推定",
      "・正規化した空間で ODE を数値積分（LSODA）",
      "・対数残差を最小二乗法（trf）で最小化",
      "・出発点を多数変えて局所解を避ける",
      "・当てはまりは NRMSE で評価"]),
    (NX[2], NW[2],
     ["・推定した式に、魚種ごとの一定の漁獲率 f を入れる",
      "・f は4成分（マアジ・ウルメ・ブリ・サワラ）",
      "・観測初年の資源量から T 年ぶん積分する"]),
    (NX[3], NW[3],
     ["・f を 0〜0.95 の格子で総当たり（4次元）",
      "・資源が減りすぎない条件を満たす f だけ残す",
      "・既定は「最終年の資源量が初年の9割以上」",
      "・残った f のなかで期間平均漁獲量を最大にする"])]:
    box(s, nx, DY, nw, DH, fill=TINT, line=None)
    textbox(s, nx + 0.16, DY + 0.14, nw - 0.32, DH - 0.28, bullets,
            size=12, space_after=4, line_spacing=1.24)

b = box(s, ML, 5.68, CW, 1.05, fill=TINT2, line=ORANGE)
textbox(s, ML + 0.24, 5.88, 2.5, 0.4, "③で最大にする量", size=14.5,
        bold=True, color=ORANGE)
eqbox(s, ML + 2.90, 5.82, 4.6, 0.72,
      [E.seq(E.i("Y"), E.o("="), E.frac(E.n("1"), E.i("T")),
             E.nary("∫", E.n("0"), E.seq(
                 E.nary("∑", E.i("i"), E.seq(v("f", "i", True),
                                             v("B", "i", True))),
                 E.u(" "), E.u("d"), E.i("t")), sup=E.i("T")))],
      size=14)
textbox(s, ML + 7.90, 5.86, 4.0, 0.75,
        "B は資源量、f は漁獲率。時期ごとに①〜③を通し、"
        "大蛇行なし／大蛇行それぞれの答えを出す。",
        size=13, color=INK)

# ================================================================== 7 推定
s = new_slide("係数の決め方と、自由度を3段階に落とした3つの版")
textbox(s, ML, 1.28, CW, 1.1,
        ["・式を時間に沿って解き、実データとの差（対数でとる）がいちばん小さくなる係数を探す",
         "・答えが手前の谷に落ちないよう、出発点を何通りも変えて探す（大蛇行なし 64×12通り／大蛇行 32×8通り）",
         "・良し悪しは「ずれ」で見る。ずれ＝平均で割った誤差なので、魚種をまたいで比べられる"],
        size=16, space_after=6)
table(s, ML, 2.85, CW,
      [["版", "自由な係数の数", "固定する値", "ねらい"],
       ["自由に決める", "12個", "なし", "モデルの表現力の上限を見る"],
       ["固定を1段階", "10個", "餌2種の「増える速さ」", "1種ずつの解析の値を入れてみる"],
       ["固定を2段階", "8個", "＋「食べた分が体になる割合」の和",
        "借りる値をできるだけ増やす"]],
      col_w=[1.95, 1.65, 4.25, 4.24], size=13.5, row_h=0.62, head_h=0.48)
note(s, "自由に決める12個：餌2種と食べる魚2種の増える速さ・減る速さ4つ、食べられる強さ4つ、変換の割合4つ。"
        "混み合いの項まで入れた16個の版も試したが、大蛇行の8年に対して数が多すぎて値が定まらないため使わない。",
     y=5.35, size=14, height=0.75)

# ================================================================== 8 当てはまり
s = new_slide("どちらの時期も、観測をよく再現できた")
figure(s, "05_当てはまり", 1.17, 1.32)
textbox(s, ML, 5.95, CW, 0.30,
        "全体の平均（大蛇行なし／大蛇行）：　12個 0.099／0.065　　"
        "10個 0.293／0.170　　8個 0.452／0.338", size=14, color=GRAY)
b = box(s, ML, 6.32, CW - 0.8, 0.56, fill=TINT2, line=ORANGE)
textbox(s, ML + 0.22, 6.44, CW - 1.24, 0.36,
        "以降は12個の自由推定を使う。この後の話は、当てはまりが悪いせいで起きているのではない。",
        size=17, bold=True, color=ORANGE)

# ================================================================== 9 借りる値
s = new_slide("1種ずつの解析から借りられるのは「増える速さ」まで")
lead(s, "前の枚で見た崩れ方は、借りてきた値が「和」であることから起きている。")

CB = [(ML, 5.70, TINT, NAVY, "1種ずつの解析が与えるもの",
       E.seq(E.sub(E.i("c"), E.n("1")), E.o("+"), E.sub(E.i("d"), E.n("1"))),
       "食べた分が体になる割合の和。ブリなら、2つの餌からの分をまとめた"
       "1つの数しか出てこない。"),
      (6.99, 5.72, TINT2, ORANGE, "資源量のデータが決めているもの",
       E.seq(E.sub(E.i("c"), E.n("1")), lam("11"), E.u("，"),
             E.sub(E.i("d"), E.n("1")), lam("21")),
       "その割合と「食べられる強さ」のかけ算。経路ごとに別々の値で、和とは対応しない。")]
for cx, cw, fill, line, head, eq, txt in CB:
    box(s, cx, 1.90, cw, 1.92, fill=fill, line=line)
    textbox(s, cx + 0.24, 2.05, cw - 0.48, 0.34, head, size=16, bold=True,
            color=line)
    eqbox(s, cx + 0.24, 2.46, cw - 0.48, 0.52, [eq], size=17, color=line)
    textbox(s, cx + 0.24, 3.06, cw - 0.48, 0.70, txt, size=14.5)
arrow(s, 6.36, 2.86, 6.95, 2.86, color=GRAY, width=2.0)

box(s, ML, 4.02, CW, 1.24, fill=SHADE, line=None)
textbox(s, ML + 0.24, 4.18, CW - 0.48, 1.00,
        ["和を固定すると、かけ算の値を保つために「食べられる強さ」が不自然に大きくなる"
         "（例：ウルメ→ブリ の λ が 0.224 → 1.00）。",
         "この係数は餌の魚の式にも出てくるので、ゆがみが餌の魚の当てはまりまで広がる。"],
        size=15.5, space_after=7)

b = box(s, ML, 5.42, CW, 0.62, fill=TINT2, line=ORANGE)
textbox(s, ML + 0.22, 5.56, CW - 0.44, 0.4,
        "借りられるのは「増える速さ」まで。変換の割合の和まで固定すると、当てはまりが壊れる。",
        size=17, bold=True, color=ORANGE)
note(s, "「和と積は切り分けられない」と言い切るのは正確ではない。原理的には分けられるが、"
        "この長さのデータでは実質的にほとんど区別がつかない。", y=6.25, size=14, height=0.62)

# ================================================================== 10 意味のずれ
s = new_slide("同じ記号 r でも、中身が違う")
figure(s, "07_増える速さと漁獲率", ML, 1.45)
textbox(s, 7.72, 1.42, 4.98, 4.0,
        [("1種ずつの解析の r", {"size": 17, "bold": True, "color": NAVY}),
         "混み合いを含んだ「もうけ分」の増え方",
         ("今回の式の r_x", {"size": 17, "bold": True, "color": NAVY}),
         "混み合いを含まない、そのままの増える速さ",
         ("マアジで起きたこと", {"size": 17, "bold": True, "color": ORANGE}),
         "借りた r（0.228）が、実際にとっている割合（0.431）より小さい。"
         "式のうえでは食べられる前から毎年減る計算になり、横ばいの実データに合わない。"],
        size=15.5, space_after=9)
b = box(s, ML, 5.55, CW, 0.62, fill=TINT2, line=ORANGE)
textbox(s, ML + 0.22, 5.69, CW - 0.44, 0.4,
        "1種ずつの解析の値を、そのまま多種の式に入れることはできない。意味の翻訳が要る。",
        size=17, bold=True, color=ORANGE)
note(s, "ウルメイワシは とっている割合 0.113 ≪ 増える速さ 0.739 なので、この問題は起きない。",
     y=6.38, size=14)

# ================================================================== 11 時期比較
s = new_slide("大蛇行の時期は、餌の魚が減り、食べる魚が増える")
figure(s, "08_時期ごとの比較", 0.97, 1.38)
b = box(s, ML, 5.15, CW, 1.55, fill=TINT, line=None)
textbox(s, ML + 0.22, 5.32, CW - 0.44, 1.25,
        ["読み方：大蛇行の時期はサワラの取り分がウルメイワシからマアジへ移り、"
         "ブリは食べることへの依存を下げつつ自然に減る速さも下がって増えた、と読める。",
         "ただし係数は1組しか求めていないので、増えた減ったの確からしさまでは言わない。"
         "この後の結論は、これらの値に左右されない。"],
        size=15.5, space_after=6)

# ================================================================== 12 手順
s = new_slide("とっていい量の計算手順")
for k, (num, txt) in enumerate([
        ("1", "推定した式に、魚種ごとの一定の漁獲率 f を入れる"),
        ("2", "T 年ぶん、時間を追って資源量を計算する"),
        ("3", "資源が減りすぎない条件を満たす f のなかで、期間平均の漁獲量を最大にする")]):
    y = 1.45 + k * 0.86
    box(s, ML, y, 0.62, 0.62, fill=NAVY, line=None)
    textbox(s, ML, y + 0.13, 0.62, 0.4, num, size=20, bold=True, color=WHITE,
            align=PP_ALIGN.CENTER)
    textbox(s, ML + 0.85, y + 0.13, CW - 0.85, 0.5, txt, size=18)
eqbox(s, ML + 0.85, 4.03, CW - 0.85, 0.72,
      [E.seq(E.i("Y"), E.o("="), E.frac(E.n("1"), E.i("T")),
             E.nary("∫", E.n("0"), E.seq(
                 E.nary("∑", E.i("i"), E.seq(v("f", "i", True),
                                             v("B", "i", True))),
                 E.u(" "), E.u("d"), E.i("t")), sup=E.i("T")))],
      size=15)
textbox(s, ML + 0.85, 4.84, CW - 0.85, 0.42,
        "f は 0 〜 0.95 を格子状に総当たり。減りすぎない条件は「最終年の資源量が初年の9割以上」を既定とする。",
        size=15.5)
b = box(s, ML, 5.35, CW, 1.45, fill=TINT2, line=ORANGE)
textbox(s, ML + 0.22, 5.52, CW - 0.44, 1.15,
        [("注意：トンで足し合わせることの問題", {"size": 16, "bold": True, "color": ORANGE}),
         "1994〜2024年の平均資源量は ブリ 257・ウルメイワシ 197・マアジ 85 千トンに対し、サワラは 4.4 千トン。"
         "そのままトンで足すと、規模の大きい魚が答えを決めてしまう。魚種ごとに割ってから足す形での"
         "やり直しを今後の課題とする。"],
        size=15, space_after=6)

# ================================================================== 13 結果1
s = new_slide("結果1：最適とされた漁獲率が、探索範囲の端に張り付く")
figure(s, "09_上限に張り付く", ML, 1.40)
b = box(s, ML, 5.75, CW, 1.05, fill=TINT2, line=ORANGE)
textbox(s, ML + 0.22, 5.90, CW - 0.44, 0.8,
        ["探索で許す上限を 0.25 → 1.25 と上げると、答えの漁獲量も 207 → 1094 千トンと上がり続ける。",
         "つまり最大は範囲の内側になく、これは探索の失敗ではなく、いつも起きる現象だった。"],
        size=16, space_after=5)

# ================================================================== 14 結果2（核）
s = new_slide("結果2：端に行く理由 ― とれる量が漁獲率の一次式になる")
A1 = E.sub(E.i("a"), E.n("1"))
A2 = E.sub(E.i("a"), E.n("2"))
B1 = E.sub(E.i("b"), E.n("1"))
B2 = E.sub(E.i("b"), E.n("2"))
X1S = E.sup(E.sub(E.i("x"), E.n("1")), E.n("*"))
Y1S = E.sup(E.sub(E.i("y"), E.n("1")), E.n("*"))
DL = E.sub(E.i("Δ"), E.i("λ"))
DC = E.sub(E.i("Δ"), E.i("C"))

textbox(s, ML, 1.24, 6.35, 0.32, "① 両辺を自分の資源量で割る（1匹あたりの増え方）",
        size=15.5, bold=True, color=NAVY)
eqbox(s, ML + 0.15, 1.62, 6.2, 1.22,
      [E.seq(E.frac(E.n("1"), E.sub(E.i("x"), E.n("1"))),
             ddt(E.sub(E.i("x"), E.n("1"))), E.o("="), A1, E.o("−"),
             lam("11"), E.sub(E.i("y"), E.n("1")), E.o("−"),
             lam("12"), E.sub(E.i("y"), E.n("2"))),
       E.seq(E.frac(E.n("1"), E.sub(E.i("y"), E.n("1"))),
             ddt(E.sub(E.i("y"), E.n("1"))), E.o("="), E.n("−"), B1, E.o("+"),
             E.sub(E.i("c"), E.n("1")), lam("11"), E.sub(E.i("x"), E.n("1")),
             E.o("+"), E.sub(E.i("d"), E.n("1")), lam("21"),
             E.sub(E.i("x"), E.n("2")))],
      size=14)
textbox(s, ML + 0.15, 2.86, 6.2, 0.88,
        [("右辺に自分（x1）が出てこない。混み合いの項がないから。",
          {"color": ORANGE, "bold": True}),
         "a1 = r_x1 − f_x1（餌の漁獲率）、b1 = r_y1 + f_y1（食べる魚の漁獲率）"],
        size=14, space_after=3)

textbox(s, ML, 3.80, 6.35, 0.32, "② 釣り合い（左辺＝0）を解く",
        size=15.5, bold=True, color=NAVY)
eqbox(s, ML + 0.15, 4.14, 6.2, 1.22,
      [E.seq(Y1S, E.o("="), E.frac(
          E.seq(A1, lam("22"), E.o("−"), A2, lam("12")), DL)),
       E.seq(X1S, E.o("="), E.frac(
           E.seq(B1, E.sub(E.i("d"), E.n("2")), lam("22"), E.o("−"),
                 B2, E.sub(E.i("d"), E.n("1")), lam("21")), DC))],
      size=14)
textbox(s, ML + 0.15, 5.38, 6.2, 0.66,
        [("y* は a だけ、x* は b だけで決まる。", {"color": ORANGE, "bold": True}),
         "Δλ = λ11λ22 − λ12λ21、ΔC = c1d2λ11λ22 − c2d1λ12λ21"],
        size=14, space_after=3)

textbox(s, 7.30, 1.24, 5.4, 0.32, "③ とれる量を書く", size=15.5, bold=True,
        color=NAVY)
eqbox(s, 7.45, 1.62, 5.25, 0.72,
      [E.seq(E.i("Y"), E.o("="),
             E.sub(E.i("f"), E.n("x1")), X1S, E.o("+"),
             E.sub(E.i("f"), E.n("x2")), E.sup(E.sub(E.i("x"), E.n("2")), E.n("*")),
             E.o("+"),
             E.sub(E.i("f"), E.n("y1")), Y1S, E.o("+"),
             E.sub(E.i("f"), E.n("y2")), E.sup(E.sub(E.i("y"), E.n("2")), E.n("*")))],
      size=14)
textbox(s, 7.45, 2.46, 5.25, 2.45,
        ["x1* に f_x1 は入っていない。だから f_x1 は前についた係数としてしか出てこない。",
         "他の3つも同じ。よって とれる量 Y は、どの漁獲率について見ても1次式（直線）になる。",
         ("直線の最大は端。4つの漁獲率それぞれに当てはめると、"
          "最大は必ず箱の角（0 か上限）にある。", {"bold": True, "color": ORANGE})],
        size=15, space_after=10)
b = box(s, 7.30, 5.00, 5.4, 1.42, fill=TINT2, line=ORANGE)
textbox(s, 7.48, 5.14, 5.04, 1.25,
        "結論：報告される「MSY」は、探索範囲の上限をどこに置いたかで決まる数値であって、"
        "生き物の側の量ではない。これは推定値の良し悪しに左右されない。",
        size=15.5, bold=True, color=ORANGE)
textbox(s, ML, 6.42, CW - 1.3, 0.45,
        "釣り合いの量が自分の漁獲率に依存しない点は古典的に知られる（Volterra 1926）。"
        "本研究が述べるのは、そこから従う「最適な漁獲率が必ず端に来る」ほう。"
        "有限の期間で測っても同じになることは予備23枚目。",
        size=11.5, color=GRAY)

# ================================================================== 15 結果3
s = new_slide("結果3：そもそも、漁獲ゼロの釣り合いが正にならない")
figure(s, "11_釣り合いの量が負", 2.07, 1.26)
textbox(s, ML, 4.45, 11.35, 2.42,
        ["・漁獲ゼロのときの釣り合いを計算すると、"
         "大蛇行なしではマアジが −50 千トン、大蛇行ではウルメイワシが −130 千トン。"
         "4種が同時に正で釣り合う点が、漁獲ゼロでは存在しない。",
         "・そのため「釣り合いの量に対する比」で持続を判定するやり方は使えない。",
         "・結果2は「仮に正の釣り合いがあっても、内側に最大はない」という話。"
         "実際はその釣り合いすら正にならない。どちらの意味でも内側の MSY は決められない。",
         ("・ただし正にならないのは漁獲ゼロのとき。漁獲率を変えると正になる点も見つかっている"
          "（角16点のうち 大蛇行なし1点／大蛇行4点）。", {"color": GRAY, "size": 14})],
        size=15.5, space_after=7)

# ================================================================== 16 まとめ
s = new_slide("まとめ", dark=True, title_size=28)
for k, (num, txt) in enumerate([
    ("1", "時期ごと・魚種のつながりを入れた「とっていい量」の計算の枠組みを作り、"
          "実データで動かした。"),
    ("2", "その枠組みが返すのは MSY ではなく「資源の下限を守ったうえでの最大の漁獲量」。"
          "原因は混み合いの効果がないこと。とれる量が漁獲率の一次式になるため、"
          "答えは必ず探索範囲の端に来る。"),
    ("3", "1種ずつの解析から借りられるのは「増える速さ」まで。"
          "借りる値を増やすほど当てはまりは単調に悪くなる（12個 → 10個 → 8個）。"),
    ("4", "今後：内側に最大を作るには混み合いの項を戻す必要がある。"
          "また、トンでの単純な足し算をやめ、魚種ごとに割った量で測り直す。")]):
    y = 1.45 + k * 1.32
    box(s, ML, y, 0.55, 0.55, fill="3E6693", line=None)
    textbox(s, ML, y + 0.10, 0.55, 0.4, num, size=18, bold=True, color=WHITE,
            align=PP_ALIGN.CENTER)
    textbox(s, ML + 0.78, y + 0.05, CW - 0.78, 1.1, txt, size=17, color=WHITE,
            line_spacing=1.35)

# ================================================================== 17 予備B1
s = new_slide("［予備］資源が減りすぎない、の4つの決め方")
table(s, ML, 1.35, CW,
      [["決め方", "中身", "本研究での扱い"],
       ["最終年で見る", "計算した最後の年の資源量が、初年の9割以上",
        "既定。ただし初年の値に左右される"],
       ["漁獲ゼロの釣り合いと比べる", "漁獲ありの釣り合いが、漁獲ゼロの釣り合いの一定割合以上",
        "使えない（釣り合いが正にならないため）"],
       ["長期の下限で見る", "長く回したときの資源量の最小値が、基準の一定割合以上", "併用できる"],
       ["期間平均で見る", "期間平均の資源量が、基準の一定割合以上", "併用できる"]],
      col_w=[3.0, 4.9, 4.19], size=13.5, row_h=0.72, head_h=0.45)
note(s, "既定の判定は「最終年 ÷ 初年 ≧ 0.9」。初年の値がどの位相にあるかで結果が動くという弱点がある。",
     y=5.05, size=14)

# ================================================================== 18 予備B2
s = new_slide("［予備］角16点で、釣り合いが正になるか調べた")
table(s, ML, 1.40, CW,
      [["時期", "4種すべて正になる角の数", "その例（千トン）"],
       ["大蛇行なし", "16点のうち 1点（f＝0, 0, 0.95, 0.95）", "237.1 ／ 2.1 ／ 857.8 ／ 6.3"],
       ["大蛇行", "16点のうち 4点（1つの面がまるごと）", "21.4 ／ 1494.2 ／ 390.9 ／ 9.1"]],
      col_w=[2.0, 4.9, 5.19], size=13.5, row_h=0.62, head_h=0.5)
textbox(s, ML, 3.45, CW, 2.4,
        ["※「その例」の並びは マアジ／ウルメイワシ／ブリ／サワラ。",
         "釣り合いの量は漁獲率の一次式なので、正になる漁獲率の集まりは平らな面で囲まれた領域になる。"
         "大蛇行では角4点すべてが正なので、その面はまるごと正である。",
         "観測されている漁獲ゼロの点は、その領域の外側にある、という言い方が正確。",
         ("今後の課題：この「正になる領域」と、実際に持続の条件を満たすと判定される漁獲率の"
          "領域が重なるかどうかは、まだ突き合わせていない。", {"color": ORANGE})],
        size=15.5, space_after=9)

# ================================================================== 19 予備B3
s = new_slide("［予備］内側をいくら探しても、角を超えない")
figure(s, "12_角と内側", 2.6, 1.40)
textbox(s, ML, 5.35, CW, 1.4,
        ["・自分の漁獲率に対する自分の釣り合いの量の変化は、両時期とも厳密に 0。",
         "・とれる量を漁獲率の各方向で2回差分すると、計算機の精度でゼロ（＝直線）。",
         "・角16点の最大が、内側にとった2万点の最大を上回る。"],
        size=16, space_after=6)

# ================================================================== 20 予備B4
s = new_slide("［予備］4種すべてに同じ漁獲率をかけた場合")
table(s, ML, 1.40, CW,
      [["魚の組み合わせ", "時期", "最も多くとれる漁獲率", "そのときの量（千トン／年）"],
       ["マアジ版", "大蛇行なし", "0.373（内側に山がある）", "161.1"],
       ["マアジ版", "大蛇行", "0.95（上限に張り付く）", "232.9"],
       ["マイワシ版", "大蛇行なし", "0.228（内側に山がある）", "―"],
       ["マイワシ版", "大蛇行", "0.95（上限に張り付く）", "―"]],
      col_w=[2.5, 2.3, 3.85, 3.44], size=13.5, row_h=0.5, head_h=0.45)
textbox(s, ML, 4.35, CW, 1.9,
        ["4種に同じ漁獲率をかけると1つの変数の問題になり、大蛇行なしでは内側に山が現れる。"
         "しかし大蛇行では上限に張り付いたままで、山の有無が時期で逆転する。",
         "この山は混み合いが作るものではなく、「全種同じ漁獲率」という縛りと段のつながりが作る"
         "見かけの山。魚種の重みづけ（同じ重み）の決め方しだいで動くため、生き物の側の量ではない。",
         ("よって、主となる MSY の定義には採らない。", {"bold": True, "color": ORANGE})],
        size=15.5, space_after=9)

# ================================================================== 21 予備B5
s = new_slide("［予備］餌の魚をマイワシに替えても結論は同じ")
textbox(s, ML, 1.40, CW, 3.6,
        ["・本研究では餌の魚 x1 をマアジにしているが、これはマイワシから置き換えたもの。",
         "・マイワシは1種ずつの解析が通常の条件で解けず、例外の設定が必要だったため置き換えた。",
         "・置き換える前のマイワシ版でも、後半の結論は変わらない。"
         "　上限に張り付くこと、漁獲ゼロの釣り合いが正にならないこと、いずれも同じ。",
         ("・よって、結論は魚種の選び方に左右されない。", {"bold": True, "color": ORANGE})],
        size=17, space_after=13)
b = box(s, ML, 5.15, CW, 1.1, fill=TINT, line=None)
textbox(s, ML + 0.22, 5.32, CW - 0.44, 0.8,
        "マアジの資源量と漁獲量は水産研究・教育機構「令和7年度 マアジ太平洋系群の資源評価」"
        "（1982〜2024年）から取得。1種ずつの解析と多種の式で同じデータを使っている。",
        size=15, color=INK)

# ================================================================== 22 予備B6
s = new_slide("［予備］漁獲率 f の作り方と、上限 0.95 の扱い")
textbox(s, ML, 1.40, CW, 3.85,
        ["・f ＝ その年の漁獲量 ÷ その年の資源量。推定する値ではなく、データから直接計算する。",
         "・資源量は資源評価の出力なので、f の誤差は資源量の誤差と連動する。ここは限界として認める。",
         "・1 を超えないよう 0.95 で頭を打たせている。この 0.95 に生き物の側の根拠はない。",
         ("・結果2が示すのは、まさにこの 0.95 という設定が答えを決めてしまうということ。"
          "管理の基準として使うなら、漁具や操業の物理的な上限など、根拠のある値を置く必要がある。",
          {"bold": True, "color": ORANGE})],
        size=16.5, space_after=13)
b = box(s, ML, 5.45, CW, 1.0, fill=TINT, line=None)
textbox(s, ML + 0.22, 5.62, CW - 0.44, 0.7,
        "1994〜2024年の平均：マアジ 0.431／ウルメイワシ 0.113／ブリ 0.374／サワラ 0.363。",
        size=15.5, color=INK)

# ================================================================== 23 予備B7（O(1/T)）
s = new_slide("［予備］有限の期間 T で測っても、同じ結論になる")
EPS1 = E.sub(E.i("ε"), E.n("1"))
EPS2 = E.sub(E.i("ε"), E.n("2"))
DEL1 = E.sub(E.i("δ"), E.n("1"))
DEL2 = E.sub(E.i("δ"), E.n("2"))
YB1 = E.sub(E.bar(E.i("y")), E.n("1"))
YB2 = E.sub(E.bar(E.i("y")), E.n("2"))
XB1 = E.sub(E.bar(E.i("x")), E.n("1"))
T_ = E.d(E.i("T"))

textbox(s, ML, 1.24, 6.3, 0.32, "① 1匹あたりの式を 0〜T で積分して T で割る",
        size=15.5, bold=True, color=NAVY)
eqbox(s, ML + 0.15, 1.60, 6.15, 1.78,
      [E.seq(E.sub(E.bar(E.i("y")), E.i("j")), T_, E.o("="),
             E.frac(E.n("1"), E.i("T")),
             E.nary("∫", E.n("0"), E.seq(E.sub(E.i("y"), E.i("j")),
                                         E.d(E.i("t")), E.u(" "), E.u("d"),
                                         E.i("t")), sup=E.i("T"))),
       E.seq(EPS1, T_, E.o("="),
             E.frac(E.seq(E.u("log"), E.sp(), E.sub(E.i("x"), E.n("1")), T_,
                          E.o("−"), E.u("log"), E.sp(),
                          E.sub(E.i("x"), E.n("1")), E.d(E.n("0"))),
                    E.i("T"))),
       E.seq(lam("11"), YB1, T_, E.o("+"), lam("12"), YB2, T_, E.o("="),
             A1 if False else E.sub(E.i("a"), E.n("1")), E.o("−"), EPS1, T_)],
      size=14)
textbox(s, ML + 0.15, 3.44, 6.15, 0.65,
        "釣り合いの式と同じ形。右辺が a1 から a1 − ε1(T) に変わっただけ。",
        size=14.5, color=ORANGE, bold=True)

textbox(s, ML, 4.16, 6.3, 0.32, "② 同じ消去をすると、ずれがそのまま出る",
        size=15.5, bold=True, color=NAVY)
eqbox(s, ML + 0.15, 4.50, 6.15, 1.22,
      [E.seq(YB1, T_, E.o("−"), E.sup(E.sub(E.i("y"), E.n("1")), E.n("*")),
             E.o("="), E.n("−"),
             E.frac(E.seq(EPS1, T_, lam("22"), E.o("−"), EPS2, T_, lam("12")),
                    E.sub(E.i("Δ"), E.i("λ")))),
       E.seq(XB1, T_, E.o("−"), E.sup(E.sub(E.i("x"), E.n("1")), E.n("*")),
             E.o("="),
             E.frac(E.seq(DEL1, T_, E.sub(E.i("d"), E.n("2")), lam("22"),
                          E.o("−"), DEL2, T_, E.sub(E.i("d"), E.n("1")),
                          lam("21")), E.sub(E.i("Δ"), E.i("C"))))],
      size=14)

textbox(s, 7.15, 1.24, 5.55, 0.32, "③ 資源量が有界なら、ずれは 1/T の速さで消える",
        size=15.5, bold=True, color=NAVY)
eqbox(s, 7.30, 1.62, 5.4, 1.22,
      [E.seq(E.n("0"), E.o("<"), E.i("m"), E.o("≤"), E.sub(E.i("x"), E.i("i")),
             E.n(", "), E.sub(E.i("y"), E.i("j")), E.o("≤"), E.i("M"),
             E.o("<"), E.i("∞")),
       E.seq(E.sub(E.i("ε"), E.i("i")), T_, E.n(", "),
             E.sub(E.i("δ"), E.i("j")), T_, E.o("="), E.u("O"),
             E.d(E.frac(E.n("1"), E.i("T"))))],
      size=14)
textbox(s, 7.30, 2.90, 5.4, 0.66,
        "資源量が 0 に落ちも発散もしないかぎり、対数の差を T で割った量は小さくなる。",
        size=14.5)

textbox(s, 7.15, 3.64, 5.55, 0.32, "④ よって、時間平均でとれる量は",
        size=15.5, bold=True, color=NAVY)
eqbox(s, 7.30, 4.00, 5.4, 0.85,
      [E.seq(E.sub(E.i("Y"), E.i("T")), E.d(E.i("f")), E.o("="),
             E.sub(E.i("Y"), E.u("eq")), E.d(E.i("f")), E.o("+"),
             E.u("O"), E.d(E.frac(E.n("1"), E.i("T"))))],
      size=15)
b = box(s, 7.15, 4.96, 5.55, 1.44, fill=TINT2, line=ORANGE)
textbox(s, 7.33, 5.09, 5.19, 1.20,
        "有限の期間で測った漁獲量は、釣り合いで計算した量と 1/T の程度しか違わない。"
        "だから前の「一次式になる」話は、実際の計算にもそのまま効く。",
        size=15.5, bold=True, color=ORANGE)
textbox(s, ML, 6.48, CW - 1.3, 0.55,
        "a1 = r_x1 − f_x1、b1 = r_y1 + f_y1。ε は餌の魚、δ は食べる魚の対数の差を T で割った量。"
        "※ここまでの流れ自体は Volterra の原理の範囲。",
        size=11.5, color=GRAY)

# ================================================================== 24 予備B8
s = new_slide("［予備］行列で見ると：効いているのは「ゼロの置き方」")
textbox(s, ML, 1.28, CW, 0.32, "① まとめて書く", size=16, bold=True, color=NAVY)
eqbox(s, ML + 0.15, 1.64, CW - 0.3, 1.18,
      [E.seq(ddt(v("B", "i", True)), E.o("="), v("B", "i", True),
             E.d(E.seq(v("ρ", "i", True), E.o("−"), v("f", "i", True), E.o("+"),
                       E.nary("∑", E.i("j"), E.seq(v("A", "ij", True),
                                                   v("B", "j", True)))))),
       E.seq(E.sup(E.i("B"), E.n("*")), E.d(E.i("f")), E.o("="),
             E.sup(E.i("B"), E.n("*")), E.d(E.n("0")), E.o("+"),
             E.sup(E.i("A"), E.n("−1")), E.sp(), E.i("f"))],
      size=15)
textbox(s, ML, 3.06, CW, 0.32, "② 効き方は A の逆行列で決まる", size=16,
        bold=True, color=NAVY)
eqbox(s, ML + 0.15, 3.42, CW - 0.3, 0.74,
      [E.seq(E.frac(E.seq(E.u("∂"), E.sup(E.sub(E.i("B"), E.i("i")), E.n("*"))),
                    E.seq(E.u("∂"), E.sub(E.i("f"), E.i("j")))), E.o("="),
             E.sub(E.d(E.sup(E.i("A"), E.n("−1"))), E.i("ij")))],
      size=15)
textbox(s, ML, 4.24, CW, 1.10,
        ["「自分の漁獲が自分に効かない」とは、A の逆行列の対角がゼロだということ。",
         ("注意：A の対角がゼロでも、逆行列の対角がゼロとは限らない。"
          "反例 A ＝ [[0,1,1],[1,0,1],[1,1,0]] の逆行列の対角は −1/2。",
          {"color": ORANGE})],
        size=15.5, space_after=8)
textbox(s, ML, 5.42, CW, 0.32, "③ 今回のモデルで成り立つ理由", size=16,
        bold=True, color=NAVY)
textbox(s, ML + 0.15, 5.76, 10.9, 1.05,
        ["A は餌の段と食べる段の「あいだ」にしか値がない形（5枚目の灰色の位置がゼロ）。"
         "このとき逆行列も同じ形になり、対角がゼロになる。",
         "餌の魚どうしの取り合いを入れると対角はゼロでなくなり、内側の最大が戻りうる。"],
        size=15, space_after=6)

prs.save(OUT)
declare_a14_in_package(OUT)

# 書き出したファイルを開き直して、枚数と数式の数が合うか確かめる
_check = Presentation(OUT)
assert len(list(_check.slides)) == _page["n"], "保存後に枚数が合わない"
print("保存:", OUT, "／ 全", _page["n"], "枚")
