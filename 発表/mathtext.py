"""OMML（数式）段落を、点検用に平文へ直す。"""
M = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"
A14 = "{http://schemas.microsoft.com/office/drawing/2010/main}"


def omath_of(p):
    """段落 p の中の m:oMath を返す（無ければ None）。"""
    for el in p._p.iter():
        if el.tag == M + "oMath":
            return el
    return None


def linearize(e):
    t = e.tag
    if t == M + "f":
        return "(" + linearize(e.find(M + "num")) + ")/(" \
               + linearize(e.find(M + "den")) + ")"
    if t == M + "sSub":
        return linearize(e.find(M + "e")) + "_" + linearize(e.find(M + "sub"))
    if t == M + "sSup":
        return linearize(e.find(M + "e")) + "^" + linearize(e.find(M + "sup"))
    if t == M + "sSubSup":
        return (linearize(e.find(M + "e")) + "_" + linearize(e.find(M + "sub"))
                + "^" + linearize(e.find(M + "sup")))
    if t == M + "nary":
        pr = e.find(M + "naryPr")
        ch = pr.find(M + "chr") if pr is not None else None
        c = ch.get(M + "val") if ch is not None else "\u2211"
        return (c + linearize(e.find(M + "sub")) + linearize(e.find(M + "sup"))
                + " " + linearize(e.find(M + "e")))
    if t == M + "d":
        return "(" + linearize(e.find(M + "e")) + ")"
    if t == M + "bar":
        return linearize(e.find(M + "e")) + "\u0304"
    if t == M + "t":
        return e.text or ""
    skip = (M + "rPr", M + "ctrlPr", M + "fPr", M + "naryPr", M + "dPr",
            M + "barPr", M + "sSubPr", M + "sSupPr", M + "sSubSupPr",
            M + "mPr", M + "argPr")
    return "".join(linearize(c) for c in e
                   if c.tag not in skip and not str(c.tag).endswith("}rPr"))


def eq_size(p, default=16.0):
    for el in p._p.iter():
        if str(el.tag).endswith("}endParaRPr") and el.get("sz"):
            return float(el.get("sz")) / 100.0
    return default


def eq_height_lines(om):
    """数式の高さを「本文何行ぶんか」で見積もる。分数や積分記号があると背が高い。"""
    tall = any(e.tag in (M + "f", M + "nary") for e in om.iter())
    return 2.25 if tall else 1.20
