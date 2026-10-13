"""
把评测结果导出成 Word（.docx）。

为什么不手写文档：作业要的是「逐题记录召回了哪些块、答对没有、错在哪」，
这些数据本来就在 data/eval_raw.json 里，人工复核结论在 eval.py 的 VERDICTS 里。
手抄一遍必然和源头脱节 —— 改了结论忘了改文档，交上去的就是两个版本。
所以从数据生成，随时可重跑。

为什么自己拼 OOXML 而不用 textutil：
    这台机器没装 pandoc。试过用系统自带的 `textutil -convert docx`，
    结果**表格全被拍平成段落**（查生成的 docx，<w:tbl> 数量为 0），
    「召回块清单」会散成一行一个单元格，没法读；标题也不带 Heading 样式，
    导航窗格跳不了。
docx 本身只是个 zip + 几个 XML，自己拼反而干净：无依赖、离线、
表格边框/列宽/字体全可控。

跑法：
    .venv/bin/python make_doc.py
产出：
    白酒财报问答_10道评测题.docx   逐题评测记录（末尾内嵌总体结论）
    结论.docx                      单独一份总体结论，供作业提交
"""

from __future__ import annotations

import html
import json
import re
import sys
import zipfile
from pathlib import Path

import config as C
from eval import QUESTIONS, VERDICTS

OUT = C.ROOT / "白酒财报问答_10道评测题.docx"
CONCL = C.ROOT / "结论.docx"

# A4 纵向，页边距 2cm。可用宽度 = 11906 - 1134*2 = 9638 twips，取 9600。
PAGE_W, PAGE_H, MARGIN, TEXT_W = 11906, 16838, 1134, 9600
SONG, HEI = "宋体", "黑体"


def esc(s) -> str:
    return html.escape(str(s))


# ---------------------------------------------------------------- 行内

def runs(text: str) -> list[tuple[str, bool]]:
    """把 `**粗体**` 切成 [(文本, 是否粗体)]。返回的文本已转义。"""
    out, i = [], 0
    for m in re.finditer(r"\*\*(.+?)\*\*", text):
        if m.start() > i:
            out.append((text[i:m.start()], False))
        out.append((m.group(1), True))
        i = m.end()
    if i < len(text):
        out.append((text[i:], False))
    return out or [("", False)]


def _rpr(size: int, bold: bool, color: str | None, font: str) -> str:
    r = (f'<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" '
         f'w:eastAsia="{font}"/>')
    if bold:
        r += "<w:b/>"
    r += f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>'
    if color:
        r += f'<w:color w:val="{color}"/>'
    return f"<w:rPr>{r}</w:rPr>"


def para(text: str, size: int = 21, bold: bool = False, color: str | None = None,
         font: str = SONG, before: int = 0, after: int = 90,
         center: bool = False, indent: int = 0, style: str | None = None) -> str:
    """一个段落。size 是半磅（21 = 10.5pt，中文文档的正文字号）。

    style：挂 Heading1/2/3 样式。挂了之后字号字体颜色由样式给，run 级别
    不能再写 rPr —— 写了就覆盖样式，导航窗格认得标题、显示出来却是正文大小。
    只保留行内 `**粗体**`。

    注意 pPr 里各元素的顺序是 schema 定死的（pStyle → pBdr → spacing → ind → jc），
    顺序错了 Word 打开会报「内容有问题」。ind 必须排在 jc 前面。
    """
    rs = "".join(
        f'<w:r>{_rpr(size, bold or b, color, font)}'
        f'<w:t xml:space="preserve">{esc(t)}</w:t></w:r>'
        for t, b in runs(text))
    if style:
        rs = "".join(
            f'<w:r>{"<w:rPr><w:b/></w:rPr>" if b else ""}'
            f'<w:t xml:space="preserve">{esc(t)}</w:t></w:r>'
            for t, b in runs(text))
    ppr = f"<w:pPr>{f'<w:pStyle w:val=\"{style}\"/>' if style else ''}"
    ppr += (f'<w:spacing w:before="{before}" w:after="{after}"'
            f' w:line="300" w:lineRule="auto"/>')
    if indent:
        ppr += f'<w:ind w:left="{indent}"/>'
    if center:
        ppr += '<w:jc w:val="center"/>'
    ppr += "</w:pPr>"
    return f"<w:p>{ppr}{rs or ''}</w:p>"


def hrule() -> str:
    return ('<w:p><w:pPr><w:pBdr><w:bottom w:val="single" w:sz="6" '
            'w:space="1" w:color="BBBBBB"/></w:pBdr>'
            '<w:spacing w:before="60" w:after="60"/></w:pPr></w:p>')


# ---------------------------------------------------------------- 表格

def table(header: list[str], body: list[list[str]],
          widths: list[int] | None = None, size: int = 18) -> str:
    n = len(header)
    w = widths or [TEXT_W // n] * n
    w = w[:n] + [TEXT_W // n] * max(0, n - len(w))

    borders = ('<w:tblBorders>'
               + "".join(f'<w:{s} w:val="single" w:sz="4" w:space="0" '
                         f'w:color="AAAAAA"/>'
                         for s in ("top", "left", "bottom", "right",
                                   "insideH", "insideV"))
               + "</w:tblBorders>")
    out = [f'<w:tbl><w:tblPr><w:tblW w:w="{TEXT_W}" w:type="dxa"/>{borders}'
           f'<w:tblLayout w:type="fixed"/></w:tblPr>',
           "<w:tblGrid>" + "".join(f'<w:gridCol w:w="{c}"/>' for c in w)
           + "</w:tblGrid>"]

    def row(cells, bold=False, shade=None):
        tcs = []
        for i, c in enumerate(cells):
            shd = f'<w:shd w:val="clear" w:fill="{shade}"/>' if shade else ""
            tcpr = (f'<w:tcPr><w:tcW w:w="{w[i]}" w:type="dxa"/>{shd}'
                    f'<w:vAlign w:val="top"/></w:tcPr>')
            tcs.append(f"<w:tc>{tcpr}{para(c, size=size, bold=bold, after=0)}</w:tc>")
        return "<w:tr>" + "".join(tcs) + "</w:tr>"

    out.append(row(header, bold=True, shade="F2F2F2"))
    for r in body:
        out.append(row(r))
    out.append("</w:tbl>")
    out.append(para("", size=6, after=0))       # 表后一点空隙
    return "".join(out)


# ---------------------------------------------------------------- markdown

def md_blocks(md: str, shift: int = 0) -> list[tuple]:
    """极简 markdown → 块列表。只支持 结论.md 里实际用到的语法。

    shift：标题整体降几级。结论.md 是独立文档，标题从 # 起；嵌进这篇里
    要挂到「四、总体结论」下面，不降级就和文档自己的章节编号撞车。
    """
    out, lines, i = [], md.split("\n"), 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip("|").split("|")])
                i += 1
            rows = [r for r in rows
                    if not all(re.fullmatch(r":?-{2,}:?", c) for c in r)]
            if rows:
                out.append(("table", rows[0], rows[1:]))
            continue
        if ln.startswith("#"):
            lvl = min(len(ln) - len(ln.lstrip("#")) + shift, 6)
            out.append(("h", lvl, ln.lstrip("#").strip()))
        elif ln.startswith("> "):
            out.append(("bq", ln[2:]))
        elif ln.startswith("- "):
            out.append(("li", ln[2:]))
        elif ln.strip() in ("---", "***"):
            out.append(("hr",))
        elif ln.strip():
            out.append(("p", ln.strip()))
        i += 1
    return out


def render(blocks: list[tuple]) -> str:
    out = []
    for b in blocks:
        k = b[0]
        if k == "h":
            lvl = min(b[1], 4)
            out.append(para(b[2], style=f"Heading{lvl}"))
            if lvl == 1:
                out.append(hrule())
        elif k == "p":
            out.append(para(b[1]))
        elif k == "meta":
            out.append(para(b[1], size=18, color="666666"))
        elif k == "bq":
            out.append(para(b[1], size=19, color="333333", indent=340))
        elif k == "li":
            out.append(para("· " + b[1], indent=200))
        elif k == "hr":
            out.append(hrule())
        elif k == "table":
            # 表格块是 ("table", 表头, 表体) 或 ("table", 表头, 表体, 列宽)
            out.append(table(b[1], b[2], widths=b[3] if len(b) > 3 else None,
                             size=17))
    return "".join(out)


# ---------------------------------------------------------------- 组装

_COVER_MARK = {"covered": "✅ 已覆盖", "partial": "⚠️ 部分覆盖", "missing": "❌ 缺据"}
# 一行汇总里用的短词。缺据和部分覆盖得分开说 —— 混成一个「缺」字，
# 会把「取到值、只是没取到同比那一列」写成「什么都没取到」。
_BAD_WORD = {"partial": "部分覆盖", "missing": "缺据"}


def coverage_summary_line(rows: list[dict]) -> str:
    """总览下面那句结论。覆盖维度的结果在这里跟召回/回答并排说清楚。"""
    n_ok = sum(1 for x in rows if x["answer_hit"] is True)
    quad = []
    for x in rows:
        r, a = x["recall_hit"], x["answer_hit"]
        if r is False and a is False:
            quad.append(f"Q{x['id']} 属检索失败（块没召回，模型也答不出）")
        elif r is True and a is False:
            quad.append(f"Q{x['id']} 属生成失败（块在，模型没用对）")
    pano = [x for x in rows if x.get("n_expected")]
    miss = []
    for x in pano:
        bad = [f"{e['company']} · {_BAD_WORD.get(e['status'], e['status'])}"
               for e in (x.get("coverage") or []) if e["status"] != "covered"]
        if bad:
            miss.append(f"Q{x['id']} {'、'.join(bad)}")
    s = (f"答对 {n_ok} 题；" + "；".join(quad) + "。") if quad else f"答对 {n_ok} 题。"
    if pano:
        n_pc = sum(1 for x in pano if x.get("coverage_hit"))
        s += (f"全景题覆盖通过 {n_pc}/{len(pano)} 题"
              + (f"（{'；'.join(miss)}）" if miss else "") + "。")
    return s


def build_blocks(rows: list[dict]) -> list[tuple]:
    ok = {True: "✓", False: "✗", None: "—"}
    B: list[tuple] = []

    B.append(("h", 1, "白酒财报问答知识库 · 10 道评测题"))

    # ---------------------------------------------------------- 说明
    B.append(("h", 2, "一、评测设置"))
    B += [
        ("li", "**语料**：16 家 A 股白酒公司 ×（2026 半年报 + 2025 年报）"
               "＝ 32 份财报全文，来源巨潮资讯网（cninfo.com.cn）"),
        ("li", "**规模**：15,017 块（正文 6,015 / 表格 9,002），"
               "每块带公司、报告、章节、页码、单位"),
        ("li", "**检索**：向量（百炼 text-embedding-v4，1024 维）＋ BM25 → "
               "RRF 融合 → 同公司配额（每家最多 3 块）；"
               "全景题另走「按公司取主要会计数据块」的锚点通道"),
        ("li", "**生成**：qwen-plus，temperature 0.1，只依据给定片段作答"),
    ]
    B.append(("p", "**判定口径** —— 每题分开判三个信号。只看最终答案对不对，"
                   "无法区分「没检索到」「检索到了但读错」和"
                   "「有一家公司压根没进视野」："))
    B.append(("table", ["信号", "含义"],
              [["召回", "召回的块里含标准答案"],
               ["回答", "模型最终回答里含标准答案"],
               ["覆盖", "全景题按公司取证时，该取到的公司是不是都取到了被问的那个字段"
                        "（单点题没有这一维）"]],
              [900, 8700]))

    # ---------------------------------------------------------- 总览
    n_ok = sum(1 for x in rows if x["answer_hit"] is True)
    n_rec = sum(1 for x in rows if x["recall_hit"] is True)
    pano = [x for x in rows if x.get("n_expected")]
    B.append(("h", 2, "二、总览"))
    B.append(("p", f"共 10 题（含 3 道跨公司全景题）。"
                   f"**召回命中 {n_rec}/10，回答命中 {n_ok}/10。**"))
    B.append(("table", ["题号", "考点", "召回", "回答", "覆盖"],
              [[f"Q{x['id']}", x["type"], ok[x["recall_hit"]],
                ok[x["answer_hit"]],
                "—" if not x.get("n_expected")
                else f"{ok[x.get('coverage_hit')]} "
                     f"{x.get('n_covered')}/{x.get('n_expected')}"]
               for x in rows],
              [900, 5100, 900, 900, 1800]))
    B.append(("meta", coverage_summary_line(rows)))

    # ---------------------------------------------------------- 覆盖地图
    B.append(("h", 2, "三、覆盖地图"))
    B += [
        ("p", "全景题不能只靠相似度排序 —— 那样每家公司取到的块质量参差，"
              "取不到的那家**不会报错，只会静默消失**。所以改按公司取证："
              "对每家公司先拿候选块，再校验**被问的那个字段到底在不在**，"
              "结果记成一张覆盖地图（✅ 已覆盖 / ⚠️ 部分覆盖 / ❌ 缺据），"
              "连取数出处、项目名、期次、口径一起交给模型。"
              "「缺据」不是在检讨检索，而是把「这家本该有数据、这次没取到」"
              "明说出来 —— 让模型不许拿行业常识或别家的数替它补上。"),
        ("p", "**口径**这一列要分两种读法：「合并（第二节主表·规则推定）」是"
              "按披露制式推的 —— 第二节「主要会计数据」按规则就是合并数，"
              "不是从行文里猜的；「未标注」是**取数出处那块表里真的分不出**"
              "合并还是母公司（第八节的合并报表与母公司报表，提取出来的块"
              "前缀一模一样），不猜，如实标。"),
    ]
    for x in pano:
        if not x.get("coverage"):
            continue
        B.append(("h", 3, f"Q{x['id']}　{x['type']}"))
        flds = "、".join(sorted({e["field"] for e in x["coverage"] if e["field"]}))
        B.append(("p", f"问的是「{flds or '（未识别出字段）'}」，"
                       f"取证范围 {x.get('coverage_scope') or '未标注'}。"
                       f"**已覆盖 {x['n_covered']} 家、部分 {x['n_partial']} 家、"
                       f"缺据 {x['n_missing']} 家。**"))
        B.append(("table",
                  # 「同比」单列一栏：筛选题（"哪几家正增长"）能据以判断的就是这一列，
                  # 不是营业收入绝对值。挤进「值」那一格写过一次，模型照样漏公司。
                  ["公司", "状态", "取数出处", "项目", "值", "同比", "期次", "口径", "说明"],
                  [[f"{e['company']}（{e['code']}）",
                    _COVER_MARK.get(e["status"], e["status"]),
                    (f"p{e['page']} · {e['source']}" if e.get("page") else "—"),
                    e["field"] or "—", e["value"] or "—", e.get("yoy") or "—",
                    e["period"] or "—",
                    e["scope"], e["note"] or ""] for e in x["coverage"]],
                  # 合计必须正好 TEXT_W=9600，多一列就从别处匀
                  [1200, 800, 850, 1100, 1250, 750, 700, 1550, 1400]))
        bad = [e["company"] for e in x["coverage"] if e["status"] != "covered"]
        if bad:
            B.append(("meta", f"⚠️ 覆盖不完整，缺的是：{'、'.join(bad)}。"))

    # ---------------------------------------------------------- 逐题
    B.append(("h", 2, "四、逐题记录"))
    for x in rows:
        B.append(("h", 3, f"Q{x['id']}　{x['type']}"))
        B.append(("p", f"**问题**：{x['q']}"))
        B.append(("p", f"**标准答案**：{x['gold_desc']}"))
        if x["note"]:
            B.append(("meta", f"**出题意图**：{x['note']}"))

        head = f"**召回了哪些块**：共 {x['n_total']} 块"
        if x["panorama"]:
            head += f"（检索 {x['n_search']} 块 ＋ 全景通道锚点，已去重）"
        if x.get("n_expected"):
            head += (f"；覆盖 **{x['n_covered']}/{x['n_expected']}** 家"
                     f"（部分 {x['n_partial']}、缺据 {x['n_missing']}，"
                     f"逐家清单见「三、覆盖地图」）")
        B.append(("p", head))
        B.append(("table",
                  ["#", "出处（公司 · 报告 · 章节 · 页）", "类型", "单位",
                   "得分", "内容摘要"],
                  [[str(h["n"]),
                    f"{h['company']} · {h['report']} · "
                    f"{h['section'] or '未定'} · 第 {h['page']} 页",
                    h["type"], h["unit"] or "无", str(h["score"]),
                    h["text"][:90].replace("\n", " ")] for h in x["hits"]],
                  [400, 2400, 620, 620, 760, 4800]))

        B.append(("p", f"**模型回答**（召回 {ok[x['recall_hit']]} / "
                       f"回答 {ok[x['answer_hit']]}）："))
        for line in x["answer"].split("\n"):
            if line.strip():
                B.append(("bq", line))
        if VERDICTS.get(x["id"]):
            B.append(("p", f"**错在哪**：{VERDICTS[x['id']]}"))

    # ---------------------------------------------------------- 结论
    concl = C.ROOT / "结论.md"
    if concl.exists():
        B.append(("h", 2, "五、总体结论"))
        B.append(("meta", "（与仓库里的 结论.md 同源，改一处即可。）"))
        B += md_blocks(concl.read_text(encoding="utf-8"), shift=1)
    return B


def build_conclusion() -> list[tuple]:
    """结论.docx：把 结论.md 单独装订成一份 Word，供作业提交。

    它和主文档末尾的「五、总体结论」是同一段文字的两种装订 —— 主文档里
    shift=1 挂在大标题下，这里 shift=0 让它自己当一级标题。

    2026-10-13 补的：这份文件以前不在任何脚本里，头上却印着「与仓库里的
    结论.md 同源，改一处即可」，于是它漂成了 10-04 的旧版 —— 成绩还停在
    「答对 8/10」「类型二 · 检索失败」。**声称同源就得真有个脚本去同步**，
    否则那句声明本身就是在骗人。
    """
    src = C.ROOT / "结论.md"
    if not src.exists():
        return []
    return ([("h", 1, "总体结论"),
             ("meta", "（由 make_doc.py 从 结论.md 生成，改一处即可 —— 别手改这份。）")]
            + md_blocks(src.read_text(encoding="utf-8")))


# ---------------------------------------------------------------- docx

CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>"""

RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""

DOC_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"""


def _heading(sid: str, name: str, lvl: int, size: int, color: str | None) -> str:
    """一个 Heading 样式。

    Word 是靠 <w:name w:val="heading N"/> 认出内置标题的（不是靠 styleId），
    所以 name 必须写这个固定的英文名，写「标题 1」导航窗格就认不出来。
    outlineLvl 决定它在导航窗格里的层级。qFormat 让样式出现在样式库里。
    """
    rpr = (f'<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"'
           f' w:eastAsia="{HEI}"/><w:b/><w:bCs/>'
           f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>')
    if color:
        rpr += f'<w:color w:val="{color}"/>'
    return (
        f'<w:style w:type="paragraph" w:styleId="{sid}">'
        f'<w:name w:val="{name}"/><w:basedOn w:val="Normal"/>'
        f'<w:next w:val="Normal"/><w:qFormat/>'
        f'<w:pPr><w:keepNext/><w:outlineLvl w:val="{lvl}"/>'
        f'<w:spacing w:before="{240 - lvl * 40}" w:after="80"'
        f' w:line="300" w:lineRule="auto"/></w:pPr>'
        f'<w:rPr>{rpr}</w:rPr></w:style>')


STYLES = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
    '<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/'
    '2006/main"><w:docDefaults><w:rPrDefault><w:rPr>'
    f'<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"'
    f' w:eastAsia="{SONG}"/><w:sz w:val="21"/><w:szCs w:val="21"/>'
    '</w:rPr></w:rPrDefault><w:pPrDefault><w:pPr>'
    '<w:spacing w:after="90" w:line="300" w:lineRule="auto"/>'
    '</w:pPr></w:pPrDefault></w:docDefaults>'
    '<w:style w:type="paragraph" w:default="1" w:styleId="Normal">'
    '<w:name w:val="Normal"/><w:qFormat/></w:style>'
    + _heading("Heading1", "heading 1", 0, 32, "8B1D24")
    + _heading("Heading2", "heading 2", 1, 26, "8B1D24")
    + _heading("Heading3", "heading 3", 2, 23, "8B1D24")
    + _heading("Heading4", "heading 4", 3, 21, None)
    + "</w:styles>")

SECT = (f'<w:sectPr><w:pgSz w:w="{PAGE_W}" w:h="{PAGE_H}"/>'
        f'<w:pgMar w:top="{MARGIN}" w:right="{MARGIN}" w:bottom="{MARGIN}" '
        f'w:left="{MARGIN}" w:header="720" w:footer="720" w:gutter="0"/>'
        f'</w:sectPr>')


def write_docx(body_xml: str, path: Path) -> None:
    doc = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
           '<w:document xmlns:w="http://schemas.openxmlformats.org/'
           'wordprocessingml/2006/main"><w:body>'
           + body_xml + SECT + "</w:body></w:document>")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", CONTENT_TYPES)
        z.writestr("_rels/.rels", RELS)
        z.writestr("word/_rels/document.xml.rels", DOC_RELS)
        z.writestr("word/styles.xml", STYLES)
        z.writestr("word/document.xml", doc)


def main() -> int:
    raw = C.DATA / "eval_raw.json"
    if not raw.exists():
        print("没找到 data/eval_raw.json，先跑 eval.py")
        return 1
    rows = json.loads(raw.read_text(encoding="utf-8"))
    # 题面描述跟着题库刷新（和 eval.py --report-only 一个口径）：改句措辞
    # 不该逼着重跑十次大模型。**测量结果**仍以 eval_raw.json 为准。
    by_id = {q["id"]: q for q in QUESTIONS}
    for x in rows:
        q = by_id.get(x["id"])
        if q:
            x["q"], x["type"] = q["q"], q["type"]
            x["gold_desc"], x["note"] = q["gold_desc"], q.get("note", "")
            x["coverage_scope"] = q.get("coverage_scope", "")
    write_docx(render(build_blocks(rows)), OUT)
    print(f"✅ 已生成 {OUT.name}"
          f"（{OUT.stat().st_size/1024:.0f} KB，{len(rows)} 题）")

    blocks = build_conclusion()
    if blocks:
        write_docx(render(blocks), CONCL)
        print(f"✅ 已生成 {CONCL.name}"
              f"（{CONCL.stat().st_size/1024:.0f} KB，从 结论.md 重出）")
    else:
        print("⚠️  没找到 结论.md，跳过 结论.docx")
    return 0


if __name__ == "__main__":
    sys.exit(main())
