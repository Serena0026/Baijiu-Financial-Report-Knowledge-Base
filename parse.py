"""
第二步：把 PDF 拆成可检索的块。

这一步是整个项目最容易出错的地方。半年报里约 65% 的字符在表格中，
而表格里最有价值的恰恰是数字 —— 数字错一位、单位错一个量级，
后面大模型答得越自信错得越离谱。下面每条规则都是拿真 PDF 试出来的。

规则一：单位行在表格 bbox 外面
    实测茅台 p11 那张子公司表：bbox.y0 = 265.4，而「单位：万元币种：人民币」
    落在 bbox 上方那条带里。PyMuPDF 不会把它算进表格 —— 不主动去捞，
    这表就成了没有单位的裸数字。元 / 万元差 10000 倍，这是最贵的错。
    所以 find_unit() 要往 bbox 上方 30pt 的带子里找。

规则二：章节靠 PDF 书签，不靠正文正则
    正文里「第一节」和标题经常不在同一行（茅台是分行的，五粮液是同行带空格），
    统一正则必然在部分公司上 0 命中。而 PDF 内嵌书签（get_toc）直接给了页码。
    实测茅台 482 条书签，第一节~第八节齐全。

规则三：页眉页脚靠词频识别
    页眉「贵州茅台酒股份有限公司2026 年半年度报告」在 110/110 页重复。
    硬编码字符串换一家公司就失效，所以统计出现频率，超过阈值的行直接丢。

规则四：表格按行组分块，每块重复表头
    一张 40 行的表整块送去 embedding，向量会被平均成一团糨糊。
    切成每 15 行一块，每块都把表头和单位再写一遍，检索才准。

已知小瑕疵：PyMuPDF 的 extract() 偶尔把末列文字截断
    （实测表头 '本报告期比上年同期增减(%)' 被切成 '...增减('）。
    用 get_text('words') 能看到完整文字，但按单元格取就会切。
    只影响末列尾部字符，语义没丢，暂不处理。

跑法：
    cd ~/projects/财报知识库
    .venv/bin/python parse.py
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pymupdf

import config as C

# ---------------------------------------------------------------- 正则

# 「单位：万元币种：人民币」也要能捞出来，所以匹配到 元/万元 就停
UNIT_RE = re.compile(r"单位\s*[：:]\s*(亿元|万元|千元|元)")
SECTION_RE = re.compile(r"^第[一二三四五六七八九十]+节")
TOC_SECTION_RE = re.compile(r"^(第[一二三四五六七八九十]+节)\s*(\S.*)$")

# 页码：'~ 6 ~'、'- 12 -'、'第 6 页'、孤零零一个数字。
# 它每页都变，靠词频统计抓不出来，只能用模式认。
PAGENO_RE = re.compile(r"^\s*[~\-—－]?\s*(?:第\s*)?\d{1,4}\s*(?:页)?\s*[~\-—－]?\s*$")


def norm_section(title: str) -> str:
    """'第一节释义' → '第一节 释义'（各家有无空格不一，统一掉）"""
    m = TOC_SECTION_RE.match(title.strip())
    return f"{m.group(1)} {m.group(2)}" if m else title.strip()


# ---------------------------------------------------------------- 规则二：章节映射

# 印刷版目录：「第三节 管理层讨论与分析 ......... 8」
TOC_LEADER_RE = re.compile(
    r"^\s*(第[一二三四五六七八九十]+节\s*\S.*?)[.\s…·]{3,}\s*(\d{1,4})\s*$"
)
# 正文里的章节标题行：行首「第X节」，后面跟短标题
BODY_HEAD_RE = re.compile(r"^\s*(第[一二三四五六七八九十]+节)\s*(\S.{0,28})$")
# 正文里提到章节但只是引用的，不算标题
HEAD_STOPWORDS = ("本报告", "详见", "参见", "中“", "中\"", "所述")


def _sections_from_bookmarks(doc) -> list[tuple[int, str]]:
    out = []
    for lvl, title, pg in doc.get_toc():
        t = title.strip()
        if lvl == 1 and SECTION_RE.match(t):
            out.append((pg, norm_section(t)))
    return out


def _sections_from_printed_toc(doc, max_pages: int = 8) -> list[tuple[int, str]]:
    """印刷版目录页带点线引导和页码。实测五粮液 p3 是这种。

    它的页码就是 PDF 页序（已验证：目录说第八节=33，实际 p33 正是第八节）。
    """
    out = []
    for pno in range(min(doc.page_count, max_pages)):
        for ln in doc[pno].get_text().splitlines():
            m = TOC_LEADER_RE.match(ln)
            if m:
                out.append((int(m.group(2)), norm_section(m.group(1).strip())))
    return out


def _sections_from_body(doc) -> list[tuple[int, str]]:
    """最后兜底：正文里找「第X节 标题」行，取每节的首次出现。

    古井贡酒、洋河 2025 年报既没书签、目录页也没页码，但正文里章节标题
    是独立成行的 —— 这时正文扫描反而最可靠。
    """
    out: dict[str, int] = {}
    for pno in range(doc.page_count):
        lines = doc[pno].get_text().splitlines()
        # 目录页会一口气列出所有章节名，每行都长得像标题。
        # 不排掉它，整份文档的章节就全"开始"在目录页了（古井贡酒 p3 实测如此）。
        if sum(1 for l in lines if SECTION_RE.match(l.strip())) >= 3:
            continue
        for ln in lines:
            s = ln.strip()
            if any(w in s for w in HEAD_STOPWORDS):
                continue
            m = BODY_HEAD_RE.match(s)
            if m:
                name = norm_section(m.group(1) + m.group(2).strip())
                out.setdefault(name, pno + 1)
    return sorted((pg, name) for name, pg in out.items())


def build_section_map(doc) -> list[tuple[int, int, str]]:
    """建 (起页, 止页, 章节名) 列表，页码 1-based。

    三级回退，因为 32 份报告里有 5 份没有 PDF 书签：
      1. PDF 内嵌书签（27/32，最准）
      2. 印刷版目录页的点线引导（五粮液）
      3. 正文里的章节标题行（古井贡酒、洋河 2025 年报）
    """
    entries = (_sections_from_bookmarks(doc)
               or _sections_from_printed_toc(doc)
               or _sections_from_body(doc))
    entries = sorted(set(entries))
    if not entries:
        return []

    ranges = []
    for i, (pg, name) in enumerate(entries):
        end = entries[i + 1][0] - 1 if i + 1 < len(entries) else doc.page_count
        # 同一页起两个章节时（茅台第一节、第二节都在 p4），止页会小于起页。
        # 不修，让后一个章节覆盖该页 —— 释义页信息量低，且后一节才是正文。
        ranges.append((pg, max(end, pg), name))
    return ranges


def section_of(page_no: int, ranges: list[tuple[int, int, str]]) -> str:
    """page_no 是 1-based。取最后一个起点 <= 本页的章节。"""
    hit = ""
    for start, _end, name in ranges:
        if start <= page_no:
            hit = name
        else:
            break
    return hit


# ---------------------------------------------------------------- 规则三：页眉页脚

def find_running_lines(doc, sample: int = 40, ratio: float = 0.3) -> set[str]:
    """跨页高频重复的短行 = 页眉页脚。"""
    counter: Counter[str] = Counter()
    pages = min(doc.page_count, sample)
    for i in range(pages):
        lines = [ln.strip() for ln in doc[i].get_text().splitlines() if ln.strip()]
        # 只看每页首尾各 3 行，页眉页脚只可能在这里
        for ln in set(lines[:3] + lines[-3:]):
            if len(ln) <= 60:          # 太长的多半是正文，不参与统计
                counter[ln] += 1
    threshold = pages * ratio
    return {ln for ln, n in counter.items() if n >= threshold}


# ---------------------------------------------------------------- 规则一：单位

def find_unit(page, tbbox, upper_bound: float = 0.0) -> str:
    """单位行在 bbox 外，往上找。

    搜索带往上取 60pt（不是 30）：实测古井贡酒 p10 的单位行离 bbox 有 40 多 pt。
    但不能越过 upper_bound —— 那是同页上一张表的底边，否则会把别人家的单位
    认领过来（比如把产量表的「单位：吨」安到财务表头上）。
    """
    top = max(upper_bound, tbbox[1] - 60)
    band = pymupdf.Rect(0, top, page.rect.width, tbbox[1] + 2)
    m = UNIT_RE.search(page.get_text("text", clip=band))
    if m:
        return m.group(1)

    # 有些表把单位塞在表内（首行或末行）
    wide = pymupdf.Rect(0, tbbox[1], page.rect.width,
                        min(page.rect.height, tbbox[3] + 24))
    m = UNIT_RE.search(page.get_text("text", clip=wide))
    return m.group(1) if m else ""


# 续页判定：表格紧贴正文区顶部，说明它是上一页那张表的延续
CONT_TOP_Y = 110.0


# ---------------------------------------------------------------- 表格处理

def clean_cell(s: str | None) -> str:
    return re.sub(r"\s*\n\s*", " ", (s or "").strip()).strip()


def _numericish(v: str) -> bool:
    """带数字且不含汉字 —— 用来判断某列是不是数值列。"""
    return bool(re.search(r"\d", v)) and not re.search(r"[一-鿿]", v)


def table_parts(table) -> tuple[str, list[str], list[list[str]]]:
    """拆出 (上方说明文字, 列名, 数据行)。

    关键判据是**列数**，不是 header.external：

    external 只表示"检测到的表头在 bbox 之外"，它不代表表内第一行不是表头。
      · 4 列的「主要会计数据」 external=False，extract() 第 0 行是表头
      · 4 列的「主要财务指标」 external=True（上方有"(二) 主要财务指标"），
        但 extract() 第 0 行仍然是真表头
      · 2 列的「公司注册地址」 external=True，第 0 行是数据 —— 键值表没有表头

    若不加区分地把第 0 行当表头，键值表会被前向填充灌成一团：
        '公司注册地址: 公司办公地址 | 贵州省仁怀市茅台镇: 贵州省仁怀市茅台镇'
    """
    rows = [[clean_cell(c) for c in r] for r in table.extract()]

    caption = ""
    try:
        if table.header is not None and table.header.external:
            caption = next((n for n in (clean_cell(x) for x in table.header.names) if n), "")
    except Exception:
        pass

    if table.col_count <= 2:
        # 键值表：没有表头，逐行按 "键：值" 渲染
        return caption, [], [r for r in rows if any(r)]

    header = list(rows[0])
    for i, h in enumerate(header):          # 合并单元格留下的空位前向填充
        if not h:
            for j in range(i - 1, -1, -1):
                if header[j]:
                    header[i] = header[j]
                    break
    return caption, header, [r for r in rows[1:] if any(r)]


DUP_WIN = 200       # 重复检测的窗口大小
DUP_STRIDE = 40


def _dup_ratio(text: str) -> float:
    """块内重复窗口的占比。

    把文本切成 DUP_WIN 字符的窗口（步长 DUP_STRIDE），数有多少窗口在
    文本里还出现过第二次。"同一段话被抄了很多遍"会接近 1；
    只是重复了一个短列名的表接近 0 —— 这正是要分开的两类。
    """
    wins = [text[i:i + DUP_WIN]
            for i in range(0, max(len(text) - DUP_WIN + 1, 1), DUP_STRIDE)]
    wins = [w for w in wins if len(w) == DUP_WIN]
    if not wins:
        return 0.0
    return sum(1 for w in wins if text.count(w) > 1) / len(wins)


def _row_line(r: list[str], header: list[str], label_col: bool) -> str:
    """一个数据行 → 渲染出来的一行文本。"""
    if header:
        if label_col:
            rest = [f"{h}: {v}" for h, v in zip(header[1:], r[1:]) if v and h]
            joined = " | ".join(rest)
            return f"{r[0]} | {joined}" if r[0] and rest else (r[0] or joined)
        pairs = [f"{h}: {v}" for h, v in zip(header, r) if v and h]
        return " | ".join(pairs)
    cells = [c for c in r if c]
    if len(cells) >= 2:
        return f"{cells[0]}：{cells[1]}"
    return cells[0] if cells else ""


TRUNC_NOTE = " …（本表过长，已截断）"


def _cap(text: str) -> str:
    """表格块的唯一长度闸门。

    正文那条路有 split_text 硬切兜底，表格这条路按 TABLE_GROUP_ROWS 分行
    拼起来，拼多长完全取决于 PDF 长什么样 —— 原来没有上限。

    实测金徽酒 2025年报 p161：表格检测把整页文字并成了一张"表"，整页文字
    成了其中一行，16 行里 15 行是同一页的副本，一个块拼到 52,823 字符，
    超过百炼 33,000 token 的 input 上限。index_build 是按长度**升序**算的，
    这块排在最后 —— 前面 1.4 万个块算完才炸，十几分钟全白费。

    这里**截断而不是拆块**：这种误判块内部本来就是同一页内容的重复，
    拆成 N 块只会把 1 个垃圾块变成 N 个垃圾块，检索时反而更容易被误召回
    （重复内容会把 BM25 的词频顶上去）。真正有用的内容一份就够。
    """
    if len(text) <= C.MAX_TABLE_CHARS:
        return text
    return text[:C.MAX_TABLE_CHARS] + TRUNC_NOTE


def table_to_chunks(table, page, meta: dict, unit: str, next_id) -> list[dict]:
    """一张表 → 若干块（每块 TABLE_GROUP_ROWS 行，重复表头）。

    返回 [] 有两种含义，parse_pdf 一视同仁：**这张表的所有区域都交回正文路径**。
    所以调用方必须把没有产出块的表的 bbox 排除在 tboxes 之外，否则它的内容
    既没进表格块、又被正文剔掉，整块消失。
    """
    if table.row_count < C.MIN_TABLE_ROWS or table.col_count < C.MIN_TABLE_COLS:
        return []

    caption, header, data = table_parts(table)
    if not data:
        return []
    if header and len([h for h in header if h]) < C.MIN_TABLE_COLS:
        return []

    # 中文财报常把表格标题塞在第一列的表头格里（「主要会计数据」那一格），
    # 于是会渲染成 "主要会计数据: 营业收入" —— 读起来莫名其妙。
    # 判据：第一列的值全是文字、一个数字都没有 → 它是行标签列，表头不该带出来。
    label_col = False
    if header:
        vals = [r[0] for r in data if r and r[0]]
        if vals and not any(_numericish(v) for v in vals):
            label_col = True

    row_lines = [_row_line(r, header, label_col) for r in data]

    # 误检闸门：整张表一起判，不能一块一块判 —— 误检块之间本来就是同一段话
    # 的副本，拆开看每一块自己反而"不重复"（拆块只会把 1 个垃圾块变成 N 个）。
    if _dup_ratio("\n".join(l for l in row_lines if l)) > C.TABLE_DUP_RATIO:
        return []

    out = []
    grp = C.TABLE_GROUP_ROWS
    for start in range(0, len(row_lines), grp):
        lines = [l for l in row_lines[start:start + grp] if l]
        if not lines:
            continue

        body = "\n".join(lines)
        # 标题优先用表格上方的说明文字（"一、公司信息"），它比列名更能自解释
        head = caption or (header[0] if header and header[0] else "")
        prefix = f"{head} " if head and head != body[:len(head)] else ""
        unit_note = f"（单位：{unit}）" if unit else ""
        # 公司/报告/章节/页码都写进正文，让向量带上"这是谁家哪一节"的语境
        text = (f"{meta['company']} {meta['report']} {meta['section']} "
                f"第{meta['page']}页 {prefix}{unit_note}\n{body}")
        out.append({
            **meta,
            "id": next_id(),
            "type": "table",
            "unit": unit,
            "text": _cap(text),
        })
    return out


# ---------------------------------------------------------------- 正文切块

def split_text(text: str, size: int, overlap: int) -> list[str]:
    """按段落聚合到接近 size 字符；超长段落硬切。"""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, buf = [], ""
    for p in paras:
        if len(p) > size:                        # 单段就超长，先冲掉缓冲再硬切
            if buf:
                chunks.append(buf)
                buf = ""
            for i in range(0, len(p), size - overlap):
                chunks.append(p[i:i + size])
            continue
        if len(buf) + len(p) + 1 > size and buf:
            chunks.append(buf)
            buf = buf[-overlap:] + "\n" + p if overlap else p
        else:
            buf = f"{buf}\n{p}" if buf else p
    if buf:
        chunks.append(buf)
    return [c for c in chunks if len(c.strip()) >= 30]   # 太短的没信息量


def page_text_without_tables(page, tboxes, running: set[str]) -> str:
    """取正文，剔除落在表格框内的块和页眉页脚。"""
    out = []
    for x0, y0, x1, y1, txt, *_ in page.get_text("blocks"):
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        inside = any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in tboxes)
        if inside:
            continue
        for ln in txt.splitlines():
            s = ln.strip()
            if s and s not in running and not PAGENO_RE.match(s):
                out.append(s)
    return "\n".join(out)


# ---------------------------------------------------------------- 主流程

def parse_pdf(path: Path, meta_base: dict, next_id) -> list[dict]:
    doc = pymupdf.open(path)
    running = find_running_lines(doc)
    sections = build_section_map(doc)
    if not sections:
        print(f"    ⚠️  {meta_base['company']} 没有章节书签，section 会是空的")

    chunks: list[dict] = []
    prev_unit = ""      # 上一张表识别到的单位，供续页继承
    for pno in range(doc.page_count):
        page = doc[pno]
        page_no = pno + 1
        meta = {**meta_base, "page": page_no,
                "section": section_of(page_no, sections)}

        try:
            detected = list(page.find_tables().tables)
        except Exception:
            detected = []

        # tboxes 只收**真的产出了块**的表，这个顺序不能反：
        # tboxes 的作用是把表格区域从正文里剔掉，若把一张被拒绝的表留在
        # 里面，它的内容既没进表格块、又被正文剔掉，整页内容凭空消失。
        # （以前就是这样：行列数不够的表、以及误检表，内容全丢了。）
        emitted: list = []
        upper = 0.0                       # 同页上一张表的底边，防止越界认领
        for t in detected:
            unit = find_unit(page, t.bbox, upper)
            if not unit and t.bbox[1] <= CONT_TOP_Y and prev_unit:
                # 贴顶的表 = 上一页的续页，单位行留在了上一页
                unit = prev_unit
            if unit:
                prev_unit = unit
            upper = t.bbox[3]
            got = table_to_chunks(t, page, meta, unit, next_id)
            if got:
                emitted.append(t)
                chunks.extend(got)
        tboxes = [t.bbox for t in emitted]

        body = page_text_without_tables(page, tboxes, running)
        for piece in split_text(body, C.CHUNK_SIZE, C.CHUNK_OVERLAP):
            chunks.append({
                **meta,
                "id": next_id(),
                "type": "text",
                "unit": "",
                "text": (f"{meta['company']} {meta['report']} {meta['section']} "
                         f"第{page_no}页\n{piece}"),
            })

    doc.close()
    return chunks


def main() -> int:
    manifest = json.loads(C.MANIFEST.read_text(encoding="utf-8")) if C.MANIFEST.exists() else []
    if not manifest:
        print("没找到 manifest.json，先跑 fetch.py")
        return 1

    counter = [0]

    def next_id() -> str:
        counter[0] += 1
        return f"c{counter[0]:06d}"

    all_chunks: list[dict] = []
    for i, m in enumerate(manifest, 1):
        path = C.PDF_DIR / m["file"]
        if not path.exists():
            print(f"  [{i}/{len(manifest)}] 缺文件 {m['file']}")
            continue
        got = parse_pdf(path, {
            "code": m["code"], "company": m["company"], "report": m["report"],
        }, next_id)
        n_tab = sum(1 for c in got if c["type"] == "table")
        print(f"  [{i}/{len(manifest)}] {m['company']:6s} {m['report']:9s} "
              f"→ {len(got):>5} 块（表格 {n_tab}）")
        all_chunks.extend(got)

    with C.CHUNKS.open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    n_tab = sum(1 for c in all_chunks if c["type"] == "table")
    no_unit = sum(1 for c in all_chunks if c["type"] == "table" and not c["unit"])
    print()
    print("=" * 60)
    print(f"总块数 {len(all_chunks):,}（正文 {len(all_chunks)-n_tab:,} / 表格 {n_tab:,}）")
    print(f"表格块中缺单位的: {no_unit:,}  ← 这些块的数字量级不可信")
    print(f"已写入 {C.CHUNKS}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
