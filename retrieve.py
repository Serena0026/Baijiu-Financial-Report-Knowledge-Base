"""
第四步：混合检索 —— 向量 + BM25 → RRF 融合 → 同公司配额。

为什么不能只用向量：

1. 向量对**精确串**很迟钝
   问「600779 的营收」，向量会把"意思相近"的其他白酒公司排在前面 ——
   因为它们的句子长得几乎一样。BM25 能一字不差地命中 600779。

2. 向量对**数字**不可靠
   「哪家净利润最高」这种，向量召回的是"谈论净利润"的段落，
   而不是"净利润数值最大"的那段。

反过来 BM25 也不够：问「直接销售取得的收入有多少」，
报告原文写的是「直销渠道收入」，字面不重合，BM25 一个都召不回。
两条腿必须都有。

融合用 RRF（Reciprocal Rank Fusion）：
    score(d) = Σ 1/(K + rank_i(d))
它只看**排名**不看分数，所以不用担心"余弦相似度 0.7 和 BM25 得分 12.3 怎么比"
—— 这两个量纲根本不可比，硬加权才是错的。K=60 是原论文的推荐值。

同公司配额：一份 116 页的年报能产出 500 多个块，16 家竞争 top-6 时，
很容易 6 个块全来自同一家公司 —— 那样"对比各家"类的问题就废了。
所以每家最多贡献 MAX_PER_COMPANY 块。

查询编码器必须和**建索引时**用的那个一致，这是本文件唯一会静默出错的地方：
本地 Qwen3 和百炼 text-embedding-v4 是两个不同的向量空间，拿 A 的查询向量
去点乘 B 的索引向量，内积毫无意义 —— 而且不抛异常，只是检索结果错得看不出来。
所以后端名从 index_meta.json 读，不靠环境变量猜。

用法：
    r = Retriever()
    hits = r.search("茅台上半年直销收入是多少")
    for h in hits: print(h["score"], h["chunk"]["text"][:60])
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import bm25
import config as C
import embed as embed_mod

# 全景题信号：这些问题问的是"跨公司"，不是"某一家"
PANORAMA_WORDS = (
    "对比", "比较", "各家", "各公司", "哪家", "哪几家", "谁家", "排名", "排行",
    "最高", "最低", "最大", "最小", "谁最", "前几", "分别是多少",
    "全景", "一览", "横向",
)

# ---------------------------------------------------------------- 全景锚点

_SECTION_RE = re.compile(r"第[一二三四五六七八九十]+节")


def _looks_like_toc(text: str) -> bool:
    """目录页识别：一串省略号引导符，或密集的「第X节」。

    锚点候选是靠关键词捞出来的，而**目录里必然列出"主要会计数据"
    这个标题**，所以不排掉的话，每家公司都会把锚点选中自己的目录页。
    """
    return text.count("....") >= 3 or len(_SECTION_RE.findall(text)) >= 4


def _report_rank(report: str) -> tuple[int, int]:
    """「2026半年报」→ (2026, 1)，「2025年报」→ (2025, 0)。

    全景题是横向比**同一期**。语料里每家公司都同时有 2026半年报 和
    2025年报，两者混着当锚点会得出错的对比结论，所以先按报告期筛，
    只在最新一期里挑。
    """
    m = re.search(r"(\d{4})", report)
    return (int(m.group(1)) if m else 0, 1 if "半年" in report else 0)


def _anchor_score(c: dict) -> float:
    """给「主要会计数据」候选块打分，数字越密越像那张表。

    不能只认"归属于上市公司股东的净利润"这句标签——表格识别会把行标签
    丢掉，只剩数字（五粮液 p7、天佑德酒 p8 就是），而那种块恰恰是最该
    当锚点的。数字密度对这些情况都稳：目录页约 0.03，数据表 0.3~0.4。
    """
    if _looks_like_toc(c["text"]):
        return -1.0
    score = sum(ch.isdigit() for ch in c["text"]) / max(len(c["text"]), 1)
    if "归属于上市公司股东的净利润" in c["text"]:
        score += 0.5        # 净利润行是全景题最常问的，见到就加权
    if c["type"] == "table":
        score += 0.05
    return score


# ---------------------------------------------------------------- 全景取证
#
# 旧做法是「每家公司取一个锚点块，取到就算数」。它有一个静默失败：
# 锚点表坏掉的那家公司（五粮液 p7、天佑德酒 p8）**照样会被放进上下文**，
# 只是里面没有被问的那个数 —— 模型于是写「该片段未提供」，
# 而没有任何东西告诉它「这家本该有数据」。16 家坏 1 家，那家就整个消失。
#
# 这里改成「按公司取证」：候选放宽到该公司的全部块，逐家校验**被问的字段
# 到底在不在**，状态记进覆盖地图（covered / partial / missing），
# 缺据的公司由提示词强制点名，不许静默略过。
#
# 注意候选池只用于**择优与判定覆盖**，进上下文的仍是一家一块 —— 广度不涨成本。

# 问句里的字段名 → 该字段在表格里的行标签候选。
# 「营业收入」和「营业总收入」是两个不同的行项目（茅台 2026H1 就是前者 +1.47%、
# 后者 922.78 亿），但都可能是题目要的那个数，所以成组收下，
# 并且把**实际命中的是哪一个**记进覆盖地图 —— 这就是"核对出处"。
FIELD_SPECS: dict[str, tuple[str, ...]] = {
    "营业收入": ("营业收入", "营业总收入"),
    # 「归属于母公司所有者的净利润」是合并利润表里的标准行名，与
    # 「归属于上市公司股东的净利润」同义（前者见天佑德酒 p104 的未分配利润附注，
    # 后者见各家第二节主表）。不把它单列出来，就只剩短标签「净利润」可用，
    # 短标签既会命中央企的「五、净利润」（含少数股东损益，不是归母数），
    # 也会命中叙述段里凑巧出现的字样。
    # 倒数第二条是被 PDF 截断的写法：那一格的最后一个「润」字被切到了下一行，
    # 提取出来就是「…所有者的净利」（天佑德酒 p104）。留着它，那块才认得出是归母行。
    "净利润": ("归属于上市公司股东的净利润", "归属于母公司所有者的净利润",
             "归属于母公司所有者的净利", "净利润"),
    "现金流": ("经营活动产生的现金流量净额",),
}

# 问题里出现这些词 → 要核该字段。顺序即优先级（先匹配到的先用）。
_FIELD_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("现金流", ("现金流", "现金流量")),
    ("净利润", ("净利润", "归母")),
    ("营业收入", ("营业收入", "营收")),
)

# 问「同比」「正增长」这类，光有字段值不够，还得有同比那一列。
YOY_LABELS = ("比上年同期增减", "同比增减", "同期增减", "同比")
_YOY_HINTS = ("同比", "增减", "正增长", "负增长", "增速")

# 同比**那一格**：标签 + 可选「(%)」+ 冒号 + 数值 + 可选「%」。
# 各家写法不齐，实测三种：
#   本报告期比上年同期增减(%): 3.19      （金徽酒 p7，带 % 号的是表头，数值不带）
#   本报告期比 上年同期增 减(%): 1.47    （茅台 p5，标签被换行切成三段 —— 先 _norm）
#   同比增减: 20.87%                     （五粮液 p12 收入构成表，数值自带 % 号）
# 标签按 YOY_LABELS 的长短顺序排列，长的在前，避免「同比」把「同比增减」截断。
_YOY_CELL_RE = re.compile(
    "(?:" + "|".join(YOY_LABELS) + r")[（(]?[%％]?[)）]?[:：]?\s*"
    r"([-−+]?\d[\d,]*(?:\.\d+)?)\s*[%％]?")

PERIOD_LABELS = ("本报告期", "上年同期", "本期", "本年", "2026 年半年度", "2026年半年度")

# 行标签后面紧跟这些词，说明它在**列名**里而不是行首（见 _label_value）。
_COLNAME_TAIL = ("比上", "比上年", "同比", "增减", "比重", "构成", "占比")

# 反过来，标签**前面**紧挨这些词，说明它是给某个格子当列名的，不是这一行的行标签：
# 天佑德酒 p14「线上销售平台: 电商平台 | 本期营业收入: 24,310,186.36」里的
# 「营业收入」其实是列名，24,310,186.36 是电商平台那个格子的值。
_CELL_LABEL_HEAD = ("本期", "上期", "本年", "上年")

# 「分行业/分产品」拆解表的特征，与整体表的特征（_TOTAL_HINTS）。
# 拆解表里的「营业收入」是分部数，不能当公司营收用（见 _is_segment_only）。
# 整体行只认 100% 那一种：拆解表里的「合计」是**列名**（「合计: 营业收入」），
# 不是整体行，拿它当通行证等于没拦。
_SEGMENT_HINTS = ("分行业", "分产品", "分地区", "分部", "主营业务分", "毛利率")
_TOTAL_HINTS = ("100%", "100.00%")
# 「主要子公司、参股公司」情况表、以及「重要联营/合营企业财务信息」摘要表的标志，
# 见 _is_entity_table
_ENTITY_TABLE_HINTS = ("公司名称", "公司类型", "主要子公司", "参股公司", "注册资本",
                       "联营企业", "合营企业")

# 章节偏好：第二节那张「主要会计数据」表是公司自己披露的合并口径主表，
# 第三节的收入构成表通常也带同比。都比散落在第八节的明细表更该当锚点。
_SECTION_BONUS = {
    "第二节 公司简介和主要财务指标": 0.30,
    "第三节 管理层讨论与分析": 0.20,
}

# 取证的 BM25 候选宽度。要够宽，宽到每家公司至少有一条落进来 ——
# Q9 那道题实测前 400 个命中才让 16 家全部出现，再窄就有公司整个漏在池子外，
# 只能靠锚点兜底（而锚点恰恰是坏的那批公司的问题所在）。
PANO_BM25_K = 400

# 口径信号只能做到"尽力而为"，见文件末尾 pano 自检的说明：
# 表格块里可靠带报表标题的几乎为零，所以这里的正则是**精确匹配整张报表名**，
# 故意不去匹配光秃秃的「母公司」——那三个字在 387 个表格块里出现，
# 绝大多数是利润表的行项目「归属于母公司所有者的净利润」，纯属假阳性。
_CONSOL_STMT_RE = re.compile(r"合并(现金流量表|资产负债表|利润表)")
_PARENT_STMT_RE = re.compile(r"母公司(现金流量表|资产负债表|利润表)")

_NUM_RE = re.compile(r"[-−]?\d[\d,]*\.?\d*")
# 金额在这类表里一律带千分位逗号，所以能用它把「本报告期（1－6月）:1,951,744,051.30」
# 里的那个「1」和后面的百分数排除掉 —— 见 _label_value。
# 负号必须一起取：亏损公司的归母净利润在表里就是「本报告期（1－6月）:-6,222,478.74」
# （水井坊 p5），只取数字会把它变成正的 —— 覆盖地图上给模型一个符号错的值，
# 比不给还坏，Q8 问的正是"哪几家为负"。
_MONEY_RE = re.compile(r"[-−]?\d{1,3}(?:,\d{3})+(?:\.\d+)?")
_PCT_RE = re.compile(r"[-−]?\d+\.?\d*\s*%")
_WS_RE = re.compile(r"\s+")


@lru_cache(maxsize=8192)
def _norm(text: str) -> str:
    """去掉全部空白后的文本，所有"标签是否出现"的判断都走它。

    PDF 提取出来的表头单元格会在换行处插空格，一条标签经常被切成两三段：

        本报告期比 上年同期增 减(%): 1.47        （茅台 p5）
        本报告期比上年同 期增减(%): -12.18       （山西汾酒 p5）

    拿不带空格的标签直接 in 是匹配不上的 —— 2026-10-12 实测：茅台、山西汾酒、
    口子窖 三家的同比列全被这一个空格判成"没有"，茅台于是退到第三节那张
    营业总收入（+1.3%）的表，把精确口径的营业收入（+1.47%）丢了。
    """
    return _WS_RE.sub("", text)


def _has_label(text: str, labels: tuple[str, ...]) -> bool:
    n = _norm(text)
    return any(lab in n for lab in labels)


def _to_float(s: str) -> float:
    """把表格里的数字串转成 float，供量级比较用；转不动就返回 -1。"""
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return -1.0


def _is_cell_header(n: str, i: int, head: str) -> bool:
    """标签前面那截是不是**整整一个单元格的列名**（「本期/上期/本年/上年」+ 标签）。

    提取出来的表格就是「名字:值|名字:值|…」这个形状，所以单元格以「|」分界：
    回退到上一个「|」，看这一段是不是恰好等于「本期」+标签。
    天佑德酒 p14「线上销售平台:电商平台|本期营业收入:24,310,186.36」正是这种 ——
    24,310,186.36 是"电商平台这一列的本期营业收入"，不是公司的营业收入。

    只看"标签前面几个字里有没有本期"会把正经的行标签也误伤：未分配利润附注里的
    「加：本期归属于母公司所有者的净利润」（五粮液 p92、天佑德酒 p104）前面也挨着
    「本期」，一起被判掉之后取证只能退回短标签「净利润」，口径跟着丢了 ——
    那正是老师说的"数字进了图谱不知道是谁的哪个口径"。
    """
    j = n.rfind("|", 0, i) + 1
    return n[j:i] == head


def _label_value(n: str, label: str, window: int = 120) -> str | None:
    """在**已去空白**的文本里找「行标签 + 邻近数值」，返回那个数的**原串**；找不到返回 None。

    只查标签会把「只有表头、没有数值」的块也算成命中 —— 天佑德酒有一块就是以
    「营业收入比上: 年同期增减」开头的分行业表头，光看标签会误判。
    距离在去空白后的串上量，所以跨行断开的标签照样能配上值。

    两种「标签在那儿、但它不是这一行的行标签」的情况要跳过：
    - 后面紧跟「比上/同比/比重」→ 它在**列名**里（见 _COLNAME_TAIL）
    - 前面整整一格是「本期/上期/本年/上年」→ 它在给某个格子当列名（见 _is_cell_header）

    取值优先取带千分位的那个：光取"第一个数字"会取到期间标记里的数字
    （「本报告期（1－6月）: 1,951,744,051.30」的第一个数字是 1），择优做量级比较时
    就会把好块判低 —— 2026-10-12 实测：口子窖 Q7 就是这么把 p5 主表的 19.5 亿
    读成 1.0，输给 p6 说明表的 48.97 的。
    """
    start = 0
    while True:
        i = n.find(label, start)
        if i < 0:
            return None
        start = i + 1
        end = i + len(label)
        if n[end: end + 4].startswith(_COLNAME_TAIL):
            continue
        if any(_is_cell_header(n, i, h) for h in _CELL_LABEL_HEAD):
            continue
        m = _MONEY_RE.search(n[end: end + window]) or _NUM_RE.search(n[end: end + window])
        if m:
            return m.group()


def _has_yoy(text: str) -> bool:
    """块里有没有"同比/增减"那根表头。空格差一格的写法也要认（见 _norm）。

    这是**粗判**，只够当择优时的软加权（_score_candidate）。判覆盖、摆上覆盖地图
    一律用 _yoy_value —— 那一格到底属不属于被问的那一行，只有它说得准。
    """
    return _has_label(text, YOY_LABELS)


def _yoy_value(text: str, value: str) -> str:
    """取**命中那一行**的同比数，返回带符号的百分数串（"+20.87%"）；取不到返回 ""。

    为什么非要按行取：光看"块里有没有同比"会把下一行的同比算到这一行头上。
    块里任何一行有同比，整块的 _has_yoy 就是 True —— 那么一家增收但没披露同比
    的公司，会被它下面「利润总额」那一行的同比顶成 covered，数还是错的。
    「宁可 partial，不可假覆盖」（见 _is_entity_table）在这里同样适用。

    只在本行内找，不跨行去够：跨行够到的必然是**下一条记录**的同比，
    把 利润总额 的增速安到 营业收入 头上，比取不到更坏。取不到就返回 ""，
    让 _pick_evidence 判 partial —— 那正是"缺据明说"要如实呈现的东西。

    2026-10-12 实测（Q9 又把五粮液漏了）：取证层已经核过同比、也把该公司标成
    covered，但覆盖地图只摆了营业收入 28,416,674,541.77，**同比 20.87% 没摆出来**。
    而五粮液那一行的同比恰恰挂在一行丢了行标签的数据上（「本报告期: 28,416,674,541.77
    | … | 同比增减: 20.87%」），模型认不出它是谁的同比，于是这家全场增速最高的公司
    再一次整个消失。数据在上下文里，地图里没有 —— 把这一格取出来摆进地图，
    这道题就从"让模型自己回原文翻 16 张表"变成"照地图数一遍"。
    """
    if not value:
        return ""
    pos = text.find(value)
    if pos < 0:
        return ""
    tail = text[pos + len(value):]
    nl = tail.find("\n")
    if nl >= 0:
        tail = tail[:nl]              # 只看同一逻辑行，理由见 docstring
    m = _YOY_CELL_RE.search(_norm(tail))
    if not m:
        return ""
    num = m.group(1).replace("−", "-")
    sign = "" if num.startswith(("-", "+")) else "+"
    return f"{sign}{num}%"


def _is_segment_only(n: str) -> bool:
    """是不是「分行业/分产品」的拆解表，且没有整体合计行。

    拆解表里的「营业收入」是分部数：天佑德酒 p17 的「酒制造｜营业收入:
    530,016,558.28｜…｜营业收入比上: 0.78%」全公司其实是 574,885,860.71 / -14.68%，
    而且那个 +0.78% 是分部增速。字面上它完全合格（行标签 + 数值 + 同比三项齐全），
    标签校验拦不住，只能另外判一层 —— 否则覆盖地图会把它标成 covered，
    比原来「静默消失」更坏：模型会照着说天佑德酒正增长。

    注意「其中：」不能算拆解特征 —— 合并利润表里的「其中：营业收入」也是整体数
    （天佑德酒 p45），把它算进来会误伤真正要的那块。
    """
    if not any(h in n for h in _SEGMENT_HINTS):
        return False
    return not any(t in n for t in _TOTAL_HINTS)


def _is_entity_table(n: str) -> bool:
    """是不是「主要子公司/参股公司」情况表 —— 这类表里的数是**子公司自己**的数。

    五粮液 p15 有「宜宾五粮液酒类销售有限责任公司｜…｜净利润: 5,003,381,542.34」，
    字面上完全符合「行标签 + 数值」，不问青红皂白就会被当成五粮液的归母净利润
    填进覆盖地图。老师那句"先核母公司/合并口径"防的正是这个：子公司口径比母公司
    口径还远一层。整块不算命中，让择优去换下一块（宁可 partial，不可假覆盖）。

    用「公司名称/公司类型/注册资本」这几个列名当特征：正常的利润表、资产负债表、
    收入构成表里一个都不会有。
    """
    return any(h in n for h in _ENTITY_TABLE_HINTS)


# 每个字段里，前几个标签才算**真的核到了这个字段**，更靠后的兜底标签只够 partial。
# 「净利润」这一组尤其要紧：题目问的都是**归母**净利润，而光秃秃的「净利润」行在
# 合并利润表里是含少数股东损益的总额（天佑德酒 p46 的 10,523,828.83，归母只有
# 10,440,139.66），在母公司利润表里是母公司数（同公司 p47 的 13,480,018.50）。
# 拿它当覆盖，正是老师批评的那种"数字进了图谱就不知道是谁的哪个口径"。
_STRICT_N = {"营业收入": 1, "净利润": 3, "现金流": 1}


def _is_strict(hit: tuple[int, str, str] | None, fields: list[str]) -> bool:
    """命中的标签够不够"就是这个字段"，还是只是个兜底写法。"""
    if hit is None:
        return False
    for name in fields:
        labs = FIELD_SPECS[name]
        if hit[1] in labs:
            return labs.index(hit[1]) < _STRICT_N.get(name, 1)
    return False


def _field_hit(text: str, fields: list[str]) -> tuple[int, str, str] | None:
    """在块里找被问的字段，返回 (标签精度, 命中标签, 命中值原串)。精度即 FIELD_SPECS 的下标。

    「营业收入」排在「营业总收入」前面是有意的：两者是不同的行项目
    （茅台 2026H1 营收 +1.47%、营业总收入 922.78 亿），能用精确的那个就别用宽的。
    择优时精度优先于分数，否则一个数字更密的第三节表会把第二节的主表挤掉。

    拆解表、子公司情况表整块不算命中，理由见 _is_segment_only / _is_entity_table。
    """
    n = _norm(text)
    if _is_segment_only(n) or _is_entity_table(n):
        return None
    best: tuple[int, str, str] | None = None
    for name in fields:
        for k, lab in enumerate(FIELD_SPECS[name]):
            v = _label_value(n, lab)
            if v is not None and (best is None or k < best[0]):
                best = (k, lab, v)
    return best


def _query_fields(query: str) -> list[str]:
    """从问句里抽出要核的字段名，按 _FIELD_HINTS 的顺序。"""
    return [name for name, hints in _FIELD_HINTS if any(h in query for h in hints)]


def _wants_yoy(query: str) -> bool:
    return any(h in query for h in _YOY_HINTS)


def _yoy_num(text: str) -> float | None:
    """把覆盖地图里的 "+20.87%" 还原成 20.87；空串或转不动返回 None。

    只用来排序和数方向小计，不参与覆盖判定 —— 转不动就当"没有同比"，不影响 status。
    """
    if not text:
        return None
    try:
        return float(text.replace("−", "-").rstrip("%％"))
    except ValueError:
        return None


def _matched_period(text: str) -> str:
    for lab in PERIOD_LABELS:
        if lab in text:
            return lab
    return ""


def _scope_of(text: str) -> str:
    """口径：合并 / 母公司 / 未标注。第三种是常态，如实标出来。"""
    n = _norm(text)
    if _CONSOL_STMT_RE.search(n):
        return "合并"
    if _PARENT_STMT_RE.search(n):
        return "母公司"
    return "未标注"


@dataclass
class Evidence:
    """一家公司在一次全景取证里的结果。status 就是覆盖地图的那一格。"""

    code: str
    company: str
    report: str
    status: str                     # covered | partial | missing
    doc: int | None = None          # 选中的块下标；missing 时为 None
    source: str = ""                # 锚点 / 回退 —— 这条数是从哪条路来的
    field: str = ""                 # 实际命中的行项目名
    value: str = ""                 # 命中的数值，原样（如 "90,703,260,964.48"）
    yoy: str = ""                   # 同一行的同比，带符号（如 "+1.47%"）；问同比时才有
    page: int | None = None         # 出处页码，进覆盖地图与报告
    period: str = ""                # 命中的期次标签
    scope: str = "未标注"            # 合并 / 母公司 / 未标注
    note: str = ""


class Retriever:
    def __init__(self, load_model: bool = True):
        if not C.VECTORS.exists() or not C.BM25_PKL.exists():
            raise RuntimeError("索引不存在，先跑 index_build.py")

        self.chunks = [json.loads(l) for l in C.CHUNKS.open(encoding="utf-8")]
        self.vecs = np.load(C.VECTORS)                     # 已 L2 归一化
        self.bm25 = bm25.BM25.load(C.BM25_PKL)
        self.backend = self._read_backend()
        self.model = self._make_encoder(self.backend) if load_model else None

        # 预建「每家公司的主要会计数据块」索引，供全景题使用。
        # 关键词捞出来的候选一家有 20~37 个，必须挑，不能直接取第一个：
        # 第一个永远是 p3 的目录页，那样 16 家的锚点就全是目录，
        # 「哪家净利润最高」只能得到 16 页目录（模型只好回答"资料中没有"）。
        # 先按报告期收敛到最新一期，再按 _anchor_score 排好，
        # panorama() 取前 per_company 个。
        self._anchors: dict[str, list[int]] = {}
        # 每家公司的最新报告期。全景题是横向比**同一期**，而普通 top-k 通道
        # 没有这层过滤（问 2026 半年报会混进 2025年报 的块），取证时用它收敛。
        self._newest: dict[str, tuple[int, int]] = {}
        cands: dict[str, list[int]] = {}
        for i, c in enumerate(self.chunks):
            comp = c["company"]
            rk = _report_rank(c["report"])
            if rk > self._newest.get(comp, (0, 0)):
                self._newest[comp] = rk
            if "主要会计数据" in c["text"] or "主要财务指标" in c["text"]:
                cands.setdefault(comp, []).append(i)
        for comp, idxs in cands.items():
            newest = self._newest.get(comp) or max(
                _report_rank(self.chunks[i]["report"]) for i in idxs)
            pool = [i for i in idxs
                    if _report_rank(self.chunks[i]["report"]) == newest]
            self._anchors[comp] = sorted(
                pool, key=lambda i: _anchor_score(self.chunks[i]), reverse=True)

    # ------------------------------------------------------------ 编码器

    @staticmethod
    def _read_backend() -> str:
        """读出建索引时用的后端名，形如 'api:text-embedding-v4'。"""
        if not C.INDEX_META.exists():
            return ""
        try:
            meta = json.loads(C.INDEX_META.read_text(encoding="utf-8"))
        except Exception:
            return ""
        return meta.get("backend") or ""

    @staticmethod
    def _make_encoder(backend: str):
        """按索引记录的后端取编码器。取不到就**报错**，不静默退回本地。"""
        kind = backend.split(":", 1)[0]
        if kind not in ("api", "local"):
            raise RuntimeError(
                "索引里没有记录向量后端（data/index_meta.json 缺 backend 字段），\n"
                "多半是旧版本建的索引。重建即可：\n"
                "    rm -f data/vectors.npy data/index_meta.json\n"
                "    .venv/bin/python index_build.py")
        try:
            return embed_mod.get_embedder(kind)
        except RuntimeError as e:
            if kind == "api":
                raise RuntimeError(
                    f"索引是用百炼 API（{backend}）建的，查询也必须用它编码，\n"
                    f"但现在读不到 DASHSCOPE_API_KEY。\n\n"
                    f"    export DASHSCOPE_API_KEY=sk-xxxx\n\n"
                    f"（若 export 过 DASHSCOPE_BASE_URL，这里也要带上同一站点。）\n"
                    f"想改用本地模型的话索引得整个重建 —— 两个向量空间不可互换。") from e
            raise

    # ------------------------------------------------------------ 向量

    def _vector_search(self, query: str, k: int) -> list[int]:
        # 两个后端的 encode 都做 L2 归一化，所以下面点积即余弦
        q = self.model.encode([query])[0]
        sims = self.vecs @ q                                   # 归一化后点积 = 余弦

        # argpartition 只保证"前 k 个是最大的"，**不保证顺序**。
        # 而 RRF 只看排名（1/(K+rank)），顺序错了融合结果就错了，
        # 而且不会报错 —— 所以必须再排一次。
        k = min(k, len(sims))
        cand = np.argpartition(-sims, k - 1)[:k]
        return [int(i) for i in cand[np.argsort(-sims[cand])]]

    # ------------------------------------------------------------ 融合

    def _rrf(self, *rank_lists: list[int], k: int = None) -> dict[int, float]:
        k = k or C.RRF_K
        fused: dict[int, float] = {}
        for ranks in rank_lists:
            for rank, doc in enumerate(ranks):
                fused[doc] = fused.get(doc, 0.0) + 1.0 / (k + rank + 1)
        return fused

    def _quota(self, fused: dict[int, float], limit: int) -> list[tuple[int, float]]:
        """按分数降序取，但同一家公司最多 MAX_PER_COMPANY 块。"""
        per: dict[str, int] = {}
        out: list[tuple[int, float]] = []
        for doc, score in sorted(fused.items(), key=lambda x: -x[1]):
            co = self.chunks[doc]["company"]
            if per.get(co, 0) >= C.MAX_PER_COMPANY:
                continue
            per[co] = per.get(co, 0) + 1
            out.append((doc, score))
            if len(out) >= limit:
                break
        return out

    # ------------------------------------------------------------ 对外

    def is_panorama(self, query: str) -> bool:
        """是否在问跨公司的横向问题。"""
        if any(w in query for w in PANORAMA_WORDS):
            return True
        # 同时提到 >=2 家公司也算
        named = sum(1 for _, short in C.COMPANIES if short in query)
        return named >= 2

    def search(self, query: str, k: int = None) -> list[dict]:
        k = k or C.TOP_K_FINAL
        v = self._vector_search(query, C.TOP_K_VECTOR)
        b = [d for d, _ in self.bm25.search(query, C.TOP_K_BM25)]
        fused = self._rrf(v, b)
        picked = self._quota(fused, k)
        return [{"chunk": self.chunks[d], "score": s, "doc": d} for d, s in picked]

    # ------------------------------------------------------------ 全景取证

    def _companies_in_query(self, query: str) -> list[tuple[str, str]]:
        """问句点名了公司就只覆盖这几家；没点名（行业级问题）才覆盖全部 16 家。

        「五粮液和洋河股份的营业收入对比」也会被判成全景题，但它只需要两家 ——
        给它塞 16 家的覆盖地图既浪费上下文，又把「其余 14 家缺据」这种
        无关噪声喂给模型，反而干扰它。
        """
        named = [(code, short) for code, short in C.COMPANIES if short in query]
        return named or list(C.COMPANIES)

    def _scope_of_doc(self, doc: int) -> str:
        """这一块的口径，以及**凭什么**说是这个口径。

        原文标明的报表标题极少（9,002 个表格块里可靠带标题的几乎为零），所以主力是
        **制式推定**：第二节「主要会计数据」按披露规则就是合并数，含「100% 占营业收入
        比重」合计行的第三节收入构成表也是合并数。这两条是规则保证，不是从行文里猜的 ——
        但也要标成「规则推定」，不能冒充原文写明了。

        真正的根治要给块打口径标签，那得改 parse.py 从表格 bbox 上方的页面横幅抓标题，
        属于「重建索引」范畴，本次不做（见结论.md 五之 1）。
        """
        c = self.chunks[doc]
        s = _scope_of(c["text"])
        if s != "未标注":
            return f"{s}（原文标明报表标题）"
        if c["section"].startswith("第二节"):
            return "合并（第二节主表·规则推定）"
        if "100%" in _norm(c["text"]):
            return "合并（收入构成合计行·规则推定）"
        if c["section"].startswith("第八节"):
            # 第八节里合并报表与母公司报表的块前缀一模一样（实测天佑德酒 p45 与 p47），
            # 块内没有任何可判别的字。不猜，标出来 —— 老师要的"先核口径"到这里只能停在
            # "核对过了，这份取证的出处无法从块内定口径"。
            return "未标注（第八节表·块内分不出合并/母公司）"
        return "未标注"

    def _bm25_by_company(self, query: str) -> dict[str, list[int]]:
        """一次 BM25 检索，按公司分组（并收敛到各家的最新报告期）。

        每条锚点只收含「主要会计数据」的块，对五粮液/天佑德酒这类锚点损坏的
        公司来说，答案其实躺在别的章节的表里 —— 所以候选要放宽到全库。
        BM25 是纯词法的，不需要查询编码器，这条路在没配 key 时也能跑。
        """
        out: dict[str, list[int]] = {}
        for d, _ in self.bm25.search(query, PANO_BM25_K):
            c = self.chunks[d]
            newest = self._newest.get(c["company"])
            if newest and _report_rank(c["report"]) != newest:
                continue          # 期次收敛：不拿 2025年报 的块去比 2026 半年报
            out.setdefault(c["company"], []).append(d)
        return out

    def _score_candidate(self, d: int, fields: list[str], want_yoy: bool) -> float:
        """择优用的分：以 _anchor_score（数字密度）为底，再按「是不是要的那个数」加权。"""
        c = self.chunks[d]
        t = c["text"]
        if _looks_like_toc(t):
            return -9.0
        s = _anchor_score(c)
        if fields and _field_hit(t, fields):
            s += 1.0
        if want_yoy and _has_yoy(t):
            s += 1.0
        s += _SECTION_BONUS.get(c["section"], 0.0)
        if _PARENT_STMT_RE.search(t):
            s -= 0.5             # 疑似母公司报表，降权（不做硬闸门，见 _scope_of）
        return s

    def _pick_evidence(self, code: str, short: str,
                       fields: list[str], want_yoy: bool,
                       anchors: list[int], pool: list[int]) -> Evidence:
        if not pool:
            return Evidence(code, short, "", "missing",
                            note="该公司在任何通道里都没有候选块")

        anchor_set = set(anchors)
        # 目录页在择优之前就剔掉：它含章节标题 + 页码，会骗过「标签邻近有数字」这条判断
        usable = [d for d in pool if not _looks_like_toc(self.chunks[d]["text"])]
        pool = usable or pool
        ranked = sorted(pool, key=lambda d: -self._score_candidate(d, fields, want_yoy))
        newest = self._newest.get(short)
        report = next((self.chunks[d]["report"] for d in ranked
                       if not newest or _report_rank(self.chunks[d]["report"]) == newest),
                      self.chunks[ranked[0]]["report"])

        def mk(status: str, doc: int | None, **kw) -> Evidence:
            t = self.chunks[doc]["text"] if doc is not None else ""
            hit = _field_hit(t, fields) if (doc is not None and fields) else None
            kw.setdefault("field", hit[1] if hit else "")
            kw.setdefault("value", hit[2] if hit else "")
            kw.setdefault("yoy", _yoy_value(t, hit[2])
                          if (hit and want_yoy) else "")
            return Evidence(code, short, report, status, doc=doc,
                            source=("锚点" if doc in anchor_set else "回退") if doc is not None else "",
                            page=self.chunks[doc]["page"] if doc is not None else None,
                            period=_matched_period(t),
                            scope=self._scope_of_doc(doc) if doc is not None else "未标注",
                            **kw)

        if not fields:
            # 问法不在这几个字段里 → 退回锚点优先，但如实标 partial：
            # 覆盖地图上要能看出「这次没能核对字段」，而不是假装已覆盖。
            pick = ranked[0]
            return mk("partial", pick, note="问句未识别出可核对的字段，仅定位到候选块")

        # 关键取舍：择优键不是「分最高的」，而是五段式 ——
        #   ① 标签精度（「营业收入」优于「营业总收入」，别拿宽口径顶掉精确口径）
        #   ② 表格块优先（叙述段里凑巧挨着的数字算不上证据）
        #   ③ 字段和同比齐不齐（要同比，就得同比那一列也在）
        #   ④ 是不是锚点（锚点是各家同版式的那张主表，同档时锚点优先）
        #   ⑤ 营业收入取命中值最大的那个 —— 只在前面几级完全打平时才用
        #   ⑥ 才轮到 _anchor_score
        #
        # 纯按分取会两头翻车：五粮液锚点块的分（0.46）高于它 p12 的收入构成表（0.31），
        # 但锚点块恰恰没有营业收入，那就再踩一次 Q9 的坑；反过来茅台 p5 锚点是好的，
        # 却会被第三节某张数字更密的表顶掉，改用宽口径的营业总收入 —— 另一个坑。
        #
        # ② 那条是拿天佑德酒换来的：它第二节主表丢了行标签（锚点坏），兜底池只剩子项
        # 表和叙述段 —— p14 那块是一段青稞采购单价的叙述，「占当期营业收入总额」
        # 后面 120 字内正好有数字，字面上就算命中了。这种块够不到 covered，最多 partial。
        #
        # ⑤ 只对营业收入用，且**放在锚点之后**：分公司整体的收入不可能小于任何分项，
        # 所以"取最大"能把整体和分项分开（天佑德酒从 p22 那个乱码表回到 p45 合并利润表的
        # 574,885,860.71）。但这一条绝不能提级 —— 试过放到锚点之前，12 家健康公司全被
        # 第八节里含「营业收入」字样、后面跟着更大数字（总资产、累计发生额）的杂块顶掉。
        # 净利润、现金流也不能这么办：子公司的净利润、分部的现金流都可能比整体大。
        use_max = "营业收入" in fields

        def key(d: int) -> tuple:
            c = self.chunks[d]
            t = c["text"]
            hit = _field_hit(t, fields)
            is_tbl = c["type"] == "table"
            # 先分档、再比标签精度。档是「这块到底能不能当证据」，精度只是「写得全不全」：
            #   0 表格 + 字段（严格标签）+ 同比齐  → 够得上 covered
            #   1 表格 + 字段，但缺同比或只是兜底标签 → 最多 partial
            #   2 有字段但不在表格里     → 最多 partial（叙述段里挨着的数字不算证据）
            #   3 连字段都没有           → missing
            # 分档必须压在精度前面：天佑德酒 p11 是一段正文，里面「归属于上市公司股东的
            # 净利润」后面跟着 1,044.01，标签精度满分；p104 的未分配利润附注表里才是真的
            # 归母数 10,440,139.66。只按精度排，正文就会把表顶掉。
            # 同比那一格必须是**这一行**的（见 _yoy_value）。用整块的 _has_yoy 会
            # 让「自己没有同比、下一行有」的块混进 0 档。
            row_yoy = _yoy_value(t, hit[2]) if (hit and want_yoy) else ""
            tier = (0 if (hit and is_tbl and _is_strict(hit, fields)
                          and (not want_yoy or row_yoy))
                    else 1 if (hit and is_tbl)
                    else 2 if hit else 3)
            return (tier,
                    hit[0] if hit else 9,
                    0 if d in anchor_set else 1,
                    -_to_float(hit[2]) if (hit and use_max) else 0.0,
                    -self._score_candidate(d, fields, want_yoy))

        pick = min(ranked, key=key)
        t = self.chunks[pick]["text"]
        is_table = self.chunks[pick]["type"] == "table"
        hit = _field_hit(t, fields)

        if hit is None:
            has_label = any(lab in t for name in fields for lab in FIELD_SPECS[name])
            return mk("partial" if has_label else "missing", pick,
                      note=("块里有该字段的标签，但未取到配对的数值"
                            if has_label else f"未取到「{'、'.join(fields)}」"))

        # 覆盖的门槛是「这一行的同比也取到了」，不是「这块里有同比字样」：
        # 后者会让一家没披露同比的公司被别行的同比顶成 covered（见 _yoy_value）。
        yoy = _yoy_value(t, hit[2]) if want_yoy else ""
        status = ("covered"
                  if (not want_yoy or yoy) and is_table and _is_strict(hit, fields)
                  else "partial")
        if status == "covered":
            note = ""
        elif not is_table:
            note = "块里有该字段的数值，但它不是表格块，不作覆盖依据"
        elif not _is_strict(hit, fields):
            note = f"只取到兜底写法的「{hit[1]}」，不敢当这一口径用，需核对原文"
        else:
            # 走到这儿只剩一种：要同比，而**这一行**没有同比那一格。
            # （不要同比的题 want_yoy=False，前两条已经把所有 partial 的成因分完了。）
            note = "取到字段值，但这一行没有同比那一格"
        return mk(status, pick, note=note)

    def panorama_evidence(self, query: str) -> list[Evidence]:
        """按公司取证，返回覆盖地图（问句没点名公司时即 16 家）。"""
        fields = _query_fields(query)
        want_yoy = _wants_yoy(query)
        by_company = self._bm25_by_company(query)
        out: list[Evidence] = []
        for code, short in self._companies_in_query(query):
            anchors = self._anchors.get(short, [])
            seen = set(anchors)
            pool = list(anchors)
            for d in by_company.get(short, []):
                if d not in seen:
                    seen.add(d)
                    pool.append(d)
            out.append(self._pick_evidence(code, short, fields, want_yoy,
                                           anchors, pool))
        if want_yoy:
            # 按**被问的那一列**排序（同比降序，没有同比的排最后）。
            # 问「哪几家正增长」时，正号就全挤在地图开头那几行，名单变成一段前缀 ——
            # 模型不必在 16 行同构数据里挑。实测它偏偏栽在这一步：同比列已经摆在地图上了
            # （五粮液那行白纸黑字写着 +20.87%），它照样写"其余各家同比均为负增长，无例外"。
            #
            # 排序是**展示顺序**，不是替它下结论：哪几家仍要它自己按公司名列出来。
            out.sort(key=lambda e: (0, -_yoy_num(e.yoy))
                     if _yoy_num(e.yoy) is not None else (1, 0.0))
        return out

    def panorama(self, query: str) -> list[dict]:
        """全景题专用通道：把取证选中的块拼成 hits（保持原有的返回形状）。

        保留这个壳是因为 server 与 eval 都按 hits 的形状消费；
        要拿覆盖地图请直接调 panorama_evidence()，它多带了每家的状态与出处。
        """
        out = []
        for ev in self.panorama_evidence(query):
            if ev.doc is None:
                continue
            out.append({"chunk": self.chunks[ev.doc], "score": 0.0, "doc": ev.doc})
        return out


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    if args and args[0] == "--pano":
        # 全景取证自检：BM25 是纯词法的，不碰查询编码器 ——
        # 所以这条路径**没配 DASHSCOPE_API_KEY 也能跑**，不用等 key 就能验覆盖。
        r = Retriever(load_model=False)
        print(f"索引 {len(r.chunks):,} 块")
        queries = args[1:] or [
            "16家白酒公司哪家净利润最高？",
            "这16家白酒公司里，2026年上半年营业收入同比实现正增长的有哪几家？"
            "各家增速分别是多少？",
            "五粮液和洋河股份的营业收入对比",
        ]
        marks = {"covered": "✓", "partial": "~", "missing": "✗"}
        for q in queries:
            evs = r.panorama_evidence(q)
            n = {s: sum(1 for e in evs if e.status == s) for s in marks}
            print(f"\nQ: {q}")
            print(f"   覆盖 {n['covered']}/{len(evs)} 家"
                  f"（partial {n['partial']}、missing {n['missing']}）")
            for e in evs:
                where = (f"{e.source} p{r.chunks[e.doc]['page']}"
                         if e.doc is not None else "—")
                print(f"  {marks[e.status]} {e.company:6s} {where:9s} "
                      f"{e.field or '（未命中字段）':12s} {e.value:>18s} "
                      f"{e.yoy or '—':>8s} {e.period:6s} 口径={e.scope}  {e.note}")
        raise SystemExit(0)

    r = Retriever()
    print(f"索引 {len(r.chunks):,} 块，锚点覆盖 {len(r._anchors)}/16 家")
    for q in ["贵州茅台2026年上半年直销渠道收入是多少？",
              "16家白酒公司哪家净利润最高？"]:
        print()
        print(f"Q: {q}   [全景={r.is_panorama(q)}]")
        for h in r.search(q, k=4):
            c = h["chunk"]
            print(f"  {h['score']:.4f}  {c['company']:6s} {c['report']:9s} "
                  f"{c['section'][:12]:14s} p{c['page']:<4} {c['text'][:44]}")
