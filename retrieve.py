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

import numpy as np

import bm25
import config as C
import embed as embed_mod

# 全景题信号：这些问题问的是"跨公司"，不是"某一家"
PANORAMA_WORDS = (
    "对比", "比较", "各家", "各公司", "哪家", "谁家", "排名", "排行",
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
        cands: dict[str, list[int]] = {}
        for i, c in enumerate(self.chunks):
            if "主要会计数据" in c["text"] or "主要财务指标" in c["text"]:
                cands.setdefault(c["company"], []).append(i)
        for comp, idxs in cands.items():
            newest = max(_report_rank(self.chunks[i]["report"]) for i in idxs)
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

    def panorama(self, query: str, per_company: int = 1) -> list[dict]:
        """全景题专用通道：每家公司都取一份「主要会计数据」块。

        纯 top-k RAG 在架构上答不了「16 家谁净利润最高」——
        它最多召回 6 个块，覆盖不到 16 家公司，模型只能凭残缺上下文编。
        这里绕过相似度，直接按公司把锚点块全捞出来，让 16 家都在场。
        """
        out = []
        for code, short in C.COMPANIES:
            for d in self._anchors.get(short, [])[:per_company]:
                out.append({"chunk": self.chunks[d], "score": 0.0, "doc": d})
        return out


if __name__ == "__main__":
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
