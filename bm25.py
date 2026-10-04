"""
BM25 关键词检索 —— 和向量检索互补的另一条腿。

为什么要两条腿：向量擅长"意思相近"（问"直销收入"能召回"直接销售取得的收入"），
但对**精确的字符串**反而迟钝 —— 问"600779"或"3.2 亿元"这种，向量会把
意思相近但公司不对的块排在前面。BM25 恰好相反。所以两个都跑，再用 RRF 融合。

中文分词的两个坑：

坑一：不能按空格切
    中文没有词边界，"贵州茅台直销收入"按空格切就是一整块，等于没切。

坑二：不能按字切
    单字太泛，"的""公""司"到处都是，IDF 压不住，检索结果全是噪声。

解法：**字 bigram**。把"贵州茅台"切成 贵州/州茅/茅台。这是无词典中文检索的
标准做法，不需要 jieba 那种外部分词器（本项目不引额外依赖）。

坑三：IDF 必须恒正
    BM25 的经典 IDF 是 log((N-df+0.5)/(df+0.5))。
    当某个词出现在超过一半的文档里（df > N/2），这个值是**负数** ——
    于是包含这个词的文档反而被扣分，检索结果完全反了。
    财报里"公司""报告期"这类词到处都是，必然触发。
    所以用恒正形式：log(1 + (N-df+0.5)/(df+0.5))，最小值是 0，不会为负。

用法：
    from bm25 import BM25
    idx = BM25.build(texts)
    idx.save(path)  /  BM25.load(path)
    idx.search("茅台上半年直销收入", k=20)   # → [(doc_idx, score), ...]
"""

from __future__ import annotations

import math
import pickle
import re
from collections import Counter
from pathlib import Path

CJK_RE = re.compile(r"[一-鿿]")
# 英文字母 / 数字 / 百分数 / 带千分位的金额，整体保留
ALNUM_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_.\-]*|\d[\d,]*\.?\d*%?")

K1 = 1.5      # 词频饱和参数
B = 0.75      # 长度归一化强度


def tokenize(text: str) -> list[str]:
    """汉字切 bigram，英文数字整体保留。"""
    toks: list[str] = []

    # 汉字 bigram：先把所有汉字抽出来再相邻两两组合。
    # 注意不能按原字符串的相邻位置切 —— 中间夹着数字或标点时，
    # "茅台2026年" 里的 台/年 会组合成 "台年" 这种没意义的 bigram。
    han = CJK_RE.findall(text)
    for i in range(len(han) - 1):
        toks.append(han[i] + han[i + 1])
    if len(han) == 1:
        toks.append(han[0])

    for m in ALNUM_RE.finditer(text):
        toks.append(m.group().lower())

    return toks


class BM25:
    def __init__(self, postings, df, doc_len, n_docs, avgdl, vocab):
        self.postings = postings      # term -> list[(doc_idx, tf)]
        self.df = df                  # term -> 文档频率
        self.doc_len = doc_len        # list[int]
        self.n_docs = n_docs
        self.avgdl = avgdl
        self.vocab = vocab

    # ------------------------------------------------------------ 构建

    @classmethod
    def build(cls, texts: list[str], progress=None) -> "BM25":
        postings: dict[str, list[tuple[int, int]]] = {}
        doc_len: list[int] = []

        for i, t in enumerate(texts):
            tf = Counter(tokenize(t))
            doc_len.append(sum(tf.values()) or 1)
            for term, n in tf.items():
                postings.setdefault(term, []).append((i, n))
            if progress and i % 2000 == 0:
                progress(i, len(texts))

        df = {term: len(pl) for term, pl in postings.items()}
        avgdl = sum(doc_len) / max(1, len(doc_len))
        return cls(postings, df, doc_len, len(texts), avgdl,
                   {t: i for i, t in enumerate(postings)})

    # ------------------------------------------------------------ 检索

    def _idf(self, term: str) -> float:
        """恒正 IDF —— 见文件头「坑三」。最小值 0，永远不会变成负分。"""
        df = self.df.get(term, 0)
        return math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 20) -> list[tuple[int, float]]:
        q_terms = tokenize(query)
        if not q_terms:
            return []

        scores: dict[int, float] = {}
        for term in set(q_terms):
            pl = self.postings.get(term)
            if not pl:
                continue
            idf = self._idf(term)
            if idf <= 0:
                continue
            for doc_idx, tf in pl:
                dl = self.doc_len[doc_idx]
                denom = tf + K1 * (1 - B + B * dl / self.avgdl)
                scores[doc_idx] = scores.get(doc_idx, 0.0) + idf * tf * (K1 + 1) / denom

        return sorted(scores.items(), key=lambda x: -x[1])[:k]

    # ------------------------------------------------------------ 持久化

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "postings": self.postings, "df": self.df, "doc_len": self.doc_len,
                "n_docs": self.n_docs, "avgdl": self.avgdl, "vocab": self.vocab,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> "BM25":
        with open(path, "rb") as f:
            d = pickle.load(f)
        return cls(d["postings"], d["df"], d["doc_len"],
                   d["n_docs"], d["avgdl"], d["vocab"])


def smoke() -> None:
    """自查：中文能不能检索到，以及 IDF 有没有负分。"""
    docs = [
        "贵州茅台2026年上半年直销渠道收入5,196,204.70万元",
        "五粮液2026年上半年营业收入为52,071,594,062.35元",
        "本报告期内公司白酒产量为33,732.27吨，同比增加8.55%",
        "董事会决议通过的本报告期利润分配预案",
    ]
    idx = BM25.build(docs)
    print("== BM25 自查 ==")
    for q in ["茅台上半年直销收入", "五粮液营业收入", "白酒产量"]:
        hits = idx.search(q, k=2)
        top = docs[hits[0][0]][:34] if hits else "(无)"
        print(f"  {q:14s} → {top}  ({hits[0][1]:.2f})" if hits else f"  {q} → 无命中")

    neg = [t for t in idx.df if idx._idf(t) <= 0]
    print(f"  IDF 非正的词: {len(neg)} 个  {'✅' if not neg else '❌ 会出现负分'}")


if __name__ == "__main__":
    smoke()
