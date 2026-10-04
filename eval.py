"""
第六步：评测 —— 10 道题逐题跑，记录「召回了哪些块 / 答对没有 / 错在哪」。

为什么每题要判**两个**信号：
    只看最终回答对不对，无法区分两种完全不同的失败 ——
      · 召回失败：正确的块压根没被捞出来，模型再强也答不出；
      · 生成失败：正确的块就在上下文里，模型却读错、算错、或不采信。
    这两种的修法完全不同（一个调检索，一个调提示词），所以每题都判：
      召回命中 = 召回的块里有没有标准答案
      回答命中 = 模型最终回答里有没有标准答案
    四象限：召回✓回答✓ 答对；召回✓回答✗ 生成的问题；
            召回✗回答✓ 罕见（模型先验或碰巧）；召回✗回答✗ 检索的问题。

题目不是随手编的，每道都先回原文核过标准答案（见每题的 考点 字段）：
    · 单点事实题（正文 / 表格中间行）—— 考最基本的取数
    · 坏表题 —— 五粮液 p7「主要会计数据」渲染错位（列被识别成转置），
      但同一批数字在 p12「营业收入构成」里是好的，考"换个地方捞得到吗"
    · 口径题 —— 归母净利润 vs 扣非归母净利润，两个数很接近，容易张冠李戴
    · 跨期 / 跨公司对比 —— 考多数字的取用与计算
    · 全景题（3 道）—— 考 16 家横向汇聚，纯 top-k 架构上答不了这类题
    · 防编造题 —— 问一家**不在语料里**的公司（顺鑫农业，config 里明确剔除），
      正确答案是"没有相关信息"，答出任何数字都算错

跑法：
    export DASHSCOPE_API_KEY=sk-xxxx
    .venv/bin/python eval.py            # 跑全部 10 题
    .venv/bin/python eval.py 3 7        # 只跑第 3、7 题
    .venv/bin/python eval.py --no-llm   # 只检索不生成（不用 key 也能跑）

产出：
    data/eval_report.md   人读的逐题报告（含召回块清单 + 回答）
    data/eval_raw.json    机器可读的原始结果
"""

from __future__ import annotations

import json
import os
import sys
import time

import config as C

# ---------------------------------------------------------------- 题目

# 每题的 checks 是"标准答案里必须出现的东西"，any 里是一个数字的几种写法
# （原值 / 去逗号 / 亿元 / 万元）—— 模型写 2.12亿 和写 211,867,041.23 都对。
# 全部 checks 都命中才算答对。
QUESTIONS: list[dict] = [
    {
        "id": 1,
        "q": "金徽酒2026年上半年经营活动产生的现金流量净额是多少？",
        "type": "单点事实 · 表格中间行",
        "gold_desc": "211,867,041.23 元（约 2.12 亿）",
        "checks": [{"desc": "现金流量净额",
                    "any": ["211,867,041.23", "211867041.23", "2.12亿", "2.1187亿"]}],
        # 195,709,201.66 是 p51「母公司现金流量表」的数，不是合并口径。
        # 实测模型答的就是它 —— 因为含正确数字的 p49 块渲染有伪影
        # （数值串进了列名位置），模型判定该块不可信，转投了格式干净的母公司表。
        "forbid": ["195,709,201.66", "195709201.66"],
        "note": "合并 vs 母公司口径陷阱。p49 表头写明「合并现金流量表」、"
                "p51 属「母公司现金流量表」，两个数相差 1,616 万。",
    },
    {
        "id": 2,
        "q": "水井坊2026年上半年归属于上市公司股东的净利润是多少？同比是增是减？",
        "type": "单点事实 · 由盈转亏",
        "gold_desc": "-6,222,478.74 元（亏损），同比 -105.9%",
        "checks": [
            {"desc": "归母净利润", "any": ["-6,222,478.74", "-6222478.74",
                                          "-622.25万", "-622万"]},
            {"desc": "同比降幅", "any": ["-105.9", "105.9"]},
        ],
    },
    {
        "id": 3,
        "q": "五粮液2026年上半年营业收入是多少？同比增减多少？",
        "type": "坏表题 · 换路径取数",
        "gold_desc": "28,416,674,541.77 元（约 284.17 亿），同比 +20.87%",
        "checks": [
            {"desc": "营业收入", "any": ["28,416,674,541.77", "28416674541.77",
                                        "284.17亿", "284.2亿"]},
            {"desc": "同比增幅", "any": ["20.87"]},
        ],
        "note": "p7「主要会计数据」渲染错位（表头行选错 + 列整体串位），"
                "营业收入压根没出现在锚点块里；但 p12「营业收入构成」有正确的"
                "本报告期金额，考的是能不能从别的块里把它捞出来。",
    },
    {
        "id": 4,
        "q": "泸州老窖2026年上半年归属于上市公司股东的净利润是多少？",
        "type": "口径题 · 归母 vs 扣非",
        "gold_desc": "4,339,264,086.33 元（约 43.39 亿）",
        "checks": [{"desc": "归母净利润",
                    "any": ["4,339,264,086.33", "4339264086.33", "43.39亿", "43.4亿"]}],
        "forbid": ["4,291,450,847.49", "4291450847.49", "42.91亿"],
        "note": "扣非归母是 4,291,450,847.49，与归母只差 1.1%，"
                "回答里出现它（或 42.91亿）就是口径答错。",
    },
    {
        "id": 5,
        "q": "贵州茅台2026年上半年营业收入相比上年同期是增长还是下降？幅度多少？",
        "type": "跨期对比 · 同公司",
        "gold_desc": "本期 90,703,260,964.48，上年同期 89,389,354,416.84，同比 +1.47%",
        "checks": [
            {"desc": "本期营收", "any": ["90,703,260,964.48", "90703260964.48",
                                        "907.03亿", "907亿"]},
            {"desc": "同比幅度", "any": ["1.47"]},
        ],
    },
    {
        "id": 6,
        "q": "贵州茅台和山西汾酒2026年上半年归属于上市公司股东的净利润分别是多少？两家差多少？",
        "type": "跨公司对比 · 两家",
        "gold_desc": "茅台 44,516,880,421.86；汾酒 6,439,313,574.11；差 38,077,566,847.75",
        "checks": [
            {"desc": "茅台净利润", "any": ["44,516,880,421.86", "44516880421.86",
                                          "445.17亿", "445.2亿"]},
            {"desc": "汾酒净利润", "any": ["6,439,313,574.11", "6439313574.11",
                                          "64.39亿", "64.4亿"]},
            {"desc": "差额", "derived": True,
             "any": ["38,077,566,847.75", "38077566847.75",
                     "380.78亿", "380.8亿"]},
        ],
        "note": "差额是算出来的，原文里不可能有 —— 标了 derived，只判回答不判召回。",
    },
    {
        "id": 7,
        "q": "这16家白酒公司里，2026年上半年营业收入最高的是哪家？金额是多少？",
        "type": "全景题 · 极值",
        "gold_desc": "贵州茅台 90,703,260,964.48 元（907.03 亿）",
        "checks": [
            {"desc": "公司", "any": ["贵州茅台"]},
            {"desc": "金额", "any": ["90,703,260,964.48", "90703260964.48",
                                    "907.03亿", "907亿"]},
        ],
        "note": "五粮液实际营收 284.17 亿排第二，但它的锚点表是坏的 —— "
                "看模型会不会漏掉它、或把第二名说错。",
    },
    {
        "id": 8,
        "q": "这16家白酒公司里，2026年上半年归母净利润为负的公司有哪几家？各家分别是多少？",
        "type": "全景题 · 异常值",
        "gold_desc": "水井坊 -6,222,478.74 元、金种子酒 -64,838,789.06 元",
        "checks": [
            {"desc": "水井坊", "any": ["水井坊"]},
            {"desc": "金种子酒", "any": ["金种子酒"]},
        ],
        "note": "两家都是小额亏损，容易被「赚得多」的大数盖过去；"
                "水井坊的负号尤其关键（-622 万，不是 +622 万）。",
    },
    {
        "id": 9,
        "q": "这16家白酒公司里，2026年上半年营业收入同比实现正增长的有哪几家？各家增速分别是多少？",
        "type": "全景题 · 方向筛选",
        "gold_desc": "五粮液 +20.87%、迎驾贡酒 +8.08%、金徽酒 +3.19%、贵州茅台 +1.47%",
        # 名称 + 增速**两两配对**，不能只查名称。
        # 实测模型回答了「五粮液…该片段未提供营业收入同比增减率 → 无法判断」——
        # 只查名称的话这也算命中，而它恰恰漏掉了增速最高的那家。
        "checks": [
            {"desc": "五粮液 +20.87%", "any": ["五粮液"]},
            {"desc": "五粮液增速", "any": ["20.87"]},
            {"desc": "迎驾贡酒 +8.08%", "any": ["迎驾贡酒"]},
            {"desc": "迎驾贡酒增速", "any": ["8.08"]},
            {"desc": "金徽酒 +3.19%", "any": ["金徽酒"]},
            {"desc": "金徽酒增速", "any": ["3.19"]},
            {"desc": "贵州茅台 +1.47%", "any": ["贵州茅台"]},
            {"desc": "贵州茅台增速", "any": ["1.47"]},
        ],
        "note": "最难的题：既要覆盖 16 家，又要判方向。白酒 2026H1 普遍下滑，"
                "正增长只有 4 家，且其中增速最高的五粮液恰好是锚点表坏掉的那家。"
                "若模型只报茅台一家（最容易召回的），就是覆盖率不足。",
    },
    {
        "id": 10,
        "q": "顺鑫农业2026年上半年的营业收入是多少？",
        "type": "防编造 · 语料外公司",
        "gold_desc": "语料中没有该公司，正确回答是「提供的资料中没有相关信息」",
        "checks": [],
        "refuse": True,
        "note": "顺鑫农业（000860）主业含大量地产，config 里明确剔除，"
                "语料里 0 个块。答出任何具体数字都是编造。",
    },
]


# 人工复核结论。自动判定只看「标准答案在不在」，判不出「为什么错」——
# 那得读模型的推理过程。这部分是作业要求的"错在哪"，所以写进脚本，
# 让它跟结果一起进报告，而不是散在聊天记录里。
VERDICTS: dict[int, str] = {
    1: "**生成失败（口径）**。检索没问题：含正确数（合并口径 211,867,041.23）的 "
       "p49 块就在上下文里，p7/p16/p49 三处互相印证。但 p49 的渲染有伪影——"
       "一个数值串进了列名位置（「收到的税费返还 | 1,922,635,320.51: 432,821.14」），"
       "模型据此判定该块「格式异常、无法清晰对应本期/上年同期」，转而采信了格式"
       "干净的另一块——而那是 p51 的**母公司现金流量表**（195,709,201.66）。"
       "错因在表格渲染，不在检索，也不在提示词。",
    2: "答对。由盈转亏、负号、同比降幅全部正确。",
    3: "答对，且正是本题想考的「换路径取数」：避开了渲染错位的 p7 主表，"
       "从 p12「营业收入构成」取到 284.17 亿，还自己验算了 20.87% 与两个"
       "绝对值变动一致。",
    4: "答对。取的是归母 43.39 亿，没有混入只差 1.1% 的扣非 42.91 亿。",
    5: "答对。额外加分：主动指出管理层讨论里的「营业总收入 922.78 亿 / +1.3%」"
       "与「营业收入 +1.47%」是两个口径（前者含利息收入等），并说明不影响结论。",
    6: "答对。两家数字都取对，差额也算对。注意本题触发了全景通道"
       "（6 块 → 22 块），对一个只涉及两家的题是上下文浪费。",
    7: "答对，但推理有**过度断言**：模型说「所有其他公司营收均在 100 亿元以下」，"
       "而它自己上一句刚说五粮液数据缺失——五粮液实际 284.17 亿。"
       "答案对，但这个「全部低于」的结论它无法验证。",
    8: "答对，质量高。主动排除了两个陷阱：曲阜孔府家子公司净利润 -280 万"
       "（子公司口径，不计入归母），以及天佑德酒那张丢标签的表"
       "（「无法确认哪一行为归母净利润，故不采信」）。",
    9: "**检索失败**。五粮液实际营收同比 +20.87%，是 16 家里增速最高的，"
       "但它被召回的锚点块是渲染错位的 p7 主表——不含营业收入、不含 20.87%，"
       "增速从未进入上下文，模型只好答「无法判断」。全景通道 16 个锚点里坏 1 个，"
       "那一家就从全景视野里整个消失，而模型只会说「无法判断」，不会说"
       "「我缺了一家的数据」。附带问题：召回里混入一个口子窖 2025年报 的块，"
       "top-k 通道没有按期次过滤。",
    10: "答对。明确拒绝，没有编造任何数字。",
}


# 判定

def norm(s: str) -> str:
    """归一化：去掉千分位逗号、各种空白、全角负号，方便比对数字。"""
    for ch in ("，", ",", " ", "　", "元", "％"):
        s = s.replace(ch, "")
    return s.replace("−", "-").replace("－", "-").replace("%", "")


def matches(text: str, alts: list[str]) -> bool:
    t = norm(text)
    return any(norm(a) in t for a in alts)


REFUSE_MARKERS = ("没有相关信息", "未提供", "没有提供", "不包含", "未披露",
                  "无法回答", "资料中没有", "未包含", "没有找到", "不含")


def judge(q: dict, text: str, recall: bool = False) -> bool:
    """判一段文本（召回块 或 模型回答）里有没有标准答案。

    recall=True 时只判「原文块里本该出现的东西」：
    派生值（如两家净利润之差）永远不可能出现在原文里 —— 它是模型算出来的。
    这类检查项标了 derived 并在召回判定里跳过。否则每道计算题都会被误判成
    「检索失败」，而实际是检索到了、模型也算对了 —— 一个假阴性会直接
    把结论引向「去调检索」，方向就错了。
    """
    if q.get("refuse"):
        return any(m in text for m in REFUSE_MARKERS)
    # 口径陷阱：答成母公司数（金徽酒）或扣非数（泸州老窖）都算错
    if not recall and any(matches(text, [f]) for f in q.get("forbid", [])):
        return False
    checks = [c for c in q["checks"] if not (recall and c.get("derived"))]
    return all(matches(text, c["any"]) for c in checks)


# ---------------------------------------------------------------- 主流程

def write_report(results: list[dict], n_chunks: int, backend: str) -> None:
    """把评测结果写成 markdown。

    抽成函数是为了 --report-only：改人工复核结论时，不该再花 10 次大模型调用。
    """
    ok = {True: "✓", False: "✗", None: "—"}
    md = ["# 评测报告 · 白酒财报问答", "",
          f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}　"
          f"索引 {n_chunks:,} 块　后端 {backend}", "",
          "判定口径：**召回** = 召回的块里含标准答案；**回答** = 模型最终回答含标准答案。",
          "两者分开记，才能分清是检索的问题还是生成的问题。", "",
          "| 题号 | 考点 | 召回 | 回答 |", "|---|---|---|---|"]
    for x in results:
        md.append(f"| {x['id']} | {x['type']} | {ok[x['recall_hit']]} "
                  f"| {ok[x['answer_hit']]} |")
    md.append("")

    for x in results:
        md += [f"## Q{x['id']}　{x['type']}", "",
               f"**问题**：{x['q']}", "",
               f"**标准答案**：{x['gold_desc']}", ""]
        if x["note"]:
            md += [f"**出题意图**：{x['note']}", ""]
        md += [f"**召回**：检索 {x['n_search']} 块"
               + (f"（触发全景通道，合计 {x['n_total']} 块）" if x["panorama"] else ""),
               ""]
        for h in x["hits"]:
            md += [f"- `[{h['n']}]` {h['company']}　{h['report']}　"
                   f"{h['section'] or '未定'}　p{h['page']}　"
                   f"{h['type']}　单位={h['unit'] or '无'}　score={h['score']}",
                   f"  > {h['text'][:220].replace(chr(10), ' ⏎ ')}"]
        md += ["", f"**回答**（召回 {ok[x['recall_hit']]} / 回答 {ok[x['answer_hit']]}）：",
               "", "```", x["answer"], "```", ""]
        if VERDICTS.get(x["id"]):
            md += [f"**人工复核**：{VERDICTS[x['id']]}", ""]

    C.DATA.joinpath("eval_report.md").write_text("\n".join(md), encoding="utf-8")


def main() -> int:
    args = [a for a in sys.argv[1:]]
    # 脱机重生成报告：只读上次的 eval_raw.json，不碰大模型。
    if "--report-only" in args:
        results = json.loads(
            C.DATA.joinpath("eval_raw.json").read_text(encoding="utf-8"))
        meta = json.loads(C.INDEX_META.read_text(encoding="utf-8"))
        write_report(results, meta["n_chunks"], meta["backend"])
        print(f"✅ 已按 data/eval_raw.json 重写 eval_report.md（{len(results)} 题）")
        return 0
    no_llm = "--no-llm" in args
    picked = {int(a) for a in args if a.isdigit()}
    todo = [q for q in QUESTIONS if not picked or q["id"] in picked]

    import retrieve
    import server

    print("加载索引 …")
    r = retrieve.Retriever()
    print(f"  {len(r.chunks):,} 块，后端 {r.backend}，锚点覆盖 {len(r._anchors)}/16 家")

    has_key = bool(os.environ.get("DASHSCOPE_API_KEY"))
    if not no_llm and not has_key:
        print("\n❌ 没设 DASHSCOPE_API_KEY，改用 --no-llm 只检索。")
        return 1
    print(f"  生成模型: {C.CHAT_MODEL if not no_llm else '（不生成）'}")
    print()

    results = []
    for q in todo:
        t0 = time.time()
        hits = r.search(q["q"], k=C.TOP_K_FINAL)
        panorama = r.is_panorama(q["q"])
        n_search = len(hits)
        if panorama:
            seen = {h["doc"] for h in hits}
            hits = hits + [h for h in r.panorama(q["q"]) if h["doc"] not in seen]

        context, cites = server.build_context(hits)

        # 召回命中：召回的块里有没有标准答案（与模型无关，纯检索质量）
        rec_hit = None
        if not q.get("refuse"):
            joined = "\n".join(h["chunk"]["text"] for h in hits)
            rec_hit = judge(q, joined, recall=True)

        answer, err = "", ""
        if no_llm:
            answer = "（--no-llm：未调用生成模型）"
            ans_hit = None
        else:
            try:
                answer = server.call_dashscope(q["q"], context)
                ans_hit = judge(q, answer)
            except Exception as e:
                answer, ans_hit = f"❌ {type(e).__name__}: {e}", False

        ok = {True: "✓", False: "✗", None: "—"}
        print(f"Q{q['id']:<2} {q['type']:<22} 召回 {ok[rec_hit]}  "
              f"回答 {ok[ans_hit]}  ({time.time()-t0:.1f}s)")
        if err:
            print(f"     {err}")

        results.append({
            "id": q["id"], "q": q["q"], "type": q["type"],
            "gold_desc": q["gold_desc"], "note": q.get("note", ""),
            "panorama": panorama, "n_search": n_search, "n_total": len(hits),
            "recall_hit": rec_hit, "answer_hit": ans_hit, "answer": answer,
            "hits": [{
                "n": i, "company": h["chunk"]["company"],
                "report": h["chunk"]["report"], "section": h["chunk"]["section"],
                "page": h["chunk"]["page"], "type": h["chunk"]["type"],
                "unit": h["chunk"].get("unit", ""), "score": round(h["score"], 5),
                "text": h["chunk"]["text"],
            } for i, h in enumerate(hits, 1)],
        })

    # ------------------------------------------------------------ 落盘
    C.DATA.joinpath("eval_raw.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    write_report(results, len(r.chunks), r.backend)

    n_ok = sum(1 for x in results if x["answer_hit"] is True)
    n_rec = sum(1 for x in results if x["recall_hit"] is True)
    n = len(results)
    print()
    print("=" * 62)
    print(f"  召回命中 {n_rec}/{n}　回答命中 {n_ok}/{n}")
    quad = {"答对": [], "生成失败": [], "检索失败": [], "存疑": []}
    for x in results:
        r, a = x["recall_hit"], x["answer_hit"]
        if r is None:
            key = "答对" if a else "存疑"
        elif r and a:
            key = "答对"
        elif r and not a:
            key = "生成失败"        # 块在，模型没用对
        elif not r and a:
            key = "存疑"            # 块不在却答对了，多为派生值或模型先验
        else:
            key = "检索失败"        # 块没召回，模型也没答对
        quad[key].append(x["id"])
    print()
    for k, v in quad.items():
        if v:
            print(f"  {k}（{len(v)}）：Q{', Q'.join(map(str, v))}")
    print()
    print(f"  → data/eval_report.md　data/eval_raw.json")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
