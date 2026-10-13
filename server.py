"""
第五步：把检索和大模型接起来，起一个本地服务。

密钥处理（这是硬约束，不能妥协）：
    API key 只从环境变量 DASHSCOPE_API_KEY 读，绝不写进任何文件，
    也绝不下发到浏览器。浏览器只能调本机的 /api/ask，由服务端去访问百炼。
    所以 web/index.html 里没有、也不该有任何密钥。

启动：
    export DASHSCOPE_API_KEY=sk-xxxx        # 只在当前终端会话有效
    cd ~/projects/财报知识库
    .venv/bin/python server.py
    然后浏览器打开 http://127.0.0.1:8000

没设 key 也能跑：会退化成"只检索不生成"，把召回的原文和出处列出来，
方便先验证检索质量，不用等 key。

URL 必须用 127.0.0.1 而不是 localhost ——
实测这台机器上 localhost 有时会先解析到 ::1，而服务只监听了 IPv4，
浏览器报"拒绝连接"，但 curl 127.0.0.1 又是通的，很容易误判。
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import config as C
import net
# 只为复用 retrieve._yoy_num（把地图里的 "+20.87%" 还原成 20.87）来数方向小计。
# 符号解析只有这一份实现，别在渲染层再写一遍 —— 两处解析口径迟早会分叉。
# retrieve 不反向 import server，没有循环。
import retrieve

PORT = 8000
WEB = C.ROOT / "web"

SYSTEM_PROMPT = """你是白酒行业财报分析助手。根据给定的财报原文片段回答问题。

规则：
1. 只依据提供的片段回答，不要用你自己的先验知识补充数字。
2. 每个数字后面用 [编号] 标出它来自哪个片段。
3. 片段里如果有「（单位：元）」这类标注，务必按该单位作答；
   如果某个数字所在的表格**没有**单位标注，要明确说明"该表未标注单位，"
   并提醒需核对原文。
4. 片段里没有的，直接说"提供的资料中没有相关信息"，不要推测。
5. 涉及多空对比或多期对比时，把口径说清楚（本期/上年同期、合并/母公司）。
6. 回答用中文，简洁，先给结论再给依据。
7. 如果上下文前面给了【证据覆盖地图】，那是本次**按公司取证**的总账：地图上
   每一行的数字都取自方括号指向的那个片段。地图里标「缺据」或「部分覆盖」的公司，
   你必须在回答里点名说明是哪几家、缺的是哪一项，不能略过不提，也不能拿行业常识
   或别家的数据替它补上。
8. 做「全部/最高/都是负数/有哪几家」这类**全称判断或筛选**之前先看覆盖地图：
   只要有任意一家是「缺据」或「部分覆盖」，就不能对它下全称结论，要写成
   "在已取到数据的 N 家里……，另有 M 家缺据/缺项，无法判断"。
9. 引用数字时除了 [编号]，还要带上口径与期次，形如
   「28,416,674,541.77 元（2026 年上半年，合并口径）[8]」；
   地图上标「口径未标注」的，要写明"该表未标注合并/母公司口径，需核对原文"。
   另注意「营业收入」与「营业总收入」是两个不同的行项目，别混用。
10. 筛选题（"有哪几家……"）要把**每一家**你据以判断的数值连同 [编号] 列出来，
    一行一家；不允许用"其余各家均为……""详见下表"之类的话代替实际数值 ——
    凡是写不出数值的判断，就等于没有依据，要改成"该家数据未取到"。
    更不允许写出"见下表"却根本没有表。
11. **筛选题一律照覆盖地图数，不要自己回原文翻。** 地图每一行都带「同比 +x%」，
    已经**按同比降序排好**（正号都在最前面），末尾还有一行**「同比方向小计」**
    给出正几家、负几家 —— 那是给你对账用的数，你列的名单家数必须与它一致，
    对不上就回去重数。地图标「同比：未取到」的公司**方向不可判断**，
    不计入任何一侧，单独点名。
    —— 2026-10-12 实测：地图当时只摆了营业收入绝对值、没摆同比，模型只好自己回
    16 张表里翻，把全场增速最高的五粮液 +20.87% 整个漏掉，还写"其余 13 家均同比下降"。
12. 结论必须与你列出的数值**逐条对得上**：说"其余各家均……"之前，回看你列的那几行，
    确认没有例外；写"共 N 家"之前，照列表数一遍再写 N。
    **先数清楚再落笔** —— 答案里不允许出现"注意：上文漏列""修正："这类自我更正，
    那说明你是边写边数的。只呈现最终结论。
    —— 本轮实测出现过自相矛盾：15 家逐家列出，其中明明有 +1.47% 和 +20.87%，
    结论却写"其余 13 家均同比下降""共 2 家"。"""


# ---------------------------------------------------------------- 生成

_COVER_MARK = {"covered": "✅ 已覆盖", "partial": "⚠️ 部分覆盖", "missing": "❌ 缺据"}


def render_coverage(coverage, numbering: dict[int, int]) -> str:
    """把覆盖地图渲染成一段抬头，插在片段之前（模型先读地图、再读原文）。

    每行都是「公司 + 值 + 期次 + 口径 + 出处页 + 片段号」，紧挨着放在一起 ——
    老师那句"数值进了图谱只剩一个节点、不知道是谁的哪个口径哪一期"要的正是这个：
    每个数都能一眼回到那张表。地图不替代原文片段，它只是取证的结论与索引。
    """
    if not coverage:
        return ""
    fields = "、".join(sorted({e.field for e in coverage if e.field})) or "（未识别出可比对的字段）"
    n_c = sum(1 for e in coverage if e.status == "covered")
    n_p = sum(1 for e in coverage if e.status == "partial")
    n_m = sum(1 for e in coverage if e.status == "missing")
    # 有没有任何一家带同比 —— 有就在抬头点明，让模型知道筛选可以直接数这一列。
    n_yoy = sum(1 for e in coverage if e.yoy)
    lines = [
        f"【证据覆盖地图】本次按公司取证：字段「{fields}」，"
        f"应取证 {len(coverage)} 家，已覆盖 {n_c} 家、部分 {n_p} 家、缺据 {n_m} 家。"
        + (f"其中 {n_yoy} 家带同比。" if n_yoy else "")
    ]
    for e in coverage:
        seg = f"{_COVER_MARK.get(e.status, e.status)}　{e.company}（{e.code}）"
        if e.doc is not None:
            n = numbering.get(e.doc)
            # 同比必须跟字段值挨着摆，不能只给绝对值。2026-10-12 实测：
            # 地图只报「营业收入 28,416,674,541.77」时，问「哪几家同比正增长」的模型
            # 得自己回 16 张原文表里翻同比，五粮液的 +20.87% 又被漏掉（它的同比恰好挂在
            # 一行丢了行标签的数据上）。摆出来，这道题就退化成照着数。
            seg += (f"　{e.field} {e.value or '—'}"
                    + (f"　同比 {e.yoy}" if e.yoy else "　同比：未取到")
                    + f"　{e.period}"
                    f"　口径：{e.scope}　{e.source} p{e.page}"
                    + (f" · 片段[{n}]" if n else ""))
        if e.note:
            seg += f"　—— {e.note}"
        lines.append("  " + seg)
    if n_yoy:
        # 方向小计 —— 给模型的**对账数**，不是替它下结论：哪几家仍要它按公司名列。
        # 2026-10-12 实测：同比列摆上地图之后，模型仍写"其余 12 家同比均为负增长，
        # 无例外"，而五粮液那行明明白白写着 +20.87%。它要对 16 行同构数据做一次计数，
        # 做不对。摆一个它能对账的数，比在提示词里再写一遍"仔细数"管用。
        # 只对同比列做：值列本身带符号（问"净利润为负"那类），模型看值列就能看出正负。
        nums = [n for n in (retrieve._yoy_num(e.yoy) for e in coverage) if n is not None]
        pos, neg = sum(1 for n in nums if n > 0), sum(1 for n in nums if n < 0)
        lines.append(f"　—— 同比方向小计（对账用）：正 {pos} 家、负 {neg} 家、"
                     f"方向不可判断 {len(coverage) - pos - neg} 家。"
                     f"你列的名单家数必须与它一致，对不上就回去重数。")
    lines.append("取证说明：以上每个数都来自方括号指向的那个原文片段，未列出的块不作为数字依据。"
                 "标「同比：未取到」的公司，其增减方向**不可判断**，"
                 "既不能计入正增长，也不能计入下降，要单独点名说明。")
    return "\n".join(lines) + "\n\n"


def build_context(hits: list[dict], coverage=None) -> tuple[str, list[dict]]:
    """把召回块拼成带编号的上下文，同时产出给前端的引用列表。

    coverage 给了就先把覆盖地图放在所有片段之前；不给就与原来完全一样
    （单点事实题不走这条）。
    """
    blocks, cites = [], []
    numbering: dict[int, int] = {}
    for i, h in enumerate(hits, 1):
        if h.get("doc") is not None:
            numbering.setdefault(h["doc"], i)
    head = render_coverage(coverage, numbering) if coverage else ""
    for i, h in enumerate(hits, 1):
        c = h["chunk"]
        unit = f"（单位：{c['unit']}）" if c.get("unit") else "（⚠️ 该表未标注单位）"
        blocks.append(
            f"[{i}] 公司：{c['company']}　报告：{c['report']}　"
            f"章节：{c['section'] or '未定'}　第{c['page']}页\n"
            f"{c['text']}\n"
        )
        cites.append({
            "n": i, "company": c["company"], "report": c["report"],
            "section": c["section"], "page": c["page"],
            "type": c["type"], "unit": c.get("unit", ""),
            "preview": c["text"][:150],
        })
    return head + "\n".join(blocks), cites


def call_dashscope(question: str, context: str) -> str:
    key = os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        raise RuntimeError("NO_KEY")

    data = net.post_json(
        f"{C.DASHSCOPE_BASE_URL}/chat/completions",
        {
            "model": C.CHAT_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"财报片段：\n{context}\n\n问题：{question}"},
            ],
            "temperature": 0.1,
        },
        headers={"Authorization": f"Bearer {key}"},
        timeout=120,
        retries=2,
    )
    return data["choices"][0]["message"]["content"]


# ---------------------------------------------------------------- HTTP

class Handler(BaseHTTPRequestHandler):
    retriever = None

    def log_message(self, fmt, *a):        # 别把每个静态请求都刷到终端
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode(),
                   "application/json; charset=utf-8")

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            f = WEB / "index.html"
            if not f.exists():
                return self._send(404, b"web/index.html missing", "text/plain")
            return self._send(200, f.read_bytes(), "text/html; charset=utf-8")
        if path == "/api/health":
            return self._json(200, {
                "ready": True,
                "has_key": bool(os.environ.get("DASHSCOPE_API_KEY")),
                "n_chunks": len(self.retriever.chunks),
                "embed_backend": self.retriever.backend,
                # 跟着后端走，别硬写 C.EMBED_MODEL —— 索引是 API 建的时候，
                # 这里报本地模型名会让人以为查询编码器用错了。
                "model": (self.retriever.backend.split(":", 1)[1]
                          if self.retriever.backend.startswith("api:")
                          else C.EMBED_MODEL),
                "chat_model": C.CHAT_MODEL,
            })
        self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        if self.path.split("?")[0] != "/api/ask":
            return self._json(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            q = (req.get("question") or "").strip()
            if not q:
                return self._json(400, {"error": "问题为空"})

            r = self.retriever
            panorama = r.is_panorama(q)
            hits = r.search(q, k=C.TOP_K_FINAL)
            coverage = None
            if panorama:
                # 全景题：按公司取证 —— 每家一块，取不到就如实标 partial/missing，
                # 并且把这张覆盖地图交给模型（见 render_coverage）。
                coverage = r.panorama_evidence(q)
                seen = {h["doc"] for h in hits}
                hits += [h for h in r.panorama(q) if h["doc"] not in seen]

            context, cites = build_context(hits, coverage=coverage)
            try:
                answer = call_dashscope(q, context)
                mode = "llm"
            except RuntimeError:
                answer = ("（未设置 DASHSCOPE_API_KEY，只做检索不做生成。\n"
                          "设置后重启即可让大模型来组织答案。）")
                mode = "retrieval_only"

            return self._json(200, {
                "answer": answer, "cites": cites,
                "mode": mode, "panorama": panorama, "n_hits": len(hits),
                "coverage": ([{"company": e.company, "code": e.code,
                               "status": e.status, "field": e.field,
                               "value": e.value, "yoy": e.yoy, "period": e.period,
                               "scope": e.scope, "source": e.source,
                               "page": e.page, "note": e.note}
                              for e in coverage] if coverage else None),
            })
        except Exception as e:
            return self._json(500, {"error": f"{type(e).__name__}: {e}"})


def main() -> int:
    print("加载索引 …")
    from retrieve import Retriever
    try:
        Handler.retriever = Retriever()
    except RuntimeError as e:
        # 最常见的就是「索引是 API 建的，但这个终端没 export key」。
        # 抛栈太难读，直接摊开说。
        print()
        print("=" * 60)
        print(f"  ❌ 启动失败：{e}")
        print("=" * 60)
        return 1
    r = Handler.retriever
    has_key = bool(os.environ.get("DASHSCOPE_API_KEY"))
    print(f"  {len(r.chunks):,} 个块就绪")
    print(f"  查询编码器: {r.backend or '(旧索引未记录)'}")
    print(f"  DASHSCOPE_API_KEY: {'已设置 ✅' if has_key else '未设置（将只检索不生成）'}")

    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print()
    print("=" * 60)
    print(f"  打开 http://127.0.0.1:{PORT}")
    print("  （用 127.0.0.1，不要用 localhost —— 见文件头说明）")
    print("  Ctrl-C 停止")
    print("=" * 60)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
    return 0


if __name__ == "__main__":
    sys.exit(main())
