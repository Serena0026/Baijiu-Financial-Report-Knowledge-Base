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
6. 回答用中文，简洁，先给结论再给依据。"""


# ---------------------------------------------------------------- 生成

def build_context(hits: list[dict]) -> tuple[str, list[dict]]:
    """把召回块拼成带编号的上下文，同时产出给前端的引用列表。"""
    blocks, cites = [], []
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
    return "\n".join(blocks), cites


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
            if panorama:
                # 全景题：把 16 家的锚点块也塞进上下文，否则覆盖不到那么宽
                seen = {h["doc"] for h in hits}
                hits += [h for h in r.panorama(q) if h["doc"] not in seen]

            context, cites = build_context(hits)
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
