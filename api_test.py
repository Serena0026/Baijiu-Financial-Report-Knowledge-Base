"""
API 连通性最小测试 —— 挂轻量，几秒钟，几分钱。

目的：把"接线对不对"和"建索引能不能跑"分开。
直接开 index_build.py 的话，如果 key 有问题、模型名写错、或者没开通服务，
你要等十分钟才知道，而且报错信息埋在日志里。这个脚本只发两个请求，
把错误立刻摊开。

它验证四件事：
  1. key 有效（不是 401）
  2. embedding 模型名对，且**返回 1024 维**（和课件一致，检索层才不用改）
  3. chat 模型名对，能出话
  4. 同一个句子两次向量一致 → 服务是确定性的，不是随机噪声

不会打印 key，也不写任何文件。

跑法：
    cd ~/projects/财报知识库
    export DASHSCOPE_API_KEY=sk-xxxx
    .venv/bin/python api_test.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

import config as C
import embed as embed_mod
import net

OK = "✅"
NO = "❌"
WARN = "⚠️ "


def pad(s: str, width: int) -> str:
    """按显示宽度补空格。中文字符占两列，str.format 的 :10s 按字符数算，
    直接用会把中文行错开。"""
    w = sum(2 if ord(c) > 0x2E80 else 1 for c in s)
    return s + " " * max(0, width - w)


def hr(title: str) -> None:
    print()
    print("=" * 62)
    print(title)
    print("=" * 62)


# 百炼两套站点，key 不通用。第一个是大陆站，第二个是国际站。
ENDPOINTS = {
    "中国大陆站": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "国际站": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
}


def test_key() -> bool:
    hr("1. 密钥")
    key = os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        print(f"{NO} 环境变量 DASHSCOPE_API_KEY 没设置。")
        print()
        print("   请在**你自己的终端**里执行（不要贴进对话，会进聊天记录）：")
        print("       export DASHSCOPE_API_KEY=sk-xxxx")
        print("   然后重跑本脚本。")
        print()
        print("   当前环境里能看到的候选变量：")
        for k in sorted(os.environ):
            if "KEY" in k.upper() or "DASHSCOPE" in k.upper() or "ALI" in k.upper():
                print(f"       {k}（已设置，长度 {len(os.environ[k])}）")
        return False

    # 只显示头尾，中间打码 —— 万一截图或贴日志也不会泄露
    shown = f"{key[:6]}…{key[-4:]}" if len(key) > 12 else "(太短，可疑)"
    print(f"{OK} 已读到密钥 {shown}（长度 {len(key)}）")

    if key.startswith("LTAI"):
        print(f"{NO} 这看起来是阿里云的 **AccessKey ID**，不是百炼的 API-KEY。")
        print("   百炼的 key 以 sk- 开头。AccessKey 是给 SDK/CLI 用的另一套凭证，")
        print("   拿它调百炼接口必然 401。")
        print("   去这里生成：https://bailian.console.aliyun.com/ → 右上角 API-KEY")
        return False
    if not key.startswith("sk-"):
        print(f"{WARN}密钥不是以 sk- 开头，百炼的通常是的，确认没复制错")
    if len(key) < 20:
        print(f"{WARN}密钥长度只有 {len(key)}，百炼的通常是 30+ 位，确认没复制漏")
    return True


def _probe_one(label: str, base: str, key: str, body: dict) -> str:
    """试一个端点，返回 'ok' / 'forbidden' / 'fail'。

    403 单独成一档而不是并进 'ok'：它只说明这个站点认这个 key，
    模型并没有开通，后面真去调照样失败 —— 所以它只能当兜底，不能当首选。
    """
    try:
        net.post_json(f"{base}/embeddings", body,
                      headers={"Authorization": f"Bearer {key}"},
                      timeout=30, retries=1)
        print(f"{OK} {pad(label,12)} {base}  → 可用")
        return "ok"
    except Exception as e:
        s = str(e)
        if "401" in s or "invalid_api_key" in s:
            print(f"{NO} {pad(label,12)} {base}  → 401 密钥对该站点无效")
            return "fail"
        if "403" in s or "AccessDenied" in s:
            print(f"{WARN}{pad(label,12)} {base}  → 403 密钥有效但模型没开通")
            return "forbidden"
        print(f"{WARN}{pad(label,12)} {base}  → {type(e).__name__}: {s[:80]}")
        return "fail"


def probe_endpoints() -> str | None:
    """确定该用哪个 base_url，并返回它。

    第一步必须是**探测当前配置的那个端点**：它才是 index_build.py 和
    server.py 真正会打的地方。两个公共站只是备用诊断，排在后面。

    （原先只探那两个公共站，再拿结果跟 config 比，不一致就警告
    "请 export 成公共站"。用专用 MaaS 端点时这是百分之百的误报 ——
    候选清单里根本没有它，比出来当然不一致。照着改会把额度和计费口径
    从专属工作区挪到公共站，纯属自找麻烦。实测踩过。）

    用错站点返回的是 401 invalid_api_key，那个报错看着完全像"key 错了"，
    会让人一直去重新生成 key，实际只要换个域名。
    """
    hr("2. 探测端点（当前配置优先，两个公共站仅作备用诊断）")
    key = os.environ["DASHSCOPE_API_KEY"]
    body = {"model": embed_mod.API_MODEL, "input": ["连通性测试"],
            "dimensions": C.EMBED_DIM, "encoding_format": "float"}

    cur = C.DASHSCOPE_BASE_URL
    cands = [("当前配置", cur)]
    cands += [(l, b) for l, b in ENDPOINTS.items() if b != cur]

    result = {b: _probe_one(l, b, key, body) for l, b in cands}

    print()
    # 当前配置能用 → 一切照旧，一个字都不用改。这是最常见的情况。
    if result.get(cur) == "ok":
        print(f"{OK} 当前配置的端点可用，后面就用它 —— 不用改任何环境变量。")
        others = [b for b, o in result.items() if b != cur and o == "ok"]
        if others:
            print(f"   （{others[0]} 也能用，但没必要换）")
        return cur

    # 当前配置不通，但某个公共站通 → 这才是真的该 export 的情况
    fallback = next((b for b, o in result.items() if b != cur and o == "ok"), None)
    if fallback:
        print(f"{NO} 当前配置的端点 {cur} 不通，但 {fallback} 可用。")
        print("   → 请这样覆盖（加到你的 export 那一行，起 server.py 也要带上）：")
        print(f"       export DASHSCOPE_BASE_URL={fallback}")
        return fallback

    # 认 key 但模型没开通：换站点也救不了，得去控制台开通
    if result.get(cur) == "forbidden":
        print(f"{WARN}当前配置的端点 {cur} 认这个 key，但模型没开通（403）。")
        print("   → 去百炼控制台「模型广场」开通 text-embedding-v4 / qwen-plus 再试。")
        return cur

    print(f"{NO} 没有任何可用端点：三个站点都拒绝了这个 key。")
    print("   → 那就不是区域问题，key 本身无效/过期/被删。")
    print("     去 https://bailian.console.aliyun.com/ 右上角 API-KEY 重新生成。")
    return None


def test_embedding(base_url: str | None = None) -> bool:
    hr("3. Embedding 接口")
    try:
        emb = embed_mod.ApiEmbedder(base_url=base_url)
    except Exception as e:
        print(f"{NO} 初始化失败：{e}")
        return False

    print(f"   模型 {emb.model}　端点 {emb.url}")

    # 同一个句子发两次，用来验证确定性
    probe = "贵州茅台2026年上半年直销渠道收入5,196,204.70万元。"
    other = "本公司主要从事白酒的生产与销售。"

    t0 = time.time()
    try:
        v = emb.encode([probe, other])
    except Exception as e:
        print(f"{NO} 调用失败：{type(e).__name__}: {e}")
        _diagnose(e)
        return False
    dt = time.time() - t0

    print(f"{OK} 调用成功　{len(v)} 条　{dt:.2f}s")

    if v.shape[1] != C.EMBED_DIM:
        print(f"{NO} 返回 {v.shape[1]} 维，期望 {C.EMBED_DIM} 维")
        print("   → 检索层是按 1024 维写的，维度不对必须改，别跳过")
        return False
    print(f"{OK} 维度 {v.shape[1]}，和课件一致")

    # 归一化后自己和自己应当是 1.0
    n0 = float(np.linalg.norm(v[0]))
    if abs(n0 - 1.0) > 1e-3:
        print(f"{WARN}向量未归一化（模长 {n0:.4f}）—— embed.py 本该归一化，检查一下")
    else:
        print(f"{OK} 向量已 L2 归一化（模长 {n0:.4f}），点积即余弦")

    # 两句不相关的文本不该给出相同的向量
    sim = float(v[0] @ v[1])
    if sim > 0.99:
        print(f"{NO} 两条不同文本的相似度 {sim:.4f} 接近 1 —— 接口可能没真在算")
        return False
    print(f"{OK} 不同文本相似度 {sim:.4f}（合理区间）")

    # 确定性：同文本再算一次应当完全一致
    v2 = emb.encode([probe])
    if not np.allclose(v[0], v2[0], atol=1e-5):
        print(f"{NO} 同一句子两次结果不一致 —— 非确定性，索引不可复现")
        return False
    print(f"{OK} 结果确定（同句两次完全一致）")

    print()
    # 单请求条数和并发数是两个数：请求数由前者定，耗时由后者定。
    # 只报请求数会让人以为 1498 次请求要等很久 —— 实际并发发出去快得多。
    reqs = 14981 / embed_mod.API_BATCH
    print(f"   全量预估：14,981 块 ÷ 每请求 {embed_mod.API_BATCH} 条 "
          f"= {reqs:.0f} 次请求")
    print(f"             并发 {embed_mod.API_WORKERS} 路发出 "
          f"（实测约 27 条/秒 → 全量 ~9 分钟）")
    print(f"             想串行：export API_WORKERS=1")
    return True


def test_chat(base_url: str | None = None) -> bool:
    hr("4. Chat 接口（生成答案用）")
    key = os.environ.get("DASHSCOPE_API_KEY")
    t0 = time.time()
    try:
        data = net.post_json(
            f"{base_url or C.DASHSCOPE_BASE_URL}/chat/completions",
            {
                "model": C.CHAT_MODEL,
                "messages": [
                    {"role": "system", "content": "你在做连通性测试，只回四个字。"},
                    {"role": "user", "content": "回复：连接正常"},
                ],
                "temperature": 0.0,
                "max_tokens": 16,
            },
            headers={"Authorization": f"Bearer {key}"},
            timeout=60, retries=1,
        )
    except Exception as e:
        print(f"{NO} 调用失败：{type(e).__name__}: {e}")
        _diagnose(e)
        return False

    dt = time.time() - t0
    msg = data["choices"][0]["message"]["content"].strip()
    used = data.get("usage", {}).get("total_tokens", "?")
    print(f"{OK} {C.CHAT_MODEL} 回复：{msg!r}　{dt:.2f}s　用了 {used} token")
    return True


def _diagnose(e: Exception) -> None:
    """把常见报错翻译成人话，省得对着 HTTP 状态码猜。"""
    s = str(e)
    print()
    if "401" in s or "InvalidApiKey" in s or "invalid_api_key" in s:
        print("   → 密钥无效或已过期。去百炼控制台重新生成一个。")
    elif "403" in s or "AccessDenied" in s:
        print("   → 密钥有效但没权限。多半是**没开通这个模型**，")
        print("     去百炼控制台「模型广场」把 text-embedding-v4 / qwen-plus 开通。")
    elif "429" in s or "Throttling" in s or "quota" in s.lower():
        print("   → 触发限流或用量超了。稍等再试，或去控制台看额度。")
    elif "404" in s or "ModelNotFound" in s or "model_not_found" in s:
        print("   → 模型名不对，或该区域没有这个模型。")
    elif "Arrearage" in s or "欠费" in s:
        print("   → 账户欠费，去充值。")
    else:
        print(f"   → 原始信息：{s[:300]}")


def main() -> int:
    print("百炼 API 连通性测试")
    print(f"端点 {C.DASHSCOPE_BASE_URL}")

    if not test_key():
        return 1
    base = probe_endpoints()
    ok_emb = test_embedding(base)
    ok_chat = test_chat(base)

    hr("结论")
    if ok_emb and ok_chat:
        print(f"{OK} 两个接口都通，可以开始建索引了。")
        print()
        # 只在探测结果确实和当前配置不一致时才提 export —— 否则又是一句
        # 看着像"你配错了"的废话，用户会去改一个本来没问题的变量
        if base and base != C.DASHSCOPE_BASE_URL:
            print("   上面用的端点和当前配置不一致，先把它固定下来：")
            print(f"       export DASHSCOPE_BASE_URL={base}")
            print("   （建议写进你的 export 那一行，起 server.py 时也要带上）")
            print()
        print("   然后建索引：")
        print("       .venv/bin/python index_build.py")
        print()
        print("   （约 10 分钟，断了直接重跑，会跳过已算过的块）")
        return 0

    if ok_emb and not ok_chat:
        print(f"{WARN}embedding 通了，chat 没通。")
        print("   → 建索引可以先跑（只需要 embedding）。")
        print("     但问答会退化成「只检索不生成」，因为生成要靠 chat 模型。")
        return 1

    print(f"{NO} embedding 没通，先别建索引 —— 跑了也是白等。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
