"""
向量后端抽象：本地跑 vs 走百炼 API。

两条路产出的向量**维度相同（1024）**，所以 retrieve.py 一行都不用改，
切换只影响建索引那一步。

| | 本地 Qwen3-Embedding-0.6B | 百炼 text-embedding-v4 |
|---|---|---|
| 耗时 | ~98 分钟（M1/MPS，实测） | ~10 分钟 |
| 内存 | 吃 8GB 统一内存，会 OOM | 不吃本机资源 |
| 花钱 | 0 | 约几元 |
| 课件一致 | 课件第 15 页就是这个模型 | 同属 Qwen3-Embedding 系列，API 版 |
| 断网可用 | 是 | 否 |

选哪个用环境变量控制，不用改代码：
    EMBED_BACKEND=api   （默认：有 key 就用 API）
    EMBED_BACKEND=local （强制本地）

密钥只从环境变量读，绝不写进文件 —— 这条没有例外。
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import config as C
import net

# text-embedding-v4 单次上限 10 条（v3 也是 10；qwen3.7 系列是 20）。
# 超了服务端直接报错，所以必须自己分批。
# 注意这是**每次请求**的上限，不是并发上限 —— 见 API_WORKERS。
API_BATCH = 10
API_MODEL = "text-embedding-v4"

# 同时在飞的请求数。
#
# 串行发是 77 分钟，不是注释里写的 10 分钟：单请求往返 1.6~2.3s，
# 1498 次请求光等往返就把时间吃光了，和算力无关。所以必须并发。
# 实测（专属 MaaS 端点，10 条/请求）：
#     串行    3.2 条/秒   → 77 分钟
#     并发 4  17.1 条/秒  → 15 分钟
#     并发 8  27.3 条/秒  →  9 分钟
# 端点没设并发限制，8 路也没触发 429（万一触发了，net.py 会按 Retry-After 退避重试，
# 只是变慢，不会算错）。想退回串行：API_WORKERS=1。
API_WORKERS = int(os.environ.get("API_WORKERS", "8"))


class LocalEmbedder:
    """本地 Qwen3-Embedding-0.6B（MPS 加速）。"""

    name = "local:Qwen3-Embedding-0.6B"

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        try:
            self.model = SentenceTransformer(str(C.MODEL_DIR), device=C.EMBED_DEVICE)
        except Exception as e:
            print(f"  {C.EMBED_DEVICE} 加载失败({e})，退回 cpu")
            self.model = SentenceTransformer(str(C.MODEL_DIR), device="cpu")

        # 必须显式设，见 config.EMBED_MAX_TOKENS —— 不设会慢 26 倍
        self.model.max_seq_length = C.EMBED_MAX_TOKENS
        self.batch = C.EMBED_BATCH

    def encode(self, texts: list[str]) -> np.ndarray:
        import torch

        while True:
            try:
                return self.model.encode(
                    texts, batch_size=self.batch,
                    normalize_embeddings=True, show_progress_bar=False)
            except RuntimeError as e:
                # 8GB 统一内存，长块凑一批很容易顶到上限。
                # 无人值守跑一个多小时，中途崩一次就全白干，所以降批重试。
                if "out of memory" in str(e).lower() and self.batch > 1:
                    self.batch = max(1, self.batch // 2)
                    torch.mps.empty_cache()
                    print(f"    ⚠️ MPS 显存不足，降到 batch={self.batch}", flush=True)
                    continue
                raise


class ApiEmbedder:
    """百炼 text-embedding-v4，1024 维。"""

    def __init__(self, model: str = API_MODEL, base_url: str | None = None):
        key = os.environ.get("DASHSCOPE_API_KEY")
        if not key:
            raise RuntimeError(
                "没找到 DASHSCOPE_API_KEY。\n"
                "请在终端里先执行：export DASHSCOPE_API_KEY=sk-xxxx\n"
                "注意：不要把它写进任何文件。")
        self.key = key
        self.model = model
        # 单次请求条数（服务端硬上限）和并发数是两回事，别混。
        self.req_size = API_BATCH
        self.workers = API_WORKERS
        # batch 是"一次 encode() 该喂多少条"，index_build 拿它当外循环步长。
        # 设成 单请求条数 × 并发数，内部才凑得出多个请求同时发 ——
        # 如果这里还是 10，index_build 每次只喂 10 条，并发等于没有。
        self.batch = self.req_size * self.workers
        # 每次 encode() 都新建线程池的话，全量要建 187 次、白起 1500 个线程。
        # 池子常驻，线程按需创建；进程退出时 ThreadPoolExecutor 注册的
        # atexit 钩子会唤醒并回收它们，不会挂住解释器。
        self._pool = ThreadPoolExecutor(max_workers=self.workers) if self.workers > 1 else None
        # base_url 可覆盖：百炼大陆站和国际站的 key 不通用，见 config.py 注释
        self.url = f"{base_url or C.DASHSCOPE_BASE_URL}/embeddings"
        self.name = f"api:{model}"

    def _post(self, batch: list[str]) -> list[list[float]]:
        data = net.post_json(
            self.url,
            {
                "model": self.model,
                "input": batch,
                "dimensions": C.EMBED_DIM,   # 显式要 1024，和课件对齐
                "encoding_format": "float",
            },
            headers={"Authorization": f"Bearer {self.key}"},
            # retries=5 配合 net.py 的网络层退避（2/4/8/16s），一次请求能扛住
            # 约 30 秒的网络抖动 —— 全量 190 次 encode() 调用，任何一次抖动
            # 打死整轮都要重跑，这个余量比多等几秒值。
            timeout=120, retries=5,
        )
        if "error" in data:
            raise RuntimeError(f"百炼返回错误：{data['error']}")
        # 服务端不保证顺序，按 index 排回来
        items = sorted(data["data"], key=lambda d: d["index"])
        return [it["embedding"] for it in items]

    def encode(self, texts: list[str]) -> np.ndarray:
        # 按单请求上限切，不是按 self.batch 切 —— batch 是调用方的喂食量
        chunks = [texts[i:i + self.req_size]
                  for i in range(0, len(texts), self.req_size)]

        if self._pool and len(chunks) > 1:
            # map 保序：结果下标和 chunks 一一对应，拼回来顺序天然是对的
            parts = list(self._pool.map(self._post, chunks))
        else:
            # 单条查询（retrieve.py 走这条）只有一个 chunk，起线程池是纯浪费
            parts = [self._post(c) for c in chunks]

        out: list[list[float]] = [v for part in parts for v in part]
        arr = np.asarray(out, dtype=np.float32)
        # 归一化，让 retrieve.py 的点积直接等于余弦相似度
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.maximum(norms, 1e-9)


def resolve_backend(backend: str | None = None) -> str:
    """决定用哪个后端。默认：有 key 走 API，没 key 退回本地。"""
    b = (backend or os.environ.get("EMBED_BACKEND") or "").lower()
    if not b:
        b = "api" if os.environ.get("DASHSCOPE_API_KEY") else "local"
    if b not in ("api", "local"):
        raise ValueError(f"未知的 EMBED_BACKEND: {b}（只支持 api / local）")
    return b


def backend_name(backend: str | None = None) -> str:
    """只返回后端标识名，**不加载模型** —— 指纹要用它，不能在算指纹时先花 3 秒加载。"""
    b = resolve_backend(backend)
    return f"api:{API_MODEL}" if b == "api" else "local:Qwen3-Embedding-0.6B"


def get_embedder(backend: str | None = None):
    b = resolve_backend(backend)
    return ApiEmbedder() if b == "api" else LocalEmbedder()


def smoke() -> None:
    """验证当前后端真的能出 1024 维向量，并排一下语义顺序。"""
    texts = [
        "贵州茅台2026年上半年直销渠道收入5,196,204.70万元。",
        "本公司主要从事白酒的生产与销售。",
        "董事会决议通过的本报告期利润分配预案。",
    ]
    e = get_embedder()
    print(f"== 后端 {e.name} ==")
    t0 = time.time()
    v = e.encode(texts)
    dt = time.time() - t0
    print(f"  形状 {v.shape}  用时 {dt:.1f}s")
    if v.shape[1] != C.EMBED_DIM:
        print(f"  ❌ 维度是 {v.shape[1]}，期望 {C.EMBED_DIM}")
        return
    print(f"  ✅ {C.EMBED_DIM} 维")

    q = e.encode(["茅台上半年直销渠道的收入是多少？"])[0]
    sims = v @ q
    order = np.argsort(-sims)
    for i in order:
        print(f"  {sims[i]:.4f}  {texts[i][:34]}…")
    print("  ✅ 语义排序正确" if order[0] == 0 else "  ⚠️ 排序异常")


if __name__ == "__main__":
    smoke()
