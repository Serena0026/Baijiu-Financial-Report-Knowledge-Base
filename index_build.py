"""
第三步：建索引 —— 把 15,000 个块变成向量矩阵 + BM25 倒排。

耗时取决于向量后端（见 embed.py）：走百炼 API 约 10 分钟，本地 MPS 约 98 分钟。
中途挂掉（内存、误触、合盖）很常见，所以必须支持断点续跑：
每跑完一批就把向量落盘，重跑时跳过已经算过的块。

指纹机制：index_meta.json 里存 chunks.jsonl 的 sha1。
只要切块结果变了（改了 CHUNK_SIZE、修了 parse 逻辑），指纹就对不上，
脚本会拒绝复用旧索引 —— 否则你会拿新旧混杂的索引检索，而且完全看不出问题。
那种 bug 最难查，因为代码看着是对的。

跑法：
    cd ~/projects/财报知识库
    .venv/bin/python index_build.py
"""

from __future__ import annotations

import hashlib
import json
import time

import numpy as np

import bm25
import config as C
import embed as embed_mod


def fingerprint(model_name: str) -> str:
    h = hashlib.sha1()
    with open(C.CHUNKS, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    h.update(model_name.encode())
    # EMBED_MAX_TOKENS 也要进指纹：改了截断长度，向量就变了，旧缓存不能复用
    h.update(f"{C.CHUNK_SIZE}|{C.CHUNK_OVERLAP}|{C.TABLE_GROUP_ROWS}"
             f"|{C.MAX_TABLE_CHARS}|{C.EMBED_MAX_TOKENS}"
             f"|{C.TABLE_DUP_RATIO}".encode())
    return h.hexdigest()[:16]


def load_cache(path) -> tuple[list[str], np.ndarray | None]:
    if not path.exists():
        return [], None
    z = np.load(path, allow_pickle=False)
    return list(z["ids"]), z["vecs"]


def save_cache(path, ids: list[str], vecs: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, ids=np.array(ids), vecs=vecs)
    tmp.replace(path)


def main() -> int:
    if not C.CHUNKS.exists():
        print("没找到 chunks.jsonl，先跑 parse.py")
        return 1

    chunks = [json.loads(l) for l in C.CHUNKS.open(encoding="utf-8")]
    print(f"读了 {len(chunks):,} 个块")

    # 后端名进指纹：本地向量和 API 向量不可混用，换了后端必须重算，
    # 否则会拿一半本地一半 API 的向量做检索，而且完全看不出问题。
    backend = embed_mod.backend_name()
    fp = fingerprint(backend)
    cache_path = C.CACHE / f"emb_{fp}.npz"
    print(f"向量后端 {backend}　指纹 {fp}")

    # 指纹一致且向量已就位 → 直接跳过（除非你删了 vectors.npy）
    if C.INDEX_META.exists() and C.VECTORS.exists():
        meta = json.loads(C.INDEX_META.read_text(encoding="utf-8"))
        if meta.get("fingerprint") == fp and meta.get("n_chunks") == len(chunks):
            print(f"索引已是最新（指纹 {fp}，{meta['n_chunks']:,} 块），跳过。")
            print("  想强制重建：删掉 data/vectors.npy 和 data/index_meta.json")
            return 0

    # ---------------------------------------------------------- 向量
    ids, vecs = load_cache(cache_path)
    done = set(ids)
    todo = [c for c in chunks if c["id"] not in done]
    print(f"向量缓存命中 {len(done):,}，待算 {len(todo):,}")

    if todo:
        emb = embed_mod.get_embedder()
        print(f"向量后端 {emb.name}")
        B = getattr(emb, "batch", C.EMBED_BATCH)

        # 按长度排序再分批：批内的 padding 按最长那条算，长度相近的凑一起
        # 能省掉大量无效计算。最后会按 id 重排回来，不影响结果。
        todo.sort(key=lambda c: len(c["text"]))

        t0 = time.time()
        for i in range(0, len(todo), B):
            batch = todo[i:i + B]
            new = emb.encode([c["text"] for c in batch])
            vecs = new if vecs is None else np.vstack([vecs, new])
            ids.extend(c["id"] for c in batch)
            # API 后端每批都要发请求，落盘太勤反而拖慢；本地每 20 批落一次
            if i % (B * 20) == 0:
                save_cache(cache_path, ids, vecs)
                done_n = i + len(batch)
                el = max(time.time() - t0, 1e-6)
                rate = done_n / el
                print(f"  {done_n:>6,}/{len(todo):,}  {rate:5.1f} 条/秒  "
                      f"剩余 {(len(todo)-done_n)/rate/60:4.1f} 分钟", flush=True)
        save_cache(cache_path, ids, vecs)
        print(f"向量算完，用时 {(time.time()-t0)/60:.1f} 分钟")

    # 按 chunks 的顺序重排（缓存里的顺序可能因续跑而不同）
    pos = {cid: i for i, cid in enumerate(ids)}
    order = np.array([pos[c["id"]] for c in chunks])
    mat = vecs[order].astype(np.float32)
    np.save(C.VECTORS, mat)
    print(f"向量矩阵 {mat.shape} → {C.VECTORS}")

    # ---------------------------------------------------------- BM25
    print("建 BM25 索引 …")
    t0 = time.time()
    idx = bm25.BM25.build([c["text"] for c in chunks])
    idx.save(C.BM25_PKL)
    print(f"  {len(idx.vocab):,} 个词条，用时 {time.time()-t0:.1f}s")

    # model/device 必须跟着 backend 走，不能无条件写 config 里那两个本地模型字段。
    # 否则走 API 建出来的索引，元数据却写着 Qwen3-Embedding-0.6B / mps，
    # 看着像本地建的 —— 判断该用哪个编码器以 backend 为准，但元数据自相矛盾
    # 会在排查时把人带偏（"明明是 API 建的，怎么写着本地模型"）。
    if backend.startswith("api:"):
        model_name, device_name = backend.split(":", 1)[1], "api"
    else:
        model_name, device_name = C.EMBED_MODEL, C.EMBED_DEVICE

    C.INDEX_META.write_text(json.dumps({
        "fingerprint": fp,
        "n_chunks": len(chunks),
        "dim": int(mat.shape[1]),
        # backend 是**建索引时真正用的**那个（api:text-embedding-v4 或
        # local:Qwen3-Embedding-0.6B）。retrieve.py 靠它决定查询该用哪个
        # 编码器 —— 两个向量空间不可互换，混用不报错但结果全错。
        "backend": backend,
        "model": model_name,
        "device": device_name,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("=" * 60)
    print(f"✅ 索引建好：{mat.shape[0]:,} 块 × {mat.shape[1]} 维")
    print(f"   指纹 {fp}   {C.VECTORS.name} / {C.BM25_PKL.name}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
