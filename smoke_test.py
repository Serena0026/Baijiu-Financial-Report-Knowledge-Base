"""
冒烟测试：验证本地 Qwen3-Embedding-0.6B 能跑通，并量出真实吞吐。

这一步是环境搭建的关键闸门。它一次性验证：
  1. Python 3.12 解释器对不对
  2. torch 装没装好、MPS 能不能用
  3. 模型权重是不是完整的
  4. 向量维度是否和课件一致（1024 维）
  5. 吞吐多少 —— 用来推算全量切块要跑多久

跑法：
    cd ~/projects/财报知识库
    .venv/bin/python smoke_test.py
"""

import os
import sys
import time
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "models" / "Qwen3-Embedding-0.6B"

# 用真实财报里会出现的句子做测试，而不是 "hello world" —— 这样长度和字符分布才接近实际
SAMPLES = [
    "贵州茅台2026年上半年直销渠道收入5,196,204.70万元，批发代理渠道收入3,869,657.74万元。",
    "报告期内，公司通过“i茅台”数字营销平台实现酒类不含税收入4,026,356.42万元。",
    "本期国内经销商增减变动主要为系列酒经销商。",
    "经营活动产生的现金流量净额同比增加438.84%，主要是公司吸收集团公司成员单位存款增加。",
    "公司: 五粮液 | 章节: 第三节 管理层讨论与分析 | 页码: 12 | 项目: 营业收入 | 本期发生额: 52,071,594,062.35 万元",
]


def main() -> int:
    print("=" * 64)
    print("环境自检")
    print("=" * 64)
    print(f"Python      : {sys.version.split()[0]}")
    print(f"解释器      : {sys.executable}")
    if not sys.version.startswith("3.12"):
        print("⚠️  不是 3.12，后续 torch 可能有问题")

    import torch
    import sentence_transformers

    print(f"torch       : {torch.__version__}")
    print(f"MPS 可用    : {torch.backends.mps.is_available()}")
    print(f"torch 线程  : {torch.get_num_threads()}")

    if not MODEL_DIR.exists():
        print(f"\n❌ 模型目录不存在: {MODEL_DIR}")
        print("   先把权重下下来再跑这个脚本。")
        return 1

    weights = list(MODEL_DIR.glob("*.safetensors"))
    if not weights:
        print(f"\n❌ {MODEL_DIR} 里没有 .safetensors 权重文件，下载可能没完成。")
        return 1
    size_gb = sum(w.stat().st_size for w in weights) / 1024**3
    print(f"权重文件    : {[w.name for w in weights]}  ({size_gb:.2f} GB)")

    print()
    print("=" * 64)
    print("加载模型")
    print("=" * 64)
    t0 = time.time()
    from sentence_transformers import SentenceTransformer

    # device="mps" 用 M1 的 GPU；真机上一旦出问题就退回 cpu
    try:
        model = SentenceTransformer(str(MODEL_DIR), device="mps")
        device_used = "mps"
    except Exception as e:
        print(f"  MPS 加载失败({e})，退回 CPU")
        model = SentenceTransformer(str(MODEL_DIR), device="cpu")
        device_used = "cpu"
    print(f"加载完成    : {time.time() - t0:.1f}s (device={device_used})")

    print()
    print("=" * 64)
    print("维度检查")
    print("=" * 64)
    vec = model.encode([SAMPLES[0]], normalize_embeddings=True)
    print(f"单句向量形状: {vec.shape}")
    if vec.shape[-1] != 1024:
        print(f"⚠️  维度是 {vec.shape[-1]}，课件里是 1024 —— 不一定是错，但要留意")
    else:
        print("✅ 1024 维，和课件一致")

    print()
    print("=" * 64)
    print("吞吐测试 —— 决定全量建索引要跑多久")
    print("=" * 64)

    # 先用一小批热身，避免把模型加载/首次推理的编译开销算进吞吐
    model.encode(SAMPLES[:2], normalize_embeddings=True)

    BATCH = 32
    N = 160
    payload = (SAMPLES * (N // len(SAMPLES) + 1))[:N]
    t0 = time.time()
    model.encode(payload, batch_size=BATCH, normalize_embeddings=True, show_progress_bar=False)
    dt = time.time() - t0
    per_sec = N / dt
    print(f"实测      : {N} 条 / {dt:.1f}s  =  {per_sec:.1f} 条/秒")

    print()
    print("全量推算（按作业的 32 份财报约 10,000 个块）:")
    for n_blocks in (5_000, 10_000):
        mins = n_blocks / per_sec / 60
        print(f"  {n_blocks:>6,} 块 → {mins:5.1f} 分钟")

    print()
    print("=" * 64)
    print("相似度合理性检查")
    print("=" * 64)
    # 语义近的应该分数高，这是模型真的在工作（而不是输出常数）的证据
    q = "茅台上半年直销渠道的收入是多少？"
    docs = [
        "贵州茅台2026年上半年直销渠道收入5,196,204.70万元，批发代理渠道收入3,869,657.74万元。",
        "本公司主要从事白酒的生产与销售，产品包括茅台酒及系列酒。",
        "董事会决议通过的本报告期利润分配预案。",
    ]
    import numpy as np

    embs = model.encode([q] + docs, normalize_embeddings=True)
    sims = embs[1:] @ embs[0]
    for d, s in sorted(zip(docs, sims), key=lambda x: -x[1]):
        print(f"  {s:.4f}  {d[:38]}...")
    if sims[0] > sims[1] and sims[0] > sims[2]:
        print("\n✅ 语义排序正确：含答案的块得分最高")
    else:
        print("\n⚠️  排序异常，模型可能没正常工作")

    print()
    print("=" * 64)
    print("✅ 冒烟测试通过 —— 环境可用，可以开始建作业项目了")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
