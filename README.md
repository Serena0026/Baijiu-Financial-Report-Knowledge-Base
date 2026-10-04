# 白酒财报问答知识库

16 家 A 股白酒上市公司 × (2026 半年报 + 2025 年报) = 32 份财报全文，
做向量 + BM25 混合检索的问答系统，答案带出处（公司 / 报告 / 章节 / 页码 / 单位）。

数据来源：**巨潮资讯网**（cninfo.com.cn），财报全文 PDF。

---

## 跑起来

```bash
cd ~/projects/财报知识库

# 密钥只从环境变量读，绝不写进任何文件
export DASHSCOPE_API_KEY=sk-xxxx
export DASHSCOPE_BASE_URL=https://<你的工作区>.maas.aliyuncs.com/compatible-mode/v1

.venv/bin/python server.py     # 打开 http://127.0.0.1:8000
```

**必须用 `127.0.0.1` 而不是 `localhost`** —— 这台机器上 localhost 有时先解析到 `::1`，
而服务只监听 IPv4，浏览器报「拒绝连接」，但 `curl 127.0.0.1` 又是通的，很容易误判。

没设 key 也能跑：会退化成「只检索不生成」，把召回的原文和出处列出来。

---

## 完整流水线

仓库里**不含**原始 PDF、向量和模型（共 2.3G），都要靠下面几步重新生成。

```bash
# 1. 下载 32 份财报（约 51M，走巨潮 API）
.venv/bin/python fetch.py

# 2. 解析 + 切块 → data/chunks.jsonl（正文按语义分块，表格按行列还原）
.venv/bin/python parse.py

# 3. 建索引 → data/vectors.npy + data/bm25.pkl（走 API 约 22 分钟）
.venv/bin/python index_build.py

# 4. 起服务
.venv/bin/python server.py
```

`data/manifest.json` 里记了每份 PDF 的 sha1，重下后可比对是否拿到同一批文件。

### 评测

```bash
.venv/bin/python eval.py              # 10 道题全跑（召回 + 生成）
.venv/bin/python eval.py 3 7          # 只跑第 3、7 题
.venv/bin/python eval.py --no-llm     # 只检索不生成
.venv/bin/python eval.py --report-only  # 按上次结果重写报告，不调大模型
```

产出 `data/eval_report.md`（人读，含逐题召回块清单 + 回答 + 人工复核）
和 `data/eval_raw.json`（机器可读）。

---

## 文件

| 文件 | 作用 |
|---|---|
| `config.py` | 语料范围、路径、模型、切块与检索参数 |
| `net.py` | 带重试和退避的 HTTP（百炼、巨潮共用） |
| `fetch.py` | 从巨潮查 orgId、按标题正则下载财报 |
| `parse.py` | PDF → 文字/表格块，带公司/章节/页码/单位 |
| `embed.py` | 向量后端（百炼 API / 本地 MPS），带断点续跑缓存 |
| `bm25.py` | BM25 倒排索引 |
| `index_build.py` | 向量矩阵 + BM25，指纹校验防新旧混杂 |
| `retrieve.py` | 向量 + BM25 → RRF 融合 → 同公司配额；全景题锚点通道 |
| `server.py` | 本地 HTTP 服务 + 调大模型生成 |
| `eval.py` | 10 道题的评测脚本与题库 |
| `make_doc.py` | 把评测结果导出成 Word（`白酒财报问答_10道评测题.docx`） |
| `web/index.html` | 问答页面（无密钥，密钥只在服务端） |
| `check_dup.py` | 表格重复块的健康检查 |
| `结论.md` | 一页结论：哪类题答得好、哪类翻车、为什么 |

---

## 几个不显然的设计决定

**为什么向量和 BM25 都要。** 向量对精确串和数字迟钝（问「600779 的营收」，它召回的是
「意思相近」的其他白酒公司，因为句子长得几乎一样）；BM25 对字面不重合的表达无效
（问「直接销售取得的收入」，原文写的是「直销渠道收入」）。两条腿缺一不可。

**RRF 而不是加权求和。** 余弦相似度 0.7 和 BM25 得分 12.3 量纲不可比，硬加权才是错的。
RRF 只看排名：`score(d) = Σ 1/(K + rank_i(d))`，K=60。

**同公司配额 `MAX_PER_COMPANY=3`。** 一份 116 页年报能产出 500 多个块，16 家竞争 top-6 时
很容易 6 个块全来自同一家，「对比各家」类的问题就废了。

**全景题走旁路通道。** 纯 top-k 架构上答不了「16 家谁净利润最高」—— 它最多召回 6 个块，
覆盖不到 16 家公司。所以按公司身份把各家的「主要会计数据」块直接捞出来。
代价见 `结论.md`：**锚点坏一家，那一家就整个消失且不报警。**

**查询编码器必须和建索引时一致。** 本地 Qwen3 和百炼 text-embedding-v4 是两个不同的
向量空间，拿 A 的查询向量点乘 B 的索引向量**不抛异常**，只是结果错得看不出来。
所以后端名从 `index_meta.json` 读，不靠环境变量猜；取不到就报错，不静默退回本地。

**密钥只从环境变量读。** 不写进文件、不下发到浏览器 —— 这条没有例外。
`web/index.html` 里没有任何密钥，浏览器只能调本机的 `/api/ask`。
