"""
全局配置：语料范围、路径、模型、切块参数。

密钥一律从环境变量读，绝不写进文件、绝不下发到浏览器。
"""

import os
from pathlib import Path

# ---------------------------------------------------------------- 路径

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
PDF_DIR = DATA / "pdf"              # 下载的原始财报
CHUNKS = DATA / "chunks.jsonl"      # 切好的块（带公司/章节/页码/单位）
VECTORS = DATA / "vectors.npy"      # 向量矩阵
BM25_PKL = DATA / "bm25.pkl"        # BM25 倒排索引
INDEX_META = DATA / "index_meta.json"   # 索引指纹，防止切块改了却没重建
MANIFEST = DATA / "manifest.json"   # 语料清单 + sha1，用于复现
CACHE = DATA / "embed_cache"        # 向量缓存（断点续跑）

MODEL_DIR = ROOT / "models" / "Qwen3-Embedding-0.6B"

for _d in (DATA, PDF_DIR, CACHE):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- 语料

# 白酒行业 A 股上市公司。
# 入选标准：以白酒生产销售为主营业务的 A 股公司；剔除主业含大量非白酒的
# （如顺鑫农业含地产）、以及 ST 股（如皇台酒业）—— 保证 16 家横向可比。
#
# 只写代码和简称；orgId 由 fetch.py 从巨潮的股票列表动态查，
# 不手写，因为它的格式不统一（有的是 gssh0600519，有的是 9900008689）。
COMPANIES = [
    ("600519", "贵州茅台"),
    ("000858", "五粮液"),
    ("000568", "泸州老窖"),
    ("600809", "山西汾酒"),
    ("002304", "洋河股份"),
    ("000596", "古井贡酒"),
    ("603369", "今世缘"),
    ("603589", "口子窖"),
    ("603198", "迎驾贡酒"),
    ("600559", "老白干酒"),
    ("600702", "舍得酒业"),
    ("600779", "水井坊"),
    ("600199", "金种子酒"),
    ("603919", "金徽酒"),
    ("600197", "伊力特"),
    ("002646", "天佑德酒"),
]

# 报告类型 → (巨潮 category 参数, 标题正则)
# 标题正则必须精确：否则会下到「摘要」或「英文版」——
# 实测五粮液的英文版发布时间(09-12)比中文版(08-29)更晚，
# 若按时间倒序取第一条就会下到英文版，后面全部解析作废。
REPORTS = {
    "2026半年报": {
        "category": "category_bndbg_szsh;",
        "date_range": "2026-01-01~2026-12-31",
        "title_re": r"^\S*2026年半年度报告$",
    },
    "2025年报": {
        "category": "category_ndbg_szsh;",
        "date_range": "2026-01-01~2026-12-31",   # 2025 年报在 2026 年披露
        "title_re": r"^\S*2025年年度报告$",
    },
}

# ---------------------------------------------------------------- 模型

EMBED_MODEL = "Qwen3-Embedding-0.6B"
EMBED_DIM = 1024                    # 课件里也是 1024 维
EMBED_DEVICE = "mps"                # M1 的 GPU；出问题代码里会退回 cpu

# 必须显式设这个。Qwen3-Embedding 的 max_seq_length 默认是 **32768**，
# 而我们的块中位只有 313 字符（≈280 token）。
# 不设的话，一个 batch 里只要混进一条长块，整批的 padding 和注意力开销
# 就按那条的长度放大 —— 实测 MPS 上掉到 0.1 条/秒，全量要 34 小时。
# 512 token 对检索足够，超长的块截断尾部（块本身有重叠，信息不会全丢）。
EMBED_MAX_TOKENS = 512

# 8GB 统一内存的硬约束：batch=16 × 512 token 过 MLP 中间层时激活值会冲到
# 8.74 GiB，直接 MPS OOM（实测）。8 能过。但代码里还有自动降批兜底，
# 因为无人值守跑一个半小时，中途爆一次就全白干。
EMBED_BATCH = 8

# 文本生成用阿里百炼（密钥只从环境变量读）
#
# 百炼有**两套互不相通的站点**，key 不通用，注册地决定了你能用哪个：
#   中国大陆站  https://dashscope.aliyuncs.com/compatible-mode/v1
#   国际站      https://dashscope-intl.aliyuncs.com/compatible-mode/v1
# 用错站点一律返回 401 invalid_api_key —— 那个报错很有误导性，
# 看着像 key 错了，其实是端点不对。所以做成可覆盖，别写死。
DASHSCOPE_BASE_URL = os.environ.get(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
CHAT_MODEL = os.environ.get("CHAT_MODEL", "qwen-plus")
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY")   # 可能为 None，用到时再报错

# ---------------------------------------------------------------- 切块

CHUNK_SIZE = 800        # 正文块目标字符数
CHUNK_OVERLAP = 100     # 正文块重叠
TABLE_GROUP_ROWS = 15   # 表格行组块：每块最多几行数据（表头每块都重复）
MIN_TABLE_ROWS = 2      # 少于这个行数的"表"多半是误检，丢弃
MIN_TABLE_COLS = 2
# 表格块字符数上限，硬闸门。正文有 split_text 兜底，表格没有 ——
# 拼多长全看 PDF 长什么样，而百炼的 input 上限是 33,000 token（约 5.5 万字符），
# 超了直接 400，一轮十几分钟的建索引白跑。
# 16000 是安全余量：现有最长的**正常**表格块是 8,116 字符，碰不到这个线，
# 只有误判块会被截。数字密集的表格每字符 token 数比中文高，别把余量压太紧。
MAX_TABLE_CHARS = 16000

# 表格误检闸门：块内**重复窗口占比**超过这个数，就判定这张表是误检，
# 不产出任何块，把内容交回正文路径。
#
# 形态：表格识别把一大段正文当成某个表头格（金徽酒 p161 那格 997 字符），
# table_parts 的前向填充又把它灌进所有空的表头格，于是每个数据行都
# 把同一段话重抄一遍 —— 一个块几万字符，里面同一页内容重复十几遍。
#
# 为什么用"重复窗口占比"而不是别的：
#   · 单元格长度不行 —— 全库最大单元格 997，渲染 52,823 字符；
#     而正常表的最大单元格也有 920。
#   · rendered/tot_cell 不行 —— 老白干 p180 中招了却只有 4.7。
#   · 200 字符的窗口能把"抄了一整段话"和"重复了一个列名"分开：
#     洋河 p36「董事出席董事会」那张表重复的是 14 字的列名，
#     200 字符的窗口跨过不同数值就不会重复 —— 那种表有用，不能拦。
# 实测（14,981 块）：正常块 <10%，超阈值的 22 块占 84,349 字符。
#
# 对所有表都判，不设长度下限：一开始想"小表重复也无所谓"留了个 1500 字符的
# 门槛，结果放过了 4 个 640~1,452 字符的同类块 —— 这种表的重复是有界的
# （渲染多长就重复多长），没必要放过。开销也不值得省：dup_ratio 是 O(n²)，
# 但全库最大的一张表渲染 52,823 字符，判一遍也就一秒量级。
TABLE_DUP_RATIO = 0.30

# ---------------------------------------------------------------- 检索

TOP_K_VECTOR = 20       # 向量召回
TOP_K_BM25 = 20         # BM25 召回
TOP_K_FINAL = 6         # 送给大模型的块数（对齐课件演示）
RRF_K = 60              # RRF 融合参数
MAX_PER_COMPANY = 3     # 同公司配额：防止一份大报告挤满 top-k

# ---------------------------------------------------------------- 网络

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
CNINFO_BASE = "http://www.cninfo.com.cn"
CNINFO_STATIC = "http://static.cninfo.com.cn"
DOWNLOAD_SLEEP = 1.0    # 下载之间的间隔，别把人家站点打疼
