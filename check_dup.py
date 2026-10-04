"""
体检脚本：数一数有多少块**内部**在重复同一段内容。

背景：表格识别偶尔会把一大段正文当成某个表头格，table_parts 的前向填充
再把它灌进所有空的表头格，于是每个数据行都重抄一遍这段文字 ——
一个块几万字符，里面同一页内容重复十几遍。这种块对检索只有坏处：
BM25 词频被同一段话顶高，容易误召回，还挤掉正常的块。

判定办法：把块切成 200 字符的窗口（步长 40），看有多少比例的窗口在块内
还出现过第二次。
  · 正常块   → 0 附近
  · 重复块   → 30% 以上（实测重灾区 40%~82%）

为什么要用窗口计数，而不是压缩比：压缩比对「数字很多的长表」也会报低
（重复的数字串压得很狠），会误伤。窗口计数只认"同一段文字出现两次"，
这正是要抓的东西。

用法：
    .venv/bin/python check_dup.py        # 汇总
    .venv/bin/python check_dup.py -v     # 列出每个重复块
"""

from __future__ import annotations

import json
import sys

import config as C

WIN = 200
STRIDE = 40
THRESHOLD = 0.30


def dup_ratio(text: str) -> float:
    """块内重复窗口的占比。"""
    wins = [text[i:i + WIN] for i in range(0, max(len(text) - WIN + 1, 1), STRIDE)]
    wins = [w for w in wins if len(w) == WIN]
    if not wins:
        return 0.0
    return sum(1 for w in wins if text.count(w) > 1) / len(wins)


def main() -> int:
    verbose = "-v" in sys.argv
    if not C.CHUNKS.exists():
        print(f"没有 {C.CHUNKS}，先跑 parse.py")
        return 1

    chunks = [json.loads(l) for l in C.CHUNKS.open(encoding="utf-8")]
    scored = [(dup_ratio(c["text"]), c) for c in chunks]
    dup = [(r, c) for r, c in scored if r > THRESHOLD]

    by_co: dict[str, int] = {}
    for _, c in dup:
        by_co[c["company"]] = by_co.get(c["company"], 0) + 1

    print(f"总块数 {len(chunks):,}")
    print(f"内容重复块 {len(dup)} 个（重复窗口占比 > {THRESHOLD:.0%}）")
    if dup:
        print(f"这些块合计 {sum(len(c['text']) for _, c in dup):,} 字符")
        print()
        for co, n in sorted(by_co.items(), key=lambda kv: -kv[1]):
            print(f"  {co:8s} {n:>3} 块")
    else:
        print("✅ 没有块在内部重复内容")
    if verbose and dup:
        print()
        for r, c in sorted(dup, key=lambda x: -x[0]):
            print(f"  {r:5.1%}  {len(c['text']):>7,}字符  {c['company']:6s} "
                  f"{c['report']} {c['type']:5s} p{c['page']:<4} | "
                  f"{c['text'][:58].replace(chr(10), ' ')!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
