"""
第一步：从巨潮资讯网拉白酒 16 家的财报 PDF。

巨潮资讯网（cninfo.com.cn）是证监会指定的 A 股信息披露平台，法定披露以它为准。
作业说"从交易所网站下载"——对 A 股而言这就是官方渠道。

它的接口有三个坑，都已在下面处理：

坑一：只传股票代码查不到
    stock="600519"            → 0 条
    stock="600519,gssh0600519" → 2 条
    而且 orgId 格式不统一：多数是 gssh/gssz + 代码，但有的是 9900008689 这种。
    所以必须先从股票列表查 orgId，不能手写。

坑二：标题必须精确过滤，否则会下到英文版
    实测五粮液返回：
        2026年半年度报告（英文版）  2026-09-12   ← 时间最新！
        2026年半年度报告            2026-08-29
        2026年半年度报告摘要        2026-08-29
    按时间倒序取第一条会拿到英文版，后面全部解析作废。

坑三：adjunctUrl 没有前导斜杠，且主机是 HTTP 明文。

跑法：
    cd ~/projects/财报知识库
    .venv/bin/python fetch.py
"""

from __future__ import annotations

import hashlib
import json
import re
import time

import config as C
import net

STOCK_LIST_URL = f"{C.CNINFO_BASE}/new/data/szse_stock.json"
QUERY_URL = f"{C.CNINFO_BASE}/new/hisAnnouncement/query"
REFERER = f"{C.CNINFO_BASE}/new/commonUrl?url=disclosure/list/notice"


def load_org_ids() -> dict[str, dict]:
    """拉全市场股票列表，建 code → {orgId, name} 映射。"""
    print("拉取巨潮股票列表…")
    data = net.get_json(STOCK_LIST_URL, timeout=60)
    listing = data.get("stockList") or []
    print(f"  共 {len(listing):,} 只股票")
    return {s["code"]: s for s in listing}


def query_announcements(code: str, org_id: str, spec: dict) -> list[dict]:
    """按报告类型查公告。"""
    return net.post_form_json(
        QUERY_URL,
        {
            "pageNum": "1",
            "pageSize": "30",
            "column": "szse",              # 这个值对沪深两市都管用
            "tabName": "fulltext",
            "stock": f"{code},{org_id}",
            "category": spec["category"],
            "seDate": spec["date_range"],
            "isHLtitle": "true",
        },
        referer=REFERER,
        timeout=30,
    ).get("announcements") or []


def pick_full_report(anns: list[dict], spec: dict) -> dict | None:
    """从公告里挑出中文全文正报，排除摘要/英文版/更新后。"""
    pat = re.compile(spec["title_re"])
    for a in anns:
        title = (a.get("announcementTitle") or "").strip()
        # 正则已经排除了「摘要」「英文版」；这里再兜一层，防止正则被改坏
        if "摘要" in title or "英文" in title or "English" in title:
            continue
        if pat.match(title):
            return a
    return None


def sha1_of(path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    stocks = load_org_ids()
    manifest: list[dict] = []
    missing: list[str] = []

    for code, short in C.COMPANIES:
        info = stocks.get(code)
        if not info:
            print(f"  {short:6s} ❌ 不在股票列表里")
            missing.append(short)
            continue
        org_id = info["orgId"]

        for report_name, spec in C.REPORTS.items():
            try:
                anns = query_announcements(code, org_id, spec)
            except Exception as e:
                print(f"  {short:6s} {report_name:9s} ❌ 查询失败: {e}")
                missing.append(f"{short}/{report_name}")
                continue

            hit = pick_full_report(anns, spec)
            if not hit:
                print(f"  {short:6s} {report_name:9s} ❌ 无中文全文（返回 {len(anns)} 条）")
                missing.append(f"{short}/{report_name}")
                time.sleep(C.DOWNLOAD_SLEEP)
                continue

            url = f"{C.CNINFO_STATIC}/" + hit["adjunctUrl"].lstrip("/")
            dest = C.PDF_DIR / f"{code}_{short}_{report_name}.pdf"
            try:
                net.download(url, dest, timeout=300)
            except Exception as e:
                print(f"  {short:6s} {report_name:9s} ❌ 下载失败: {e}")
                missing.append(f"{short}/{report_name}")
                continue

            size_mb = dest.stat().st_size / 1048576
            manifest.append({
                "code": code,
                "orgId": org_id,
                "company": short,
                "report": report_name,
                "title": hit["announcementTitle"].strip(),
                "date": hit.get("announcementTime"),
                "url": url,
                "file": dest.name,
                "bytes": dest.stat().st_size,
                "sha1": sha1_of(dest),
            })
            print(f"  {short:6s} {report_name:9s} ✅ {dest.name}  ({size_mb:.1f} MB)")
            time.sleep(C.DOWNLOAD_SLEEP)

    C.MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print()
    print("=" * 60)
    print(f"成功 {len(manifest)} 份 / 目标 {len(C.COMPANIES) * len(C.REPORTS)} 份")
    if missing:
        print(f"缺失：{missing}")
    print(f"清单已写入 {C.MANIFEST}")
    print("=" * 60)
    return 0 if manifest else 1


if __name__ == "__main__":
    raise SystemExit(main())
