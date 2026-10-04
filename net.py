"""
统一 HTTP 层 —— 这个文件解决两个会让人怀疑人生的坑。

坑一：死代理
    这台机器的 macOS 系统代理指向 127.0.0.1:12450，但那个端口没有任何进程监听
    （WPS 云同步留下的残留）。Python 的 urllib 在 macOS 上会通过 _scproxy 读系统
    代理设置，于是所有请求都 "Connection refused"。而 curl 不读系统代理，所以
    curl 一切正常 —— 这个差异极容易让人误判成"网络不通"。
    解法：ProxyHandler({}) 显式清空代理。

坑二：证书
    python.org 版 Python 不使用 macOS 钥匙串。一旦禁用代理走 HTTPS，
    就会 CERTIFICATE_VERIFY_FAILED（无法获取本地颁发者证书）。
    巨潮是 HTTP 明文，所以这个坑在抓 PDF 时不会暴露 —— 等调 DashScope（HTTPS）
    才突然炸出来。解法：显式挂 certifi 的 CA bundle。

所有对外请求都必须走这里，不要在别处直接调 urllib / requests。
"""

from __future__ import annotations

import json
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import certifi

from config import USER_AGENT

# ---------------------------------------------------------------- opener

def _build_opener() -> urllib.request.OpenerDirector:
    ctx = ssl.create_default_context(cafile=certifi.where())
    return urllib.request.build_opener(
        urllib.request.ProxyHandler({}),          # 坑一
        urllib.request.HTTPSHandler(context=ctx),  # 坑二
    )


_OPENER = _build_opener()


class HttpError(RuntimeError):
    pass


def _request(
    url: str,
    *,
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 30,
    retries: int = 5,
    backoff: float = 1.5,
) -> bytes:
    """发一个请求，失败按指数退避重试。返回响应体字节。

    退避分两条曲线，因为两类失败的时间尺度差一个量级：

    · 服务端**回话了**（429/5xx）：短退避即可，并尊重它的 Retry-After
    · **网络层**失败（DNS 解析不出、连接被拒、TLS 中断）：这类通常持续
      几秒到几十秒。原来两类共用 1.5^n，总共只扛得住 ~4 秒 —— 实测
      2026-09-30 一次 DNS 抖动就打死了一轮跑了 13 分钟的建索引。
      所以网络层单独走 2/4/8/16 秒、封顶 30 秒的曲线。
    """
    hdrs = {"User-Agent": USER_AGENT}
    if headers:
        hdrs.update(headers)

    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=data, headers=hdrs)
            with _OPENER.open(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            # 4xx 是请求本身的问题，重试没意义（429 除外）
            if e.code == 429 or e.code >= 500:
                last = e
            else:
                body = ""
                try:
                    body = e.read().decode("utf-8", "replace")[:200]
                except Exception:
                    pass
                raise HttpError(f"HTTP {e.code} {url} :: {body}") from e
        except Exception as e:                     # 网络层错误，可重试
            last = e

        if attempt < retries - 1:
            if isinstance(last, urllib.error.HTTPError):
                # 服务端明确回话了：短退避就够，但它说等多久就等多久
                wait = backoff ** (attempt + 1)
                ra = last.headers.get("Retry-After")
                if ra and ra.isdigit():
                    wait = max(wait, float(ra))
            else:
                # 网络层错误：DNS 抖动、连接被拒、TLS 中断，都是"等一会儿就好"
                # 那一类。封顶 30 秒，免得真断网时死等。
                wait = min(2.0 * (2 ** attempt), 30.0)
            time.sleep(wait)

    raise HttpError(f"重试 {retries} 次仍失败 {url} :: {last}") from last


# ---------------------------------------------------------------- 便捷方法

def get_json(url: str, **kw) -> Any:
    return json.loads(_request(url, **kw).decode("utf-8"))


def post_json(url: str, obj: Any, *, headers: dict[str, str] | None = None, **kw) -> Any:
    """POST 一个 JSON 体，返回解析后的 JSON。百炼的接口都走这里。"""
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    return json.loads(_request(url, data=body, headers=hdrs, **kw).decode("utf-8"))


def post_form_json(url: str, form: dict[str, str], *, referer: str | None = None, **kw) -> Any:
    """巨潮的接口都是 form 表单 + XHR 头。"""
    headers = {
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "X-Requested-With": "XMLHttpRequest",
    }
    if referer:
        headers["Referer"] = referer
    body = urllib.parse.urlencode(form).encode()
    return json.loads(_request(url, data=body, headers=headers, **kw).decode("utf-8"))


def download(url: str, dest: Path, *, timeout: int = 300, retries: int = 3) -> Path:
    """下载文件；已存在且非空则跳过（绝不要重复下载）。"""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with _OPENER.open(req, timeout=timeout) as resp, open(tmp, "wb") as f:
                while chunk := resp.read(1 << 16):
                    f.write(chunk)
            tmp.replace(dest)
            return dest
        except Exception:
            tmp.unlink(missing_ok=True)
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    return dest


def smoke() -> None:
    """自查：确认这层真的能出去。

    注意 DashScope 根路径返回 404 —— 这不是失败，恰恰证明 TCP 和 TLS 都通了。
    （巨潮是 HTTP 明文，验不出证书问题，所以必须拿一个 HTTPS 站点来测。）
    """
    print("== net.py 自查 ==")
    body = _request("http://www.cninfo.com.cn/new/index", timeout=20)
    print(f"  巨潮 (HTTP)              : {len(body):,} 字节  ✅")

    try:
        body = _request("https://dashscope.aliyuncs.com", timeout=20, retries=1)
        print(f"  DashScope (HTTPS)        : {len(body):,} 字节  ✅")
    except HttpError as e:
        if "HTTP 4" in str(e):
            print(f"  DashScope (HTTPS)        : 已连上，服务端回了 4xx（正常）✅")
        else:
            raise
    print("  → 代理已绕过、证书已挂载，这层可用")


if __name__ == "__main__":
    smoke()
