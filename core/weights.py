"""Generic model weight downloader with mirror fallback.

Drop-in model plugins declare a URL (or list of mirrors) in their manifest;
``ensure_weights`` fetches it on first use into the shared weights directory.
Reuses the hard-won lessons: hf-mirror first, original as fallback, size check.
"""
from __future__ import annotations

import os
import shutil
import urllib.request
from pathlib import Path

from .config import WEIGHTS_DIR


def _head_size(url: str, timeout: int = 30) -> int:
    try:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "myvideoai"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return int(r.headers.get("Content-Length", 0))
    except Exception:
        return 0


def ensure_weights(urls: str | list[str], filename: str,
                   dest_dir: Path = WEIGHTS_DIR, log=None) -> Path:
    """Return a local path to ``filename``, downloading from the first working
    mirror if absent. ``urls`` is one URL or an ordered list of mirrors.

    A partial file is never trusted: if its size mismatches the server's
    Content-Length it is deleted and re-downloaded from the next mirror.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / filename
    if target.exists() and target.stat().st_size > 0:
        return target
    if isinstance(urls, str):
        urls = [urls]
    last = "no urls"
    for url in urls:
        expected = _head_size(url)
        tmp = target.with_suffix(target.suffix + ".part")
        try:
            if log:
                log(f"[weights] fetching {filename} from {url.split('/')[2]}")
            req = urllib.request.Request(url, headers={"User-Agent": "myvideoai"})
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, length=1 << 20)
            if expected and tmp.stat().st_size != expected:
                last = f"size mismatch {tmp.stat().st_size} != {expected}"
                tmp.unlink(missing_ok=True)
                continue
            os.replace(tmp, target)
            return target
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
            tmp.unlink(missing_ok=True)
    raise RuntimeError(f"cannot fetch {filename}: {last}")
