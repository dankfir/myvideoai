"""Pre-fetch the model weights used by myvideoai.

Thin wrapper around :func:`myvideoai.core.weights.ensure_weights` -- the single
downloader in this project. It resolves the weights directory from the package
location (never a hard-coded machine path) and fetches the requested assets
through the mirror-fallback logic, with Content-Length validation and atomic
``.part`` replacement.

Usage:
    python -m myvideoai.download_assets                 # all assets below
    python -m myvideoai.download_assets GFPGANv1.4.pth  # just one
"""
from __future__ import annotations

import sys

from .core.config import CUGAN_WEIGHTS, GFPGAN_WEIGHTS, WEIGHTS_DIR
from .core.weights import ensure_weights

# asset name -> ordered list of mirrors (mirror first, upstream as fallback)
ASSETS: dict[str, list[str]] = {
    "GFPGANv1.4.pth": list(GFPGAN_WEIGHTS["GFPGANv1.4.pth"]),
    "detection_Resnet50_Final.pth": list(GFPGAN_WEIGHTS["detection_Resnet50_Final.pth"]),
    "parsing_parsenet.pth": list(GFPGAN_WEIGHTS["parsing_parsenet.pth"]),
    "RealCUGAN_4x.pth": list(CUGAN_WEIGHTS["cugan_4x"]),
}


def fetch(name: str) -> bool:
    """Fetch one asset into the shared weights directory. Returns success."""
    urls = ASSETS[name]
    try:
        path = ensure_weights(urls, name, dest_dir=WEIGHTS_DIR, log=print)
    except RuntimeError as e:
        print(f"[fail] {name}: {e}", flush=True)
        return False
    print(f"[ok] {name} {path.stat().st_size / 1e6:.1f}MB -> {path}", flush=True)
    return True


def main(argv: list[str]) -> int:
    targets = [a for a in argv if a in ASSETS]
    unknown = [a for a in argv if a not in ASSETS]
    for a in unknown:
        print(f"[skip] unknown asset {a}", flush=True)
    if not targets:
        targets = list(ASSETS)
    print(f"[weights] directory: {WEIGHTS_DIR}", flush=True)
    ok = sum(1 for name in targets if fetch(name))
    print(f"DONE {ok}/{len(targets)}", flush=True)
    return 0 if ok == len(targets) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
