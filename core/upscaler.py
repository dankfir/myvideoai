"""Real-ESRGAN upscaling / restoration wrapper.

Uses the official realesrgan + basicsr packages and released weights.
Supports tiling to fit small VRAM (e.g. RTX 4060 8GB laptop).
"""
from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np

from .config import REALESRGAN_WEIGHTS, WEIGHTS_DIR
from .weights import ensure_weights


class RealESRGANUpscaler:
    """Lazy-loaded Real-ESRGAN upscaler with FP16 + tile support."""

    def __init__(self, model_key: str = "x4plus", device: str = "cuda",
                 half: bool = True, tile: int = 0):
        self.model_key = model_key
        self.device = device
        self.half = half and device == "cuda"
        self.tile = tile
        self._upsampler = None
        self.native_scale = 4

    def _build(self):
        # basicsr uses deprecated pkg_resources / torchvision.transforms.functional_tensor
        # -> patch the known import break on newer torchvision before importing.
        try:
            import torchvision.transforms.functional_tensor  # noqa: F401
        except Exception:
            import types
            import torchvision.transforms.functional as F
            mod = types.ModuleType("torchvision.transforms.functional_tensor")
            mod.rgb_to_grayscale = F.rgb_to_grayscale
            sys.modules["torchvision.transforms.functional_tensor"] = mod

        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        if "anime" in self.model_key:
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                            num_block=6, num_grow_ch=32, scale=4)
            self.native_scale = 4
        elif self.model_key == "x2plus":
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                            num_block=23, num_grow_ch=32, scale=2)
            self.native_scale = 2
        else:  # x4plus
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                            num_block=23, num_grow_ch=32, scale=4)
            self.native_scale = 4

        wpath = ensure_weights(REALESRGAN_WEIGHTS[self.model_key],
                               f"{self.model_key}.pth",
                               dest_dir=WEIGHTS_DIR, log=print)

        tile = self.tile if self.tile > 0 else 0
        self._upsampler = RealESRGANer(
            scale=self.native_scale,
            model_path=str(wpath),
            model=model,
            tile=tile,
            tile_pad=10,
            pre_pad=0,
            half=self.half,
            device=self.device,
        )

    def upscale(self, frame_bgr: np.ndarray, outscale: Optional[float] = None) -> np.ndarray:
        """Upscale one BGR uint8 frame. outscale None -> native scale."""
        if self._upsampler is None:
            self._build()
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        # RealESRGANer.enhance expects BGR uint8, returns BGR uint8
        out, _ = self._upsampler.enhance(frame_bgr, outscale=outscale or self.native_scale)
        return out
