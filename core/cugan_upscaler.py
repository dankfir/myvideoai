"""Real-CUGAN (bilibili/ailab) upscaler wrapper for anime/illustration content.

Uses the official UpCunet architecture (MIT, Copyright (c) 2022 bilibili)
with the released 4x weight.
Complements Real-ESRGAN: RealCUGAN is tuned for anime line art / flat colors,
Real-ESRGAN for photorealistic content.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from .config import CUGAN_WEIGHTS, WEIGHTS_DIR
from .weights import ensure_weights


class CUGANUpscaler:
    """Lazy-loaded Real-CUGAN 4x upscaler with FP16 + tile mode support.

    tile_mode: 0=full, 1=half long edge, 2=half both, 3=third, 4=quarter
    (higher tile mode = less VRAM, per official Real-CUGAN).
    """

    def __init__(self, weight_key: str = "cugan_4x", device: str = "cuda",
                 half: bool = True, tile_mode: int = 0):
        self.weight_key = weight_key
        self.device = device
        self.half = half and device == "cuda"
        self.tile_mode = tile_mode
        self.native_scale = 4
        self._upscaler = None

    def _build(self):
        from .cugan_arch import RealWaifuUpScaler
        wpath = ensure_weights(CUGAN_WEIGHTS[self.weight_key],
                               f"{self.weight_key}.pth",
                               dest_dir=WEIGHTS_DIR, log=print)
        self._upscaler = RealWaifuUpScaler(
            scale=self.native_scale, weight_path=str(wpath),
            half=self.half, device=self.device)

    def upscale(self, frame_bgr: np.ndarray, outscale: Optional[float] = None) -> np.ndarray:
        if self._upscaler is None:
            self._build()
        out = self._upscaler(frame_bgr, self.tile_mode)  # BGR uint8 -> BGR uint8, 4x
        target = outscale or self.native_scale
        if target != self.native_scale:
            import cv2
            h, w = int(round(frame_bgr.shape[0] * target)), int(round(frame_bgr.shape[1] * target))
            out = cv2.resize(out, (w, h), interpolation=cv2.INTER_AREA if target < self.native_scale else cv2.INTER_CUBIC)
        return out
