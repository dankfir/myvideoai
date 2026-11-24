"""Built-in Real-CUGAN, wrapped as a first-class ModelPlugin.

Same contract as the ONNX and third-party torch plugins — the pipeline cannot
tell built-in models apart from dropped-in ones.
"""
from __future__ import annotations

import numpy as np

from myvideoai.core.cugan_upscaler import CUGANUpscaler


class CUGANModelPlugin:
    def __init__(self):
        self.upscaler = None

    def setup(self, ctx) -> None:
        device = ctx.device_for("torch") or "cuda"
        # Accept both the new canonical key and the legacy cfg field name.
        tile_mode = int(ctx.params.get("tile_mode",
                                       ctx.params.get("cugan_tile_mode", 0)) or 0)
        ctx.log(f"[builtin.cugan] device={device} tile_mode={tile_mode} "
                f"half={ctx.half}")
        self.upscaler = CUGANUpscaler(device=device, half=ctx.half,
                                      tile_mode=tile_mode)

    def process(self, frame: np.ndarray, outscale: float) -> np.ndarray:
        return self.upscaler.upscale(frame, outscale=outscale)

    def teardown(self) -> None:
        self.upscaler = None
