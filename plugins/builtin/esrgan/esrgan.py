"""Built-in Real-ESRGAN, wrapped as a first-class ModelPlugin.

This is the dogfooding step: the built-in upscaler now goes through the exact
same ``setup(ctx) / process(frame, outscale) / teardown()`` contract as any
third-party ONNX or torch model. No privileged path in the pipeline.
"""
from __future__ import annotations

import numpy as np

# The registry loads plugin modules via spec_from_file_location (no package
# context), so absolute imports are required here.
from myvideoai.core.upscaler import RealESRGANUpscaler
from myvideoai.core.device import auto_tile


class ESRGANModelPlugin:
    def __init__(self):
        self.upscaler = None

    def setup(self, ctx) -> None:
        # torch models must follow cfg.device, NOT the ONNX-oriented ctx.device.
        device = ctx.device_for("torch") or "cuda"
        tile = int(ctx.params.get("tile", 0) or 0)
        if tile <= 0 and bool(ctx.params.get("auto_tile", True)):
            tile = auto_tile(device)
        model_key = ctx.params.get("upscale_model", "x4plus")
        ctx.log(f"[builtin.esrgan] model={model_key} device={device} "
                f"tile={tile} half={ctx.half}")
        self.upscaler = RealESRGANUpscaler(model_key, device=device,
                                           half=ctx.half, tile=tile)

    def process(self, frame: np.ndarray, outscale: float) -> np.ndarray:
        return self.upscaler.upscale(frame, outscale=outscale)

    def teardown(self) -> None:
        self.upscaler = None
