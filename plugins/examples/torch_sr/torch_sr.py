"""Example: wrap an existing PyTorch SR model with the ModelPlugin protocol.

Shows the model hook is backend-agnostic — ONNX and torch models share ONE
contract. Reuses the project's proven RealESRGANUpscaler so this adapter is
~30 lines; a genuinely new torch architecture just swaps the loader here.

This plugin is disabled by default. Enable with:
    --backend model --plugins examples.torch_sr
"""
from __future__ import annotations


class TorchSRPlugin:
    kind = "model"

    def __init__(self):
        self.upscaler = None
        self.native = 4

    def setup(self, ctx) -> None:
        from myvideoai.core.upscaler import RealESRGANUpscaler
        from myvideoai.core.weights import ensure_weights
        from pathlib import Path
        p = ctx.params
        self.native = int(p.get("native_scale", 4))
        model_file = p.get("model_file", "RealESRGAN_x4plus.pth")
        url = p.get("model_url", "")
        if url:
            ensure_weights(url, model_file, dest_dir=Path(ctx.weights_dir),
                           log=ctx.log)
        # torch runtime device (torch semantics), not the ONNX/DirectML one
        dev = ctx.device_for("torch") or "cuda"
        self.upscaler = RealESRGANUpscaler("x4plus", device=dev,
                                           half=(dev == "cuda" and ctx.half),
                                           tile=int(p.get("tile", 256)))
        ctx.log(f"[torch_sr] ready on {dev} (native x{self.native})")

    def process(self, frame, outscale: float):
        return self.upscaler.upscale(frame, outscale=outscale)

    def teardown(self) -> None:
        self.upscaler = None
