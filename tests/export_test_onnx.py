"""Export a tiny ONNX super-res model as a drop-in test fixture.

Produces a 2x PixelShuffle SR model (NCHW RGB in/out). This is NOT a quality
model — its only job is to prove the plugin system can load ANY ONNX model
with zero core-code changes. Run once:

    python -m myvideoai.tests.export_test_onnx
"""
from pathlib import Path

import torch
import torch.nn as nn

from ..core.config import WEIGHTS_DIR


class TinySR(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 12, 3, padding=1)
        self.ps = nn.PixelShuffle(2)   # 12ch -> 3ch, H/W x2

    def forward(self, x):
        return self.ps(self.conv(x))


def main():
    model = TinySR().eval()
    dummy = torch.zeros(1, 3, 32, 32)
    dest = Path(WEIGHTS_DIR) / "test_sr_2x.onnx"
    torch.onnx.export(model, dummy, str(dest),
                      input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {2: "h", 3: "w"},
                                    "output": {2: "h", 3: "w"}},
                      opset_version=17, dynamo=False)
    print("exported", dest, dest.stat().st_size, "bytes")
    # sanity: output shape must be 2x
    with torch.no_grad():
        y = model(dummy)
    print("output shape", tuple(y.shape))


if __name__ == "__main__":
    main()
