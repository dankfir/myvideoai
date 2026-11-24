"""Generic ONNX super-resolution model plugin (data-driven, zero-code drop-in).

To add a new ONNX super-res model: copy this folder, edit plugin.json's params
(model_url / model_file / input_name / output_name / native_scale / ...). No
python changes, no core-code changes. The same class serves every model.

Input convention (matches most ONNX SR exports): NCHW, RGB, float32 in [0,1].
Output: NCHW, RGB, float32; resized to the requested outscale.
Tile inference keeps VRAM bounded on 8GB cards.
"""
from __future__ import annotations

import numpy as np
import cv2

from myvideoai.core.weights import ensure_weights


class ONNXSuperResPlugin:
    """kind=model. setup() builds the session using the resolved backend."""

    def __init__(self):
        self.sess = None
        self.in_name = None
        self.out_name = None
        self.native = 4
        self.tile = 0
        self.norm = "0to1"
        self.layout = "nchw"
        self.ctx = None

    # -- helpers ---------------------------------------------------------------
    def _log(self, msg):
        if self.ctx:
            self.ctx.log(f"[onnx_sr] {msg}")

    def _bgr_to_tensor(self, bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x = rgb.astype(np.float32) / 255.0 if self.norm == "0to1" else rgb.astype(np.float32)
        x = np.transpose(x, (2, 0, 1))[None, ...]  # NCHW
        return np.ascontiguousarray(x)

    def _tensor_to_bgr(self, y: np.ndarray) -> np.ndarray:
        if self.layout == "nhwc":
            y = np.transpose(y, (0, 3, 1, 2))
        y = np.squeeze(y, 0)
        y = np.transpose(y, (1, 2, 0))
        y = y * 255.0 if self.norm == "0to1" else y
        y = np.clip(y, 0, 255).astype(np.uint8)
        return cv2.cvtColor(y, cv2.COLOR_RGB2BGR)

    def _infer_tiled(self, x: np.ndarray) -> np.ndarray:
        """x: 1CHW float. Tile spatially if self.tile>0, else whole frame."""
        _, c, h, w = x.shape
        if self.tile <= 0 or (h <= self.tile and w <= self.tile):
            return self.sess.run([self.out_name], {self.in_name: x})[0]
        t = self.tile
        out_h, out_w = h * self.native, w * self.native
        acc = np.zeros((1, c, out_h, out_w), np.float32)
        wsum = np.zeros((1, c, out_h, out_w), np.float32)
        for y in range(0, h, t):
            for xx in range(0, w, t):
                patch = x[:, :, y:y + t, xx:xx + t]
                ph, pw = patch.shape[2], patch.shape[3]
                p = np.zeros((1, c, t, t), np.float32)
                p[:, :, :ph, :pw] = patch
                o = self.sess.run([self.out_name], {self.in_name: p})[0]
                oh, ow = ph * self.native, pw * self.native
                acc[:, :, y * self.native:y * self.native + oh,
                    xx * self.native:xx * self.native + ow] += o[:, :, :oh, :ow]
                wsum[:, :, y * self.native:y * self.native + oh,
                     xx * self.native:xx * self.native + ow] += 1
        return acc / np.maximum(wsum, 1e-6)

    # -- ModelPlugin contract --------------------------------------------------
    def setup(self, ctx) -> None:
        import onnxruntime as ort
        self.ctx = ctx
        p = ctx.params
        self.native = int(p.get("native_scale", 4))
        self.tile = int(p.get("tile", 0))
        self.norm = p.get("norm", "0to1")
        self.layout = p.get("layout", "nchw")
        model_file = p.get("model_file", "model.onnx")
        model_url = p.get("model_url", "")
        path = ensure_weights(model_url, model_file,
                              dest_dir=__import__("pathlib").Path(ctx.weights_dir),
                              log=self._log) if model_url else (
            __import__("pathlib").Path(ctx.weights_dir) / model_file)
        if not path.exists():
            raise FileNotFoundError(f"ONNX model not found: {path}")
        provider = ctx.ort_provider or "CPUExecutionProvider"
        self.sess = ort.InferenceSession(str(path), providers=[provider])
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name
        self._log(f"loaded {model_file} via {provider} in={self.in_name} out={self.out_name}")

    def process(self, frame: np.ndarray, outscale: float) -> np.ndarray:
        x = self._bgr_to_tensor(frame)
        y = self._infer_tiled(x)
        out = self._tensor_to_bgr(y)
        target_h = int(round(frame.shape[0] * outscale))
        target_w = int(round(frame.shape[1] * outscale))
        if out.shape[:2] != (target_h, target_w):
            out = cv2.resize(out, (target_w, target_h), interpolation=cv2.INTER_AREA
                             if outscale < self.native else cv2.INTER_LANCZOS4)
        return out

    def teardown(self) -> None:
        self.sess = None
