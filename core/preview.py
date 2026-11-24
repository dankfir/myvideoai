"""Timeline frame preview: grab a frame at time T, run enhancement, return
original + enhanced pair for side-by-side comparison.

Used by the GUI timeline scrubber; kept in core so it is headless-testable.
"""
from __future__ import annotations

import subprocess
from typing import Optional, Tuple

import numpy as np

from .config import EnhanceConfig
from .video_io import FFMPEG, probe


def grab_frame(video: str, time_sec: float) -> np.ndarray:
    """Grab one BGR frame at time_sec via ffmpeg seek (no re-encode)."""
    info = probe(video)
    cmd = [
        FFMPEG, "-v", "error", "-ss", f"{max(0.0, time_sec):.3f}",
        "-i", str(video), "-frames:v", "1",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]
    size = info.width * info.height * 3
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        raw = p.stdout.read(size)
    finally:
        p.stdout.close()
        p.wait()
    if len(raw) < size:
        raise RuntimeError(f"could not grab frame at {time_sec}s from {video}")
    return np.frombuffer(raw, np.uint8).reshape(info.height, info.width, 3).copy()


def render_preview(video: str, time_sec: float, cfg: EnhanceConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Return (original_bgr, enhanced_bgr) for a timeline position.

    Uses the SAME model chain as the full pipeline (registry + model hooks), so
    a drop-in ONNX/torch model selected in the GUI is exactly what the preview
    shows — no hidden esrgan/cugan fallback.
    """
    frame = grab_frame(video, time_sec)
    out = frame

    # Build the model chain exactly like pipeline.run(): plugin registry +
    # context factory, honoring model_backend / plugin_params / selected file.
    from .pipeline import _build_registry, _ctx_factory
    from .config import OUTPUT_DIR

    work = str(OUTPUT_DIR / "_preview_work")
    reg = _build_registry(cfg)
    model_apply = model_close = None
    provides = False
    if reg is not None:
        ctxf = _ctx_factory(cfg, work, lambda m: None)
        provides = any(m.provides == "upscale" for m in reg.model_manifests())
        model_apply, model_close = reg.model_chain(ctxf)

    if cfg.do_upscale:
        if model_apply is not None:
            try:
                out = model_apply(out, cfg.outscale if provides else 1.0)
            finally:
                model_close()
        else:  # no upscale plugin selected — faithful error, never silent fallback
            raise RuntimeError(
                f"no upscale model plugin for backend={cfg.upscale_backend!r}; "
                f"enable a model plugin (e.g. builtin.onnx_superres)")

    if cfg.do_face:
        from .face_enhancer import FaceEnhancer
        out = FaceEnhancer(device=cfg.device, weight=cfg.face_weight).enhance(out)
    return frame, out
