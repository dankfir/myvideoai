"""End-to-end enhancement pipeline: decode -> upscale -> frame-plugins -> interpolate -> encode -> video-plugins.

Streams frames so memory stays bounded regardless of video length.
Plugins are first-class: built-in HDR/stabilization AND the built-in upscalers
(esrgan/cugan) all run through the same registry (see myvideoai/plugins).
"""
from __future__ import annotations

import os
import time
from dataclasses import replace
from typing import Callable, Optional

import numpy as np

from .config import (EnhanceConfig, PLUGINS_DIR, USER_PLUGINS_DIR, OUTPUT_DIR,
                     WEIGHTS_DIR)
from .video_io import probe, iter_frames, FrameWriter, FFMPEG, FFPROBE
from .model_backend import pick


ProgressCB = Optional[Callable[[int, int, str], None]]


def _build_registry(cfg: EnhanceConfig):
    """Discover plugins, auto-inject the built-in upscaler, keep enabled ids.

    The built-in esrgan/cugan upscalers ARE model plugins now. Selecting one via
    --backend injects its plugin id here, so built-in and third-party models go
    through the exact same model_chain in run(). An explicitly enabled
    third-party upscale plugin wins over the auto-injected built-in one.
    """
    from ..plugins import discover, PluginRegistry
    from ..plugins.spec import PluginManifest

    manifests = discover(PLUGINS_DIR) + discover(USER_PLUGINS_DIR)
    wanted = set(cfg.enabled_plugins)

    if cfg.do_upscale and cfg.upscale_backend in ("esrgan", "cugan"):
        explicit_upscaler = any(
            m.id in wanted and m.kind == "model" and m.provides == "upscale"
            for m in manifests
        )
        if not explicit_upscaler:
            wanted.add("builtin." + cfg.upscale_backend)

    if not wanted:
        return None
    selected: list[PluginManifest] = []
    seen = set()
    for m in manifests:
        if m.id in wanted and m.id not in seen:
            # explicit opt-in wins over manifest default — but never mutate the
            # shared manifest object; work on a copy so other callers of
            # discover() keep seeing the declared default.
            new = replace(m, enabled=True)
            # `_dir` is stashed by discover() as a non-dataclass attribute;
            # replace() drops it, and the loader needs it to find the entry.
            new.__dict__["_dir"] = m.__dict__.get("_dir")
            selected.append(new)
            seen.add(m.id)
    missing = wanted - seen
    if missing:
        raise ValueError(f"enabled plugins not found: {sorted(missing)}")
    upscale_ids = [m.id for m in selected
                   if m.kind == "model" and m.provides == "upscale"]
    if len(upscale_ids) > 1:
        raise ValueError(
            f"multiple upscale model plugins enabled: {upscale_ids}; pick exactly one"
        )
    if not selected:
        return None
    return PluginRegistry(selected)


def _ctx_factory(cfg: EnhanceConfig, work_dir: str, log):
    """Build a PluginContext factory that merges per-plugin params overrides."""
    from ..plugins.spec import PluginContext

    # resolve model backend once (auto → cuda > directml > cpu)
    choice = pick(cfg.model_backend)

    def factory(manifest):
        params = dict(manifest.params)
        if manifest.kind == "model":
            # cfg-derived runtime values override manifest defaults; explicit
            # --plugin-param still wins (applied last).
            params.update({
                "upscale_model": cfg.upscale_model,
                "tile": cfg.tile,
                "cugan_tile_mode": cfg.cugan_tile_mode,
            })
        params.update(cfg.plugin_params.get(manifest.id, {}))
        return PluginContext(
            work_dir=work_dir,
            device=choice.device,
            torch_device=cfg.device,
            half=cfg.half,
            backend=choice.name,
            ort_provider=choice.ort_provider,
            weights_dir=str(WEIGHTS_DIR),
            ffmpeg=FFMPEG,
            ffprobe=FFPROBE,
            params=params,
            emit=log,
        )
    return factory


def run(cfg: EnhanceConfig, progress: ProgressCB = None) -> str:
    """Execute one enhancement job. Returns output path."""
    info = probe(cfg.input_path)
    src_w, src_h = info.width, info.height
    src_fps = info.fps

    def log(msg: str):
        if progress:
            progress(-1, 0, msg)
        else:
            print(msg)

    registry = _build_registry(cfg)
    # Use the project output dir for intermediates rather than the system temp
    # dir, so the ffmpeg encode pipe has a guaranteed-writable location.
    if cfg.temp_dir:
        work_dir = cfg.temp_dir
    else:
        work_dir = str(OUTPUT_DIR / "_work")
    os.makedirs(work_dir, exist_ok=True)

    # --- determine output geometry ---
    if cfg.do_upscale:
        scale = max(1.0, float(cfg.outscale))
        out_w = int(round(src_w * scale))
        out_h = int(round(src_h * scale))
        out_w += out_w % 2
        out_h += out_h % 2
    else:
        out_w, out_h = src_w, src_h

    out_fps = src_fps * (cfg.interp_mult if cfg.do_interp else 1)

    # Plugin context factory (shared by model / frame / video hooks).
    ctx_factory = _ctx_factory(cfg, work_dir, log)

    # --- build models lazily (ALL upscaling goes through the model chain) ---
    interp = None
    face = None
    model_apply = model_close = None
    model_provides_upscale = False
    if registry is not None:
        mmanifests = registry.model_manifests()
        model_provides_upscale = any(m.provides == "upscale" for m in mmanifests)
        if mmanifests:
            model_apply, model_close = registry.model_chain(ctx_factory)
            log(f"[plugins] model chain: {[m.id for m in mmanifests]} "
                f"(upscale={'yes' if model_provides_upscale else 'no'})")
    if cfg.do_upscale and not model_provides_upscale:
        raise ValueError(
            "upscale requested but no upscale-capable model plugin is enabled "
            "(use --backend esrgan|cugan|model)"
        )
    if cfg.do_face:
        from .face_enhancer import FaceEnhancer
        face = FaceEnhancer(device=cfg.device, weight=cfg.face_weight)
    if cfg.do_interp:
        from .interpolator import RIFEInterpolator
        interp = RIFEInterpolator(cfg.interp_model, device=cfg.device, half=cfg.half)

    # --- build frame plugin chain (setup once) ---
    frame_apply = frame_close = None
    has_video_plugins = False
    if registry is not None:
        frame_apply, frame_close = registry.frame_chain(ctx_factory)
        # does the registry have any video hooks?
        has_video_plugins = any(p["kind"] == "video" for p in registry.list_plugins())

    os.makedirs(os.path.dirname(os.path.abspath(cfg.output_path)) or ".", exist_ok=True)

    # If video plugins will run, encode to a temp file first, then chain-transform.
    final_path = cfg.output_path
    encode_path = (os.path.join(work_dir, "_encoded.mp4")
                   if has_video_plugins else cfg.output_path)

    writer = FrameWriter(encode_path, out_w, out_h, out_fps,
                         codec=cfg.codec, crf=cfg.crf, preset=cfg.preset,
                         pix_fmt=cfg.pix_fmt,
                         audio_from=cfg.input_path if info.has_audio else None)

    total = info.n_frames or 0
    idx = 0
    t0 = time.time()
    prev_out = None
    chain_open = True

    try:
        for frame in iter_frames(cfg.input_path, target_fps=src_fps, size=(src_w, src_h)):
            if model_apply is not None:
                # provides=upscale → do the resize; else frame-level (outscale 1.0)
                frame = model_apply(frame, cfg.outscale if model_provides_upscale else 1.0)
            if face is not None:
                frame = face.enhance(frame)

            # frame plugins run on every full-res frame, before interpolation
            if frame_apply is not None:
                frame = frame_apply(frame)

            if interp is not None and prev_out is not None:
                for k in range(1, cfg.interp_mult):
                    t = k / cfg.interp_mult
                    mid = interp.interpolate(prev_out, frame, t)
                    writer.write(mid)

            writer.write(frame)
            prev_out = frame
            idx += 1
            if progress and (idx % 5 == 0 or idx == total):
                fps_now = idx / max(1e-6, time.time() - t0)
                progress(idx, total, f"{fps_now:.2f} it/s")
        if frame_close is not None:
            frame_close()
        if model_close is not None:
            model_close()
        chain_open = False
    finally:
        writer.close()
        if chain_open:
            if frame_close is not None:
                frame_close()
            if model_close is not None:
                model_close()

    # --- video plugins: whole-file post pass (stabilize -> hdr, etc.) ---
    if has_video_plugins:
        log(f"[plugins] video chain: {[p['id'] for p in registry.list_plugins('video')]}")
        registry.run_video(encode_path, final_path, ctx_factory)

    return final_path
