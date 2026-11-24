"""myvideoai command-line interface.

Examples
--------
# upscale 2x (default)
python -m myvideoai.cli -i input.mp4 -o output.mp4

# 4x upscale + double fps
python -m myvideoai.cli -i in.mp4 -o out.mp4 --scale 4 --interp 2

# denoise-only (no upscale) via 2x model already handles restoration
python -m myvideoai.cli -i in.mp4 -o out.mp4 --scale 2 --codec h264_nvenc
"""
from __future__ import annotations

import argparse
import os
import sys

from .core.config import EnhanceConfig, OUTPUT_DIR
from .core.pipeline import run


def _progress(done, total, extra):
    if done < 0:  # plugin log message routed through progress
        sys.stdout.write(f"\n{extra}\n")
        sys.stdout.flush()
        return
    if total:
        pct = 100.0 * done / total
        bar = "#" * int(pct // 2) + "-" * (50 - int(pct // 2))
        sys.stdout.write(f"\r[{bar}] {done}/{total} ({pct:5.1f}%) {extra}")
    else:
        sys.stdout.write(f"\r{done} frames {extra}")
    sys.stdout.flush()


def build_parser():
    p = argparse.ArgumentParser("myvideoai", description="Open-source AI video enhancement")
    p.add_argument("-i", "--input", default=None, help="input video path (or use --batch)")
    p.add_argument("-o", "--output", default=None, help="output path (default output/<name>_enhanced.mp4)")
    p.add_argument("--scale", type=float, default=2.0, help="output scale factor vs source (e.g. 2, 4)")
    p.add_argument("--no-upscale", action="store_true", help="skip upscaling")
    p.add_argument("--backend", default="esrgan", choices=["esrgan", "cugan", "model"],
                   help="upscaler backend: esrgan (photo) / cugan (anime) / model (drop-in plugin)")
    p.add_argument("--model-backend", default="auto",
                   choices=["auto", "cuda", "directml", "cpu"],
                   help="execution backend for MODEL plugins (auto detects best; "
                        "directml works on AMD/Intel GPUs)")
    p.add_argument("--model", default="x4plus", choices=["x4plus", "x4plus_anime", "x2plus"],
                   help="Real-ESRGAN weight preset (only for --backend esrgan)")
    p.add_argument("--cugan-tile", type=int, default=0, choices=[0, 1, 2, 3, 4],
                   help="RealCUGAN tile mode (0=full .. 4=quarter, lower VRAM)")
    p.add_argument("--face", action="store_true", help="enable GFPGAN face restoration")
    p.add_argument("--face-weight", type=float, default=0.5, help="GFPGAN strength 0-1")
    p.add_argument("--batch", nargs="+", default=None, help="batch mode: multiple input videos")
    p.add_argument("--interp", type=int, default=1, choices=[1, 2, 4],
                   help="fps multiplier via RIFE (1=off, 2=double, 4=quad)")
    p.add_argument("--tile", type=int, default=0, help="tile size for low VRAM (0=auto)")
    p.add_argument("--crf", type=int, default=18)
    p.add_argument("--preset", default="medium")
    p.add_argument("--codec", default="libx264", choices=["libx264", "h264_nvenc", "hevc_nvenc"])
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--no-half", action="store_true", help="disable fp16")
    # plugin features (built-in plugins, same entry point as third-party)
    p.add_argument("--hdr", action="store_true", help="SDR->HDR10 via libplacebo (video plugin)")
    p.add_argument("--stabilize", action="store_true", help="vidstab stabilization (video plugin)")
    p.add_argument("--plugins", nargs="*", default=[], help="extra plugin ids to enable")
    p.add_argument("--plugin-param", action="append", default=[],
                   metavar="PLUGIN.K=VALUE", help="override a plugin param, e.g. builtin.hdr.crf=20")
    p.add_argument("--list-plugins", action="store_true", help="list available plugins and exit")
    return p


def _make_cfg(inp, out, args):
    enabled = list(args.plugins)
    if args.backend == "model":
        # pick the first enabled (or default) model plugin that provides upscale
        from .plugins import discover
        from .core.config import PLUGINS_DIR, USER_PLUGINS_DIR
        # builtin.esrgan/cugan are the dedicated --backend esrgan|cugan models;
        # --backend model means a drop-in plugin, so exclude them from the
        # default candidate list (otherwise they win by alphabetical order).
        model_ids = [m.id for m in (discover(PLUGINS_DIR) + discover(USER_PLUGINS_DIR))
                     if m.kind == "model" and m.provides == "upscale"
                     and m.id not in ("builtin.esrgan", "builtin.cugan")]
        chosen = next((p for p in enabled if p in model_ids),
                      model_ids[0] if model_ids else "builtin.onnx_superres")
        if chosen not in enabled:
            enabled.append(chosen)
    if args.hdr:
        enabled.append("builtin.hdr")
    if args.stabilize:
        enabled.append("builtin.stabilize")
    # parse --plugin-param PLUGIN.K=V into nested dict (values coerced to int/float/bool)
    pp: dict = {}
    for item in args.plugin_param:
        if "." not in item or "=" not in item:
            print(f"[warn] ignore bad --plugin-param '{item}' (want PLUGIN.K=V)")
            continue
        key, raw = item.split("=", 1)
        pid, _, param = key.rpartition(".")
        val = raw
        if raw.lower() in ("true", "false"):
            val = raw.lower() == "true"
        else:
            try:
                val = int(raw)
            except ValueError:
                try:
                    val = float(raw)
                except ValueError:
                    pass
        pp.setdefault(pid, {})[param] = val
    return EnhanceConfig(
        input_path=inp,
        output_path=out,
        do_upscale=not args.no_upscale,
        upscale_backend=args.backend,
        upscale_model=args.model,
        outscale=args.scale,
        tile=args.tile,
        cugan_tile_mode=args.cugan_tile,
        do_face=args.face,
        face_weight=args.face_weight,
        do_interp=args.interp > 1,
        interp_mult=args.interp,
        crf=args.crf,
        preset=args.preset,
        codec=args.codec,
        device=args.device,
        model_backend=args.model_backend,
        half=not args.no_half,
        enabled_plugins=list(dict.fromkeys(enabled)),
        plugin_params=pp,
    )


def _default_out(inp):
    base = os.path.splitext(os.path.basename(inp))[0]
    return str(OUTPUT_DIR / f"{base}_enhanced.mp4")


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.list_plugins:
        from .plugins import discover
        from .core.config import PLUGINS_DIR, USER_PLUGINS_DIR
        ms = discover(PLUGINS_DIR) + discover(USER_PLUGINS_DIR)
        print(f"{'id':28} {'kind':7} {'version':9} description")
        for m in ms:
            print(f"{m.id:28} {m.kind:7} {m.version:9} {m.description}")
        return 0

    # resolve input list: --batch or -i
    inputs = args.batch if args.batch else ([args.input] if args.input else [])
    if not inputs:
        print("[error] provide -i INPUT or --batch F1 F2 ...")
        return 2
    missing = [p for p in inputs if not os.path.isfile(p)]
    if missing:
        print(f"[error] input not found: {missing}")
        return 2

    jobs = []
    for inp in inputs:
        out = args.output if (args.output and len(inputs) == 1) else _default_out(inp)
        jobs.append((inp, out))

    print(f"[myvideoai] {len(jobs)} job(s) queued, backend={args.backend} "
          f"model_backend={args.model_backend} face={args.face} scale={args.scale} "
          f"interp=x{args.interp}")
    done = 0
    for i, (inp, out) in enumerate(jobs, 1):
        print(f"\n[{i}/{len(jobs)}] {inp} -> {out}")
        cfg = _make_cfg(inp, out, args)
        try:
            path = run(cfg, progress=_progress)
            print(f"\n[done] {path}")
            done += 1
        except Exception as e:
            print(f"\n[error] {inp}: {type(e).__name__}: {e}")
    print(f"\n[summary] {done}/{len(jobs)} succeeded")
    return 0 if done == len(jobs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
