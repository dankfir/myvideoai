"""myvideoai MCP server — zero-dependency, stdio JSON-RPC 2.0.

Exposes every myvideoai capability as an MCP tool so an agent can drive the whole
application (video enhancement, probing, timeline preview, plugin authoring,
model registration, backend detection) without touching CLI/GUI internals.

Why hand-rolled instead of the official ``mcp`` SDK:
  - pip is unusable in this sandbox (ACL blocks wheel unpack), and the SDK pulls
    a dependency tree (anyio / httpx / pydantic-settings ...) that would each
    need the same manual wheel dance. MCP's stdio transport IS line-delimited
    JSON-RPC 2.0 — implementable in ~350 lines with the stdlib, zero deps.
  - Any MCP client (Claude Desktop / Cline / OpenWebUI, etc.) speaks this.

Protocol channel discipline (critical): stdout carries ONLY JSON-RPC messages.
Every log/warning/debug goes to stderr. Breaking this corrupts the stream.

Launch (from the directory that contains the ``myvideoai`` package):
  set PYTHONPATH=<repo-root>\\.vendored\\torch_pkgs;<repo-root>\\.vendored\\py_pkgs
  set KMP_DUPLICATE_LIB_OK=TRUE
  python -m myvideoai.mcp_server
"""
from __future__ import annotations

import json
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Protocol helpers — stdout is sacred, logs go to stderr
# ---------------------------------------------------------------------------
# Keep a private handle to the real stdout, then redirect the module-global
# sys.stdout to stderr.  Any library that does a bare print() (tqdm, torch,
# ffmpeg wrappers) now lands on stderr instead of corrupting the JSON-RPC
# stream. This is the #1 stdio-MCP trap: protocol channel must be pristine.
_REAL_STDOUT = sys.stdout
sys.stdout = sys.stderr


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _send(obj: dict) -> None:
    _REAL_STDOUT.write(json.dumps(obj, ensure_ascii=False) + "\n")
    _REAL_STDOUT.flush()


def _ok(result: Any, rid: Any) -> dict:
    return {"jsonrpc": "2.0", "id": rid, "result": result}


def _err(code: int, message: str, rid: Any) -> dict:
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message}}


def _tool_text(text: str, is_error: bool = False) -> dict:
    d = {"content": [{"type": "text", "text": text}]}
    if is_error:
        d["isError"] = True
    return d


def _tool_json(obj: Any) -> dict:
    return {"content": [{"type": "text", "text": json.dumps(obj, ensure_ascii=False, indent=2)}]}


# ---------------------------------------------------------------------------
# Argument validation helpers
# ---------------------------------------------------------------------------
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,63}$")


def _safe_plugin_id(pid: str) -> str:
    """Plugin ids become directory names — reject traversal / absolute paths."""
    if not isinstance(pid, str) or not _SAFE_ID.match(pid):
        raise ValueError(f"unsafe plugin id {pid!r}: use letters/digits/._- (max 64), no path separators")
    return pid


# ---------------------------------------------------------------------------
# Tool implementations — each is a thin adapter over core/ modules
# ---------------------------------------------------------------------------
def tool_probe_video(args: dict) -> dict:
    from .core.video_io import probe
    path = args.get("video") or args.get("input")
    if not path:
        raise ValueError("probe_video needs 'video' path")
    info = probe(path)
    return _tool_json({
        "width": info.width, "height": info.height, "fps": info.fps,
        "frames": info.n_frames, "duration": info.duration,
        "codec": info.codec, "has_audio": info.has_audio,
    })


def _enabled_upscale_model(ids: list[str]) -> bool:
    """True if one of the enabled plugin ids is a model plugin providing upscale.

    The old implementation guessed from the plugin id string
    ("onnx" in id or "sr" in id), which misfires on ids that merely mention
    those substrings and misses every plugin that does not. Resolve the ids
    against the discovered manifests and ask what they actually declare.

    Never prints to stdout — the protocol channel is sacred.
    """
    from .core.config import PLUGINS_DIR, USER_PLUGINS_DIR
    from .plugins import discover
    from .plugins.spec import PluginError
    try:
        manifests = discover(PLUGINS_DIR) + discover(USER_PLUGINS_DIR)
    except PluginError as e:
        _log(f"[mcp] plugin discovery failed: {e}")
        return False
    wanted = set(ids)
    return any(m.id in wanted and m.kind == "model" and m.provides == "upscale"
               for m in manifests)


def tool_list_backends(args: dict) -> dict:
    from .core import model_backend as mb
    choices = mb.detect()
    return _tool_json({
        "available": [
            {"name": c.name, "device": c.device, "ort_provider": c.ort_provider, "note": c.note}
            for c in choices
        ],
        "auto_pick": {
            "name": mb.pick("auto").name, "device": mb.pick("auto").device,
            "ort_provider": mb.pick("auto").ort_provider,
        },
    })


def tool_list_plugins(args: dict) -> dict:
    from .core.config import PLUGINS_DIR, USER_PLUGINS_DIR
    from .plugins import discover
    items = []
    for m in discover(PLUGINS_DIR) + discover(USER_PLUGINS_DIR):
        items.append({
            "id": m.id, "name": m.name, "kind": m.kind, "enabled": m.enabled,
            "description": m.description, "order": m.order,
            "provides": m.provides, "backends": m.backends, "params": m.params,
        })
    return _tool_json({"plugins": items, "count": len(items)})


def tool_enhance_video(args: dict) -> dict:
    from .core.config import EnhanceConfig, OUTPUT_DIR
    from .core.pipeline import run
    inp = args.get("input") or args.get("video")
    if not inp:
        raise ValueError("enhance_video needs 'input' video path")
    backend = args.get("backend", "esrgan")
    if backend not in ("esrgan", "cugan", "model"):
        raise ValueError(f"backend must be esrgan|cugan|model, got {backend!r}")
    mult = int(args.get("interp", 1))
    if mult not in (1, 2, 4):
        raise ValueError("interp must be 1, 2 or 4")
    out = args.get("output") or str(OUTPUT_DIR / (Path(inp).stem + "_enhanced.mp4"))
    enabled = list(args.get("plugins") or [])
    if args.get("hdr"):
        enabled.append("builtin.hdr")
    if args.get("stabilize"):
        enabled.append("builtin.stabilize")
    if backend == "model" and not _enabled_upscale_model(enabled):
        # No enabled plugin actually provides upscaling — fall back to the
        # built-in ONNX super-resolution plugin. The check is capability-based
        # (manifest kind/provides), not a substring guess on plugin ids.
        enabled.append("builtin.onnx_superres")

    plugin_params = dict(args.get("plugin_params") or {})
    cfg = EnhanceConfig(
        input_path=inp, output_path=out,
        do_upscale=bool(args.get("upscale", True)),
        upscale_backend=backend,
        upscale_model=args.get("upscale_model", "x4plus"),
        outscale=float(args.get("scale", 2.0)),
        tile=int(args.get("tile", 0)),
        cugan_tile_mode=int(args.get("cugan_tile_mode", 0)),
        do_face=bool(args.get("face", False)),
        face_weight=float(args.get("face_weight", 0.5)),
        do_interp=mult > 1, interp_mult=mult,
        crf=int(args.get("crf", 18)),
        codec=args.get("codec", "libx264"),
        device=args.get("device", "cuda"),
        model_backend=args.get("model_backend", "auto"),
        enabled_plugins=list(dict.fromkeys(enabled)),
        plugin_params=plugin_params,
    )

    def emit(done: int, total: int, note: str):
        _log(f"[enhance] {note}")

    try:
        result = run(cfg, progress=emit)
    except Exception as e:
        return _tool_text(f"enhance failed: {type(e).__name__}: {e}\n{traceback.format_exc()[-1200:]}", True)
    from .core.video_io import probe as _probe
    try:
        oi = _probe(out)
        summary = {"width": oi.width, "height": oi.height, "fps": oi.fps,
                   "frames": oi.n_frames, "has_audio": oi.has_audio}
    except Exception:
        summary = {}
    return _tool_text(
        f"enhanced -> {result}\noutput: {json.dumps(summary, ensure_ascii=False)}\n"
        f"backend={backend} scale={cfg.outscale} interp=x{mult} face={cfg.do_face} "
        f"hdr={'builtin.hdr' in enabled} stabilize={'builtin.stabilize' in enabled}"
    )


def tool_preview_frame(args: dict) -> dict:
    from .core.config import EnhanceConfig, OUTPUT_DIR
    from .core.preview import render_preview
    inp = args.get("input") or args.get("video")
    if not inp:
        raise ValueError("preview_frame needs 'input' video path")
    t = float(args.get("time", 0.0))
    backend = args.get("backend", "esrgan")
    cfg = EnhanceConfig(
        upscale_backend=backend, outscale=float(args.get("scale", 2.0)),
        do_face=bool(args.get("face", False)),
        enabled_plugins=(args.get("plugins") or []) + (
            ["builtin.onnx_superres"] if backend == "model" else []),
        plugin_params=dict(args.get("plugin_params") or {}),
        model_backend=args.get("model_backend", "auto"),
    )
    try:
        orig, enh = render_preview(inp, t, cfg)
    except Exception as e:
        return _tool_text(f"preview failed: {type(e).__name__}: {e}", True)
    import cv2
    import numpy as np
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(inp).stem)[:32]
    op = OUTPUT_DIR / f"mcp_preview_{tag}_orig.png"
    ep = OUTPUT_DIR / f"mcp_preview_{tag}_enh.png"
    cp = OUTPUT_DIR / f"mcp_preview_{tag}_compare.png"
    cv2.imwrite(str(op), orig)
    cv2.imwrite(str(ep), enh)
    # side-by-side, fit same height
    dh = 360
    def fit(im):
        h, w = im.shape[:2]
        return cv2.resize(im, (int(w * dh / h), dh))
    cmp = np.hstack([fit(orig), fit(enh)])
    cv2.imwrite(str(cp), cmp)
    return _tool_text(
        f"preview @ {t}s: orig {orig.shape[1]}x{orig.shape[0]} -> enh {enh.shape[1]}x{enh.shape[0]}\n"
        f"original: {op}\nenhanced: {ep}\ncompare(left=orig,right=enh): {cp}"
    )


def tool_write_plugin(args: dict) -> dict:
    """Scaffold a real, loadable plugin (frame / model / video) into USER_PLUGINS_DIR."""
    from .core.config import USER_PLUGINS_DIR
    pid = _safe_plugin_id(args.get("id", ""))
    kind = args.get("kind", "frame")
    if kind not in ("frame", "model", "video"):
        raise ValueError("kind must be frame|model|video")
    name = args.get("name") or pid
    desc = args.get("description", "Agent-authored myvideoai plugin")
    provides = args.get("provides", "upscale" if kind == "model" else "")
    backends = args.get("backends", ["cuda", "cpu"] if kind == "model" else [])
    params = args.get("params", {} if kind != "model" else {"tile": 0})

    d = USER_PLUGINS_DIR / pid.replace(".", "_")
    d.mkdir(parents=True, exist_ok=True)
    manifest = {
        "id": pid, "name": name, "version": "1.0.0", "kind": kind,
        "description": desc, "author": args.get("author", "agent"),
        "license": args.get("license", ""), "entry": "plugin.py::Plugin",
        "enabled": bool(args.get("enabled", True)), "order": int(args.get("order", 100)),
        "params": params, "backends": backends, "provides": provides,
    }
    (d / "plugin.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    if kind == "frame":
        code = '''"""frame hook: transform one BGR frame, return a BGR frame (same shape)."""
import cv2
import numpy as np

class Plugin:
    def __init__(self):
        self.params = {}

    def setup(self, ctx):
        self.params = ctx.params

    def process(self, frame: np.ndarray) -> np.ndarray:
        amount = float(self.params.get("amount", 1.3))
        # TODO: replace with your effect. Keep output shape == input shape.
        blur = cv2.GaussianBlur(frame, (0, 0), 1.2)
        return cv2.addWeighted(frame, amount, blur, 1.0 - amount, 0)

    def teardown(self):
        pass
'''
    elif kind == "model":
        code = '''"""model hook: super-res a frame; MAY change dimensions."""
import numpy as np
import cv2

class Plugin:
    def __init__(self):
        self.params = {}
        self.ctx = None

    def setup(self, ctx):
        self.ctx = ctx
        self.params = ctx.params
        # TODO: load your model here (torch / onnxruntime) using ctx.torch_device or ctx.ort_provider.
        # Weights live in ctx.weights_dir; use myvideoai.core.weights.ensure_weights to fetch.

    def process(self, frame: np.ndarray, outscale: float) -> np.ndarray:
        # TODO: run inference on `frame` (BGR uint8) and return a BGR frame.
        h, w = frame.shape[:2]
        target = (int(w * outscale), int(h * outscale))
        return cv2.resize(frame, target, interpolation=cv2.INTER_LANCZOS4)

    def teardown(self):
        pass
'''
    else:  # video
        code = '''"""video hook: transform a whole video file with ffmpeg (two-pass OK)."""
import subprocess
from pathlib import Path

class Plugin:
    def __init__(self):
        self.params = {}

    def setup(self, ctx):
        self.params = ctx.params
        self.ffmpeg = ctx.ffmpeg
        self.work = Path(ctx.work_dir)

    def process(self, in_path: str, out_path: str, ctx) -> str:
        # TODO: build your ffmpeg command. Use ctx.work for intermediates.
        cmd = [self.ffmpeg, "-y", "-v", "error", "-i", in_path,
               "-c", "copy", out_path]
        subprocess.run(cmd, check=True)
        return out_path

    def teardown(self):
        pass
'''
    (d / "plugin.py").write_text(code, encoding="utf-8")
    return _tool_text(
        f"plugin scaffolded.\nid={pid} kind={kind} provides={provides or '-'}\n"
        f"dir={d}\nfiles: plugin.json, plugin.py\n"
        f"enable it with: --plugins {pid}  (or plugin_params / GUI plugin list)\n"
        f"edit plugin.py's TODO markers, then call validate_plugin(id='{pid}')."
    )


def tool_register_model(args: dict) -> dict:
    """Register a drop-in ONNX super-res model without writing core code."""
    from .core.config import USER_PLUGINS_DIR, WEIGHTS_DIR
    pid = _safe_plugin_id(args.get("id", ""))
    model_file = args.get("model_file", "")
    if not model_file:
        raise ValueError("register_model needs 'model_file' (filename in models/weights)")
    d = USER_PLUGINS_DIR / pid.replace(".", "_")
    d.mkdir(parents=True, exist_ok=True)
    src = args.get("source_path", "")
    if src and Path(src).exists():
        import shutil
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, WEIGHTS_DIR / Path(src).name)
        model_file = Path(src).name
    model_url = args.get("model_url", "")
    manifest = {
        "id": pid, "name": args.get("name", pid), "version": "1.0.0",
        "kind": "model", "description": args.get("description", "Agent-registered drop-in ONNX SR model"),
        "author": args.get("author", "agent"), "license": args.get("license", ""),
        "entry": "myvideoai.plugins.builtin.onnx_superres::ONNXSuperResPlugin",
        "entry_file": "", "enabled": True, "order": int(args.get("order", 50)),
        "provides": "upscale", "backends": args.get("backends", ["cuda", "directml", "cpu"]),
        "params": {
            "model_file": model_file,
            "model_url": model_url,
            "input_name": args.get("input_name", "input"),
            "output_name": args.get("output_name", "output"),
            "native_scale": int(args.get("native_scale", 4)),
            "tile": int(args.get("tile", 256)),
            "norm": args.get("norm", "0to1"),
            "layout": args.get("layout", "nchw"),
        },
    }
    # registry loads entry_module::Class relative to the plugin dir. For a shared
    # builtin class we write a tiny re-export module so discovery stays uniform.
    (d / "plugin.py").write_text(
        "from myvideoai.plugins.builtin.onnx_superres.onnx_superres import ONNXSuperResPlugin as Plugin  # noqa\n",
        encoding="utf-8",
    )
    manifest["entry"] = "plugin.py::Plugin"
    (d / "plugin.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    exists = (WEIGHTS_DIR / model_file).exists()
    return _tool_text(
        f"model plugin registered.\nid={pid}\nmodel_file={model_file} "
        f"({'present in weights' if exists else 'NOT FOUND — supply model_url or source_path'})\n"
        f"native_scale={manifest['params']['native_scale']} tile={manifest['params']['tile']}\n"
        f"dir={d}\nuse with: --backend model --plugins {pid}\n"
        f"NOTE: ONNX model must be exported with dynamic_axes for H/W, else real frame sizes are rejected."
    )


def tool_validate_plugin(args: dict) -> dict:
    pid = args.get("id") or args.get("plugin")
    if not pid:
        raise ValueError("validate_plugin needs 'id'")
    from .core.config import PLUGINS_DIR, USER_PLUGINS_DIR
    from .plugins import discover
    found = [m for m in discover(PLUGINS_DIR) + discover(USER_PLUGINS_DIR) if m.id == pid]
    if not found:
        return _tool_text(f"plugin not found: {pid}", True)
    m = found[0]
    checks = [f"id={m.id} kind={m.kind} entry={m.entry}"]
    try:
        from .plugins.registry import PluginRegistry
        reg = PluginRegistry([m])
        checks.append("manifest+entry: OK (importable)")
    except Exception as e:
        return _tool_text(f"VALIDATION FAILED\n" + "\n".join(checks) + f"\nerror: {e}", True)
    return _tool_text("VALID\n" + "\n".join(checks) +
                      f"\nprovides={m.provides or '-'} backends={m.backends}")


def tool_get_info(args: dict) -> dict:
    return _tool_text(
        "myvideoai — AI video quality restoration\n"
        "capabilities: Real-ESRGAN/Real-CUGAN/drop-in ONNX/torch super-res, "
        "GFPGAN face restore, RIFE v4.6 interpolation, SDR->HDR10, video stabilize, "
        "timeline preview, batch queue, pluggable frame/model/video hooks.\n"
        "backends: CUDA / DirectML(cross-GPU) / CPU (auto-detect).\n"
        "MCP tools: probe_video, list_backends, list_plugins, enhance_video, "
        "preview_frame, write_plugin, register_model, validate_plugin, get_info.\n"
        "MCP resources: myvideoai://docs/plugin-spec, myvideoai://docs/model-plugin-design, "
        "myvideoai://docs/readme."
    )


TOOLS = [
    {"name": "get_info", "description": "Overview of myvideoai capabilities and how to drive it via MCP.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "probe_video", "description": "Probe a video: width/height/fps/frame count/duration/codec/audio.",
     "inputSchema": {"type": "object", "properties": {"video": {"type": "string"}}, "required": ["video"]}},
    {"name": "list_backends", "description": "Detect available inference backends (CUDA/DirectML/CPU) and what auto picks.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "list_plugins", "description": "List all discovered plugins with id/kind/enabled/params/provides.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "enhance_video", "description": "Enhance a video end-to-end: upscale (esrgan/cugan/model), interpolation, face, HDR, stabilization and extra plugins. Returns output path + probe summary.",
     "inputSchema": {"type": "object", "properties": {
         "input": {"type": "string"}, "output": {"type": "string"},
         "backend": {"type": "string", "enum": ["esrgan", "cugan", "model"], "default": "esrgan"},
         "scale": {"type": "number", "default": 2.0},
         "upscale": {"type": "boolean", "default": True},
         "upscale_model": {"type": "string", "default": "x4plus"},
         "tile": {"type": "integer", "default": 0},
         "cugan_tile_mode": {"type": "integer", "default": 0},
         "interp": {"type": "integer", "enum": [1, 2, 4], "default": 1},
         "face": {"type": "boolean", "default": False},
         "face_weight": {"type": "number", "default": 0.5},
         "hdr": {"type": "boolean", "default": False},
         "stabilize": {"type": "boolean", "default": False},
         "plugins": {"type": "array", "items": {"type": "string"}},
         "plugin_params": {"type": "object"},
         "model_backend": {"type": "string", "enum": ["auto", "cuda", "directml", "cpu"], "default": "auto"},
         "device": {"type": "string", "default": "cuda"},
         "crf": {"type": "integer", "default": 18},
         "codec": {"type": "string", "default": "libx264"},
     }, "required": ["input"]}},
    {"name": "preview_frame", "description": "Render a timeline frame through the same model chain as the full pipeline; save original/enhanced/side-by-side PNGs.",
     "inputSchema": {"type": "object", "properties": {
         "input": {"type": "string"}, "time": {"type": "number", "default": 0.0},
         "backend": {"type": "string", "enum": ["esrgan", "cugan", "model"], "default": "esrgan"},
         "scale": {"type": "number", "default": 2.0}, "face": {"type": "boolean", "default": False},
         "plugins": {"type": "array", "items": {"type": "string"}},
         "plugin_params": {"type": "object"},
         "model_backend": {"type": "string", "enum": ["auto", "cuda", "directml", "cpu"], "default": "auto"},
     }, "required": ["input"]}},
    {"name": "write_plugin", "description": "Create a new loadable plugin (frame/model/video) with a working scaffold. Returns the directory to edit.",
     "inputSchema": {"type": "object", "properties": {
         "id": {"type": "string"}, "kind": {"type": "string", "enum": ["frame", "model", "video"]},
         "name": {"type": "string"}, "description": {"type": "string"},
         "provides": {"type": "string"}, "backends": {"type": "array", "items": {"type": "string"}},
         "params": {"type": "object"}, "enabled": {"type": "boolean"},
         "order": {"type": "integer"}, "author": {"type": "string"}, "license": {"type": "string"},
     }, "required": ["id", "kind"]}},
    {"name": "register_model", "description": "Register a drop-in ONNX super-res model as a model plugin (zero core code). Supports copying a local .onnx into the weights dir or downloading via model_url.",
     "inputSchema": {"type": "object", "properties": {
         "id": {"type": "string"}, "model_file": {"type": "string"},
         "source_path": {"type": "string"}, "model_url": {"type": "string"},
         "name": {"type": "string"}, "description": {"type": "string"},
         "native_scale": {"type": "integer", "default": 4}, "tile": {"type": "integer", "default": 256},
         "input_name": {"type": "string", "default": "input"},
         "output_name": {"type": "string", "default": "output"},
         "norm": {"type": "string", "default": "0to1"}, "layout": {"type": "string", "default": "nchw"},
         "backends": {"type": "array", "items": {"type": "string"}},
         "order": {"type": "integer", "default": 50},
     }, "required": ["id", "model_file"]}},
    {"name": "validate_plugin", "description": "Validate that a plugin id resolves and its manifest+entry import cleanly.",
     "inputSchema": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}},
]

TOOL_DISPATCH = {
    "get_info": tool_get_info,
    "probe_video": tool_probe_video,
    "list_backends": tool_list_backends,
    "list_plugins": tool_list_plugins,
    "enhance_video": tool_enhance_video,
    "preview_frame": tool_preview_frame,
    "write_plugin": tool_write_plugin,
    "register_model": tool_register_model,
    "validate_plugin": tool_validate_plugin,
}

RESOURCES = {
    "myvideoai://docs/plugin-spec": ("text/markdown", "myvideoai/docs/PLUGIN_SPEC.md"),
    "myvideoai://docs/model-plugin-design": ("text/markdown", "myvideoai/docs/MODEL_PLUGIN_DESIGN.md"),
    "myvideoai://docs/readme": ("text/markdown", "myvideoai/README.md"),
}

ROOT = Path(__file__).resolve().parent.parent  # repository root (parent of the package)


def _handle(msg: dict) -> dict | None:
    rid = msg.get("id")
    method = msg.get("method", "")
    params = msg.get("params") or {}

    if method == "initialize":
        return _ok({
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}, "resources": {}},
            "serverInfo": {"name": "myvideoai", "version": "1.0.0"},
        }, rid)
    if method in ("notifications/initialized", "initialized"):
        return None  # notification, no response
    if method == "ping":
        return _ok({}, rid)
    if method == "tools/list":
        return _ok({"tools": TOOLS}, rid)
    if method == "tools/call":
        name = params.get("name", "")
        args = params.get("arguments") or {}
        fn = TOOL_DISPATCH.get(name)
        if fn is None:
            return _ok(_tool_text(f"unknown tool: {name}", True), rid)
        try:
            # MCP responses wrap tool output in a "result" object.  Forgetting
            # this wrapper is why a raw {"content":...} broke strict clients.
            return _ok(fn(args), rid)
        except Exception as e:
            _log(traceback.format_exc())
            return _ok(_tool_text(f"{type(e).__name__}: {e}", True), rid)
    if method == "resources/list":
        return _ok({"resources": [{"uri": u, "name": u.split("/")[-1], "mimeType": mt}
                                   for u, (mt, _) in RESOURCES.items()]}, rid)
    if method == "resources/read":
        uri = params.get("uri", "")
        if uri not in RESOURCES:
            return _err(-32002, f"unknown resource: {uri}", rid)
        mt, rel = RESOURCES[uri]
        p = ROOT / rel
        if not p.exists():
            return _err(-32002, f"resource file missing: {rel}", rid)
        return _ok({"contents": [{"uri": uri, "mimeType": mt, "text": p.read_text(encoding="utf-8")}]}, rid)

    return _err(-32601, f"method not found: {method}", rid)


def main() -> int:
    _log(f"[myvideoai MCP] stdio server ready; {len(TOOLS)} tools, {len(RESOURCES)} resources")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception as e:
            _send(_err(-32700, f"parse error: {e}", None))
            continue
        # batch support
        if isinstance(msg, list):
            for m in msg:
                resp = _handle(m)
                if resp is not None:
                    _send(resp)
        else:
            resp = _handle(msg)
            if resp is not None:
                _send(resp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
