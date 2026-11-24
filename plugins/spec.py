"""Plugin contract definitions for myvideoai.

A plugin is a directory containing a manifest (``plugin.json``) and a python
module. The manifest is pure data; the module implements one of the three hooks.
The design keeps the surface tiny — a declarative manifest plus a plain code
entry, with no dependency injection framework and no bytecode hooks.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

import numpy as np


class PluginError(RuntimeError):
    """Raised when a plugin manifest or implementation is invalid."""


# ---------------------------------------------------------------------------
# Manifest — declarative, data-only, no imports
# ---------------------------------------------------------------------------
@dataclass
class PluginManifest:
    id: str                       # globally unique, e.g. "builtin.hdr"
    name: str                     # human label
    version: str = "1.0.0"
    kind: str = "frame"           # "frame" | "video" | "model"
    description: str = ""
    author: str = ""
    license: str = ""
    entry: str = ""               # "module.py::ClassName" relative to plugin dir
    enabled: bool = True
    order: int = 100              # lower runs earlier within a kind
    after: list[str] = field(default_factory=list)   # ids this plugin runs after
    before: list[str] = field(default_factory=list)  # ids this plugin runs before
    params: dict[str, Any] = field(default_factory=dict)  # default params (CLI/GUI override)
    requires: list[str] = field(default_factory=list)     # plugin ids that must be present
    backends: list[str] = field(default_factory=list)     # model hook: supported backends
    provides: str = ""            # model hook: what it provides, e.g. "upscale"

    @classmethod
    def from_file(cls, path: Path) -> "PluginManifest":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            raise PluginError(f"bad manifest {path}: {e}") from e
        known = set(cls.__dataclass_fields__)
        unknown = set(data) - known
        if unknown:
            raise PluginError(f"{path}: unknown manifest keys {sorted(unknown)}")
        if "id" not in data or "name" not in data:
            raise PluginError(f"{path}: manifest needs id and name")
        if data.get("kind", "frame") not in ("frame", "video", "model"):
            raise PluginError(f"{path}: kind must be 'frame', 'video' or 'model'")
        return cls(**data)


# ---------------------------------------------------------------------------
# Runtime context — the only capability surface given to plugins
# ---------------------------------------------------------------------------
@dataclass
class PluginContext:
    """Everything a plugin is allowed to touch. Injected, not global."""

    work_dir: str = ""            # temp dir for intermediate files (videos)
    device: str = "cuda"          # backend-resolved device (ONNX/DirectML semantics)
    torch_device: str = "cuda"    # cfg.device — torch model plugins must use this
    half: bool = True
    backend: str = "auto"         # resolved backend name (cuda/directml/cpu)
    ort_provider: str = ""        # resolved onnxruntime provider name
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"
    params: dict[str, Any] = field(default_factory=dict)  # merged params for THIS plugin
    weights_dir: str = ""         # shared model weights directory
    emit: Optional[Callable[[str], None]] = None          # optional log callback

    def log(self, msg: str):
        if self.emit:
            self.emit(msg)

    def device_for(self, runtime: str) -> str:
        """Resolve the device for a given plugin runtime.

        ``torch`` runtimes must use the torch device (``cuda``/``cpu``), while
        ONNX/DirectML runtimes use the backend-resolved provider device.
        """
        if runtime == "torch":
            return self.torch_device or self.device
        return self.device


class FramePlugin(Protocol):
    """frame-hook contract: transform one BGR frame, return a BGR frame.

    Implement one or both of ``setup`` / ``process`` / ``teardown``.
    ``process`` is required. Shape/dtype of output must match input.
    """

    def setup(self, ctx: PluginContext) -> None: ...
    def process(self, frame: np.ndarray) -> np.ndarray: ...
    def teardown(self) -> None: ...


class VideoPlugin(Protocol):
    """video-hook contract: transform a whole video file into another.

    Called once after encoding. ``process`` receives input path, output path and
    context; it must write a valid video to ``out_path`` and return that path.
    Two-pass filters (e.g. stabilization) use ``ctx.work_dir`` for analysis files.
    """

    def setup(self, ctx: PluginContext) -> None: ...
    def process(self, in_path: str, out_path: str, ctx: PluginContext) -> str: ...
    def teardown(self) -> None: ...


class ModelPlugin(Protocol):
    """model-hook contract: image-to-image transform that MAY change dimensions.

    Unlike frame hooks (shape-preserving filters), model hooks are for
    super-resolution / restoration: ``process`` takes a BGR frame and an
    ``outscale`` and returns a (possibly larger) BGR frame. One instance is
    loaded once via ``setup(ctx)`` using ``ctx.backend`` / ``ctx.ort_provider``,
    then reused across all frames via ``process``.
    """

    def setup(self, ctx: PluginContext) -> None: ...
    def process(self, frame: np.ndarray, outscale: float) -> np.ndarray: ...
    def teardown(self) -> None: ...
