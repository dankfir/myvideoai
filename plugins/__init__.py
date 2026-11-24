"""myvideoai plugin system.

Three concepts shape the system:

1. Manifest (plugin.yaml / plugin.json) — declarative metadata + capability declaration.
2. Hook — the contract a plugin implements. Three kinds:
     - ``frame``: numpy BGR frame -> frame, shape-preserving (streaming pipeline)
     - ``model``: BGR frame -> (possibly larger) BGR frame, e.g. super-resolution
     - ``video``: whole-file ffmpeg filtergraph (runs as a post-encode pass)
3. Registry — discovers, validates, orders (by ``after``/``before``) and runs plugins.

Built-in features (SDR->HDR, stabilization) are themselves plugins, dogfooding
the very contract third parties use — if the contract can't express them, it's
the wrong contract.
"""
from .spec import (PluginManifest, FramePlugin, VideoPlugin, ModelPlugin,
                   PluginContext, PluginError)
from .registry import PluginRegistry, discover

__all__ = [
    "PluginManifest", "FramePlugin", "VideoPlugin", "ModelPlugin", "PluginContext",
    "PluginError", "PluginRegistry", "discover",
]
