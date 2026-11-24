"""Plugin discovery, validation, ordering and execution.

Design rules (Working Backwards from the plugin author):
- Zero glue: drop a folder with plugin.json + module.py, it just works.
- Fail loud: bad manifests never silently skip; errors name the plugin id.
- Deterministic order: ``order`` then ``after``/``before`` topological tie-break.
- Sandboxed capability: plugins only see PluginContext, never the pipeline guts.
"""
from __future__ import annotations

import importlib.util
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .spec import (PluginManifest, FramePlugin, VideoPlugin, ModelPlugin,
                   PluginContext, PluginError)


@dataclass
class LoadedPlugin:
    manifest: PluginManifest
    impl: object
    directory: Path


def _load_entry(manifest: PluginManifest, directory: Path):
    if not manifest.entry:
        raise PluginError(f"{manifest.id}: no entry declared")
    module_part, _, cls_name = manifest.entry.partition("::")
    module_path = directory / module_part
    if not module_path.suffix:
        module_path = module_path.with_suffix(".py")
    if not module_path.exists():
        raise PluginError(f"{manifest.id}: entry module not found: {module_path}")
    spec = importlib.util.spec_from_file_location(f"mvp_{manifest.id.replace('.', '_')}", module_path)
    if spec is None or spec.loader is None:
        raise PluginError(f"{manifest.id}: cannot import {module_path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise PluginError(f"{manifest.id}: import failed: {type(e).__name__}: {e}") from e
    if not cls_name:
        raise PluginError(f"{manifest.id}: entry needs ::ClassName")
    obj = getattr(module, cls_name, None)
    if obj is None:
        raise PluginError(f"{manifest.id}: class {cls_name} not found in {module_part}")
    return obj()


def discover(root: str | Path) -> list[PluginManifest]:
    """Scan a plugins root (e.g. myvideoai/plugins) for plugin.json manifests.

    Layout expected::

        <root>/<group>/<name>/plugin.json + module
        <root>/<name>/plugin.json + module        (also accepted)
    """
    root = Path(root)
    found: list[PluginManifest] = []
    if not root.exists():
        return found
    for manifest_path in sorted(root.glob("**/plugin.json")):
        try:
            m = PluginManifest.from_file(manifest_path)
        except PluginError:
            raise
        m.__dict__["_dir"] = manifest_path.parent  # stash dir for loader
        found.append(m)
    return found


class PluginRegistry:
    """Load enabled plugins, order them, and run the two hook chains."""

    def __init__(self, manifests: list[PluginManifest]):
        self._manifests = [m for m in manifests if m.enabled]
        self._by_id = {m.id: m for m in self._manifests}
        dup = [mid for mid in self._by_id if sum(1 for m in self._manifests if m.id == mid) > 1]
        if dup:
            raise PluginError(f"duplicate plugin ids: {dup}")
        self._loaded: dict[str, LoadedPlugin] = {}
        self._frame_order: list[str] = []
        self._model_order: list[str] = []
        self._video_order: list[str] = []
        self._validate_and_load()

    # -- validation / ordering ------------------------------------------------
    def _validate_and_load(self):
        for m in self._manifests:
            for dep in m.requires:
                if dep not in self._by_id:
                    raise PluginError(f"{m.id}: requires missing plugin '{dep}'")
        frame = [m for m in self._manifests if m.kind == "frame"]
        model = [m for m in self._manifests if m.kind == "model"]
        video = [m for m in self._manifests if m.kind == "video"]
        self._frame_order = [m.id for m in self._order(frame)]
        self._model_order = [m.id for m in self._order(model)]
        self._video_order = [m.id for m in self._order(video)]
        for m in self._manifests:
            directory = Path(m.__dict__.get("_dir", "."))
            impl = _load_entry(m, directory)
            self._loaded[m.id] = LoadedPlugin(m, impl, directory)
            # verify the required method exists
            required = "process" if m.kind in ("frame", "model", "video") else None
            if required and not hasattr(impl, required):
                raise PluginError(f"{m.id}: {type(impl).__name__} has no {required}()")

    def _order(self, items: list[PluginManifest]) -> list[PluginManifest]:
        """Stable order by `order`, then satisfy after/before via repeated passes."""
        items = sorted(items, key=lambda m: (m.order, m.id))
        ids = {m.id for m in items}
        for m in items:
            for dep in m.after + m.before:
                if dep not in ids:
                    # dependency may be disabled / absent — ignore, order is best-effort
                    continue
        # simple topological relaxation on after/before constraints (max len passes)
        for _ in range(len(items)):
            changed = False
            pos = {m.id: i for i, m in enumerate(items)}
            for m in items:
                target = pos[m.id]
                for dep in m.after:
                    if dep in pos and pos[dep] > target:
                        target = pos[dep]
                for dep in m.before:
                    if dep in pos and pos[dep] < target:
                        target = pos[dep]
                if target != pos[m.id]:
                    items.remove(m)
                    items.insert(target, m)
                    changed = True
            if not changed:
                break
        return items

    # -- introspection --------------------------------------------------------
    def list_plugins(self, kind: Optional[str] = None) -> list[dict]:
        result = []
        for order, kind_name in (("frame", self._frame_order), ("model", self._model_order),
                                 ("video", self._video_order)):
            if kind and kind != order:
                continue
            for pid in kind_name:
                m = self._loaded[pid].manifest
                result.append({"id": m.id, "name": m.name, "kind": m.kind,
                               "version": m.version, "description": m.description})
        return result

    def has(self, plugin_id: str) -> bool:
        return plugin_id in self._loaded

    def model_manifests(self) -> list[PluginManifest]:
        """Loaded model-plugin manifests in execution order."""
        return [self._loaded[pid].manifest for pid in self._model_order]

    # -- execution ------------------------------------------------------------
    def frame_chain(self, ctx_factory):
        """Build a frame transformation chain ONCE (setup called once per plugin).

        Returns ``(apply, close)`` where apply(frame) -> frame and close() tears down.
        Callers should build the chain before the frame loop and close it after.
        """
        chain = []
        for pid in self._frame_order:
            lp = self._loaded[pid]
            ctx = ctx_factory(lp.manifest)
            impl = lp.impl
            if hasattr(impl, "setup"):
                impl.setup(ctx)
            chain.append((pid, impl))

        def apply(frame: np.ndarray) -> np.ndarray:
            for pid, impl in chain:
                out = impl.process(frame)
                if out is None:
                    raise PluginError(f"{pid}: process() returned None")
                if out.shape != frame.shape:
                    raise PluginError(f"{pid}: changed frame shape {frame.shape} -> {out.shape}")
                frame = out
            return frame

        def close():
            for _, impl in chain:
                if hasattr(impl, "teardown"):
                    impl.teardown()

        return apply, close

    def run_frame(self, frame: np.ndarray, ctx_factory) -> np.ndarray:
        """Convenience: one-shot frame chain (setup/process/teardown per call)."""
        apply, close = self.frame_chain(ctx_factory)
        try:
            return apply(frame)
        finally:
            close()

    def model_chain(self, ctx_factory):
        """Build a model transformation chain ONCE (setup called once per model).

        Model hooks differ from frame hooks: ``process(frame, outscale)`` may
        change dimensions (super-resolution). Returns ``(apply, close)`` where
        ``apply(frame, outscale)`` → frame. Models run before frame hooks.
        """
        chain = []
        for pid in self._model_order:
            lp = self._loaded[pid]
            ctx = ctx_factory(lp.manifest)
            impl = lp.impl
            if hasattr(impl, "setup"):
                impl.setup(ctx)
            chain.append((pid, impl))

        def apply(frame: np.ndarray, outscale: float) -> np.ndarray:
            for pid, impl in chain:
                out = impl.process(frame, outscale)
                if out is None:
                    raise PluginError(f"{pid}: process() returned None")
                frame = out
            return frame

        def close():
            for _, impl in chain:
                if hasattr(impl, "teardown"):
                    impl.teardown()

        return apply, close

    def run_video(self, in_path: str, out_path: str, ctx_factory) -> str:
        cur = in_path
        chain = self._video_order
        for i, pid in enumerate(chain):
            lp = self._loaded[pid]
            ctx = ctx_factory(lp.manifest)
            target = out_path if i == len(chain) - 1 else str(
                Path(ctx.work_dir or tempfile.gettempdir()) / f"_mvp_{i}_{pid.replace('.', '_')}.mp4")
            impl = lp.impl
            if hasattr(impl, "setup"):
                impl.setup(ctx)
            try:
                cur = impl.process(cur, target, ctx)
            finally:
                if hasattr(impl, "teardown"):
                    impl.teardown()
            if not cur or not Path(cur).exists():
                raise PluginError(f"{pid}: did not produce output {cur}")
        # no video plugins: copy is caller's job; here just report
        return cur if chain else in_path
