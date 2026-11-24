"""Smoke + functional tests for the plugin system."""
import sys, os, tempfile, shutil
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent))

from myvideoai.plugins import discover, PluginRegistry, PluginContext


def test_discover_and_contract():
    ms = discover(ROOT / "plugins")
    ids = [m.id for m in ms]
    assert "builtin.hdr" in ids and "builtin.stabilize" in ids and "examples.unsharp" in ids, ids
    reg = PluginRegistry(ms)
    kinds = {p["id"]: p["kind"] for p in reg.list_plugins()}
    assert kinds["builtin.hdr"] == "video"
    assert kinds["builtin.stabilize"] == "video"
    # unsharp is enabled=false, so not loaded
    assert "examples.unsharp" not in kinds
    print("OK discover+contract", ids)


def test_frame_plugin_enabled():
    ms = discover(ROOT / "plugins")
    for m in ms:
        if m.id == "examples.unsharp":
            m.enabled = True
    reg = PluginRegistry(ms)
    frame = np.zeros((8, 8, 3), np.uint8) + 100
    ctx = PluginContext(params={"amount": 0.5})
    out = reg.run_frame(frame, lambda m: PluginContext(params={**m.params, "amount": 0.5}))
    assert out.shape == frame.shape
    print("OK frame plugin chain", out.shape)


def test_video_plugin_order():
    ms = discover(ROOT / "plugins")
    reg = PluginRegistry(ms)
    assert reg._video_order == ["builtin.stabilize", "builtin.hdr"], reg._video_order
    print("OK video plugin order", reg._video_order)


def test_model_backend_detection():
    from myvideoai.core.model_backend import detect, pick
    names = [c.name for c in detect()]
    assert "cpu" in names and names[0] in ("cuda", "directml"), names
    assert pick("auto").name == names[0]
    assert pick("no-such-backend").name == "cpu"  # graceful degrade
    print("OK model backends", names)


def test_model_plugin_dropin():
    """One ONNX model, all backends — zero core-code changes."""
    from myvideoai.core.model_backend import pick
    from myvideoai.core.config import WEIGHTS_DIR
    fixture = Path(WEIGHTS_DIR) / "test_sr_2x.onnx"
    if not fixture.exists():
        print("SKIP model dropin (no test_sr_2x.onnx fixture)")
        return
    ms = [m for m in discover(ROOT / "plugins") if m.id == "builtin.onnx_superres"]
    for m in ms:
        m.enabled = True
        m.params.update({"model_file": "test_sr_2x.onnx", "native_scale": 2, "tile": 0})
    reg = PluginRegistry(ms)
    assert any(x.provides == "upscale" for x in reg.model_manifests())
    for be in ("cuda", "directml", "cpu"):
        ch = pick(be)
        def cf(man, ch=ch):
            return PluginContext(device=ch.device, backend=ch.name,
                                 ort_provider=ch.ort_provider,
                                 weights_dir=str(WEIGHTS_DIR),
                                 params=dict(man.params), emit=None)
        apply, close = reg.model_chain(cf)
        frame = np.zeros((32, 32, 3), np.uint8)
        out = apply(frame, 2.0); close()
        assert out.shape == (64, 64, 3), (be, out.shape)
    print("OK model dropin across cuda/directml/cpu")


if __name__ == "__main__":
    test_discover_and_contract()
    test_frame_plugin_enabled()
    test_video_plugin_order()
    test_model_backend_detection()
    test_model_plugin_dropin()
    print("ALL PLUGIN TESTS PASSED")
