"""Headless test: GUI model-file picker -> cfg -> model chain -> end to end."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
app = QApplication.instance() or QApplication([])

from myvideoai.gui.app import MainWindow

w = MainWindow()

# --- test 1: picker row visibility follows backend (isHidden() works headless) ---
assert w.dropin_row.isHidden(), "row should start hidden"
w.backend.setCurrentIndex(2)
assert not w.dropin_row.isHidden(), "row should show for model backend"

# --- test 2: _make_cfg maps the GUI controls into plugin_params ---
w.onnx_model.setText("test_sr_2x.onnx")
w.onnx_native.setValue(2)
w.onnx_tile.setValue(0)
w.scale.setValue(2.0)
w.model_backend.setCurrentText("auto")
cfg = w._make_cfg("dummy_in.mp4", "dummy_out.mp4")
assert cfg.upscale_backend == "model", cfg.upscale_backend
assert cfg.model_backend == "auto"
assert cfg.enabled_plugins and cfg.enabled_plugins[0] == "builtin.onnx_superres", cfg.enabled_plugins
pp = cfg.plugin_params["builtin.onnx_superres"]
assert pp["model_file"] == "test_sr_2x.onnx", pp
assert pp["native_scale"] == 2 and pp["tile"] == 0, pp
print("[PASS] GUI picker -> cfg mapping:", pp)

# --- test 3: that cfg drives a real end-to-end drop-in run ---
from myvideoai.core.pipeline import run
src = "myvideoai/samples/sample_lowres.mp4"
out = "myvideoai/output/gui_dropin_test.mp4"
cfg.input_path = src
cfg.output_path = out
cfg.do_face = False
cfg.do_interp = False
cfg.enabled_plugins = ["builtin.onnx_superres"]  # no hdr/stabilize in this smoke
run(cfg)
print("[PASS] end-to-end GUI-driven drop-in:", out)
