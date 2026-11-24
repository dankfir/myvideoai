"""PySide6 GUI for myvideoai.

- Batch queue (multiple input videos, processed sequentially)
- Timeline preview (grab frame at time T -> enhance -> side-by-side compare)
- Upscaler backend (Real-ESRGAN / Real-CUGAN), GFPGAN face toggle
- Background QThread keeps UI responsive
"""
from __future__ import annotations

import os
import sys
import traceback

from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QComboBox, QProgressBar, QTextEdit, QDoubleSpinBox,
    QSpinBox, QListWidget, QCheckBox, QSlider, QSplitter, QListWidgetItem,
    QLineEdit,
)
from PySide6.QtGui import QImage, QPixmap

from ..core.config import EnhanceConfig, OUTPUT_DIR
from ..core.pipeline import run
from ..core.preview import render_preview, grab_frame
from ..core.video_io import probe


class Worker(QThread):
    """Background job: process a batch of configs sequentially."""
    progress = Signal(int, int, str)      # done_frames, total_frames, note
    job_started = Signal(int, int, str)   # job_idx, job_total, basename
    finished_ok = Signal(str)             # last output path
    failed = Signal(str)

    def __init__(self, cfgs: list):
        super().__init__()
        self.cfgs = cfgs
        self._abort = False

    def run(self):
        try:
            for i, cfg in enumerate(self.cfgs, 1):
                if self._abort:
                    break
                self.job_started.emit(i, len(self.cfgs), os.path.basename(cfg.input_path))
                def cb(done, total, extra, i=i):
                    self.progress.emit(done, total, f"[{i}/{len(self.cfgs)}] {extra}")
                run(cfg, progress=cb)
            self.finished_ok.emit(self.cfgs[-1].output_path if self.cfgs else "")
        except Exception:
            self.failed.emit(traceback.format_exc())

    def abort(self):
        self._abort = True


class PreviewWorker(QThread):
    """Background timeline preview render."""
    done = Signal(object, object)  # original_bgr, enhanced_bgr
    failed = Signal(str)

    def __init__(self, video, time_sec, cfg):
        super().__init__()
        self.video, self.time_sec, self.cfg = video, time_sec, cfg

    def run(self):
        try:
            orig, enh = render_preview(self.video, self.time_sec, self.cfg)
            self.done.emit(orig, enh)
        except Exception:
            self.failed.emit(traceback.format_exc())


def _bgr_to_pixmap(bgr) -> QPixmap:
    import cv2
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("myvideoai - AI 画质修复 (批量 + 时间轴预览)")
        self.resize(860, 680)
        self.worker = None
        self.preview_worker = None
        self.inputs = []
        self.duration = 0.0
        self.orig_frame = None
        self.enh_frame = None

        lay = QVBoxLayout(self)

        # ---- file / batch row ----
        frow = QHBoxLayout()
        frow.addWidget(QLabel("输入视频:"))
        self.in_label = QLabel("(未选择，可多选)")
        frow.addWidget(self.in_label, 1)
        b_add = QPushButton("添加视频…")
        b_add.clicked.connect(self.pick_inputs)
        b_clear = QPushButton("清空")
        b_clear.clicked.connect(self.clear_inputs)
        frow.addWidget(b_add)
        frow.addWidget(b_clear)
        lay.addLayout(frow)

        self.queue = QListWidget()
        self.queue.setMaximumHeight(80)
        lay.addWidget(self.queue)

        # ---- options ----
        opt = QHBoxLayout()
        opt.addWidget(QLabel("后端:"))
        self.backend = QComboBox()
        self.backend.addItems(["esrgan (写实)", "cugan (动漫)", "model (drop-in)"])
        opt.addWidget(self.backend)
        opt.addWidget(QLabel("模型后端:"))
        self.model_backend = QComboBox()
        self.model_backend.addItems(["auto", "cuda", "directml", "cpu"])
        opt.addWidget(self.model_backend)
        opt.addWidget(QLabel("倍数:"))
        self.scale = QDoubleSpinBox()
        self.scale.setRange(1.0, 4.0); self.scale.setSingleStep(0.5); self.scale.setValue(2.0)
        opt.addWidget(self.scale)
        opt.addWidget(QLabel("补帧:"))
        self.interp = QComboBox(); self.interp.addItems(["关闭", "x2", "x4"])
        opt.addWidget(self.interp)
        opt.addWidget(QLabel("编码:"))
        self.codec = QComboBox(); self.codec.addItems(["libx264", "h264_nvenc", "hevc_nvenc"])
        opt.addWidget(self.codec)
        lay.addLayout(opt)

        mrow = QHBoxLayout()
        mrow.addWidget(QLabel("ESRGAN 模型:"))
        self.model = QComboBox(); self.model.addItems(["x4plus", "x4plus_anime", "x2plus"])
        mrow.addWidget(self.model)
        self.face = QCheckBox("GFPGAN 人脸修复")
        mrow.addWidget(self.face)
        self.hdr = QCheckBox("SDR→HDR10")
        mrow.addWidget(self.hdr)
        self.stabilize = QCheckBox("视频防抖")
        mrow.addWidget(self.stabilize)
        mrow.addStretch(1)
        lay.addLayout(mrow)

        # ---- drop-in model file (shown only when backend == model) ----
        self.dropin_row = QWidget()
        drow = QHBoxLayout(self.dropin_row)
        drow.setContentsMargins(0, 0, 0, 0)
        drow.addWidget(QLabel("ONNX 模型:"))
        self.onnx_model = QLineEdit()
        self.onnx_model.setReadOnly(True)
        self.onnx_model.setPlaceholderText("(未选择，用插件默认 model.onnx)")
        drow.addWidget(self.onnx_model, 1)
        b_model = QPushButton("选择 .onnx…")
        b_model.clicked.connect(self.pick_onnx_model)
        drow.addWidget(b_model)
        drow.addWidget(QLabel("原生倍数:"))
        self.onnx_native = QSpinBox()
        self.onnx_native.setRange(1, 8); self.onnx_native.setValue(2)
        drow.addWidget(self.onnx_native)
        drow.addWidget(QLabel("推理 tile:"))
        self.onnx_tile = QSpinBox()
        self.onnx_tile.setRange(0, 1024); self.onnx_tile.setValue(0)
        self.onnx_tile.setToolTip("0=整帧推理; 显存不足时设 256/384")
        drow.addWidget(self.onnx_tile)
        lay.addWidget(self.dropin_row)
        self.dropin_row.setVisible(False)
        self.backend.currentIndexChanged.connect(self.on_backend_changed)

        # ---- plugins (discovered at startup; built-in toggles above are plugins too) ----
        prow = QHBoxLayout()
        prow.addWidget(QLabel("附加插件:"))
        self.plugin_list = QListWidget()
        self.plugin_list.setMaximumHeight(54)
        self.plugin_list.setSelectionMode(QListWidget.MultiSelection)
        try:
            from ..plugins import discover
            from ..core.config import PLUGINS_DIR, USER_PLUGINS_DIR
            for m in discover(PLUGINS_DIR) + discover(USER_PLUGINS_DIR):
                it = QListWidgetItem(f"{m.id}  ({m.kind}) — {m.description}")
                it.setData(0x100, m.id)  # Qt.UserRole
                if m.enabled:
                    it.setSelected(True)
                self.plugin_list.addItem(it)
        except Exception as e:
            print(f"[plugin discover failed] {e}")
        prow.addWidget(self.plugin_list, 1)
        lay.addLayout(prow)

        # ---- action row ----
        ar = QHBoxLayout()
        self.run_btn = QPushButton("开始修复 (批量)")
        self.run_btn.clicked.connect(self.start_run)
        ar.addWidget(self.run_btn)
        self.abort_btn = QPushButton("中止")
        self.abort_btn.setEnabled(False)
        self.abort_btn.clicked.connect(self.abort_run)
        ar.addWidget(self.abort_btn)
        lay.addLayout(ar)

        self.bar = QProgressBar()
        lay.addWidget(self.bar)

        # ---- timeline preview ----
        pl = QVBoxLayout()
        pl.addWidget(QLabel("▎时间轴预览 (拖动选时间点 → 预览增强对比)"))
        trow = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1000); self.slider.setValue(0)
        self.slider.valueChanged.connect(self.on_scrub)
        trow.addWidget(self.slider, 1)
        self.time_label = QLabel("0.0s")
        trow.addWidget(self.time_label)
        b_prev = QPushButton("预览此帧")
        b_prev.clicked.connect(self.do_preview)
        trow.addWidget(b_prev)
        pl.addLayout(trow)

        self.preview_lbl = QLabel("(选择视频后拖动时间轴，点预览)")
        self.preview_lbl.setAlignment(Qt.AlignCenter)
        self.preview_lbl.setMinimumHeight(240)
        self.preview_lbl.setStyleSheet("border:1px solid #555;")
        pl.addWidget(self.preview_lbl)
        lay.addLayout(pl)

        self.log = QTextEdit(); self.log.setReadOnly(True); self.log.setMaximumHeight(110)
        lay.addWidget(self.log)

    # ---------- file handling ----------
    def pick_inputs(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择视频(可多选)", "", "视频文件 (*.mp4 *.mov *.mkv *.avi *.webm);;所有文件 (*)")
        if not paths:
            return
        for p in paths:
            if p not in self.inputs:
                self.inputs.append(p)
                self.queue.addItem(p)
        self.in_label.setText(f"已选 {len(self.inputs)} 个")
        self._log(f"添加 {len(paths)} 个，共 {len(self.inputs)} 个")
        if self.duration <= 0 and self.inputs:
            try:
                self.duration = probe(self.inputs[0]).duration
            except Exception:
                self.duration = 0.0

    def clear_inputs(self):
        self.inputs.clear(); self.queue.clear(); self.in_label.setText("(未选择，可多选)")
        self.duration = 0.0

    # ---------- drop-in model ----------
    def on_backend_changed(self, idx):
        """Show the ONNX model picker only for the drop-in backend."""
        self.dropin_row.setVisible(idx == 2)

    def pick_onnx_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 ONNX 超分模型", "", "ONNX 模型 (*.onnx);;所有文件 (*)")
        if not path:
            return
        from pathlib import Path
        from ..core.config import WEIGHTS_DIR
        import shutil
        src = Path(path)
        # Stage into weights dir so the plugin resolves it by plain filename
        # (avoids absolute-path / mirror ambiguity in plugin setup()).
        try:
            WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
            dst = WEIGHTS_DIR / src.name
            if src.resolve() != dst.resolve():
                shutil.copy2(src, dst)
        except Exception as e:
            self._log(f"[警告] 模型已选但未复制: {e}")
            dst = src
        self.onnx_model.setText(dst.name)
        self._log(f"[模型] drop-in: {dst.name}  (把原生倍数/推理 tile 按模型调整)")

    def _log(self, s): self.log.append(s)

    # ---------- timeline ----------
    def on_scrub(self, val):
        if self.duration > 0:
            self.time_label.setText(f"{self.duration * val / 1000:.1f}s / {self.duration:.1f}s")

    def do_preview(self):
        if not self.inputs:
            self._log("[错误] 先选择视频"); return
        if self.preview_worker and self.preview_worker.isRunning():
            self._log("[提示] 预览渲染中…"); return
        t = self.duration * self.slider.value() / 1000 if self.duration > 0 else 0.0
        cfg = self._make_cfg(self.inputs[0], "")  # out path unused in preview
        self._log(f"预览 @ {t:.1f}s …")
        self.preview_worker = PreviewWorker(self.inputs[0], t, cfg)
        self.preview_worker.done.connect(self.on_preview_done)
        self.preview_worker.failed.connect(self.on_preview_fail)
        self.preview_worker.start()

    def on_preview_done(self, orig, enh):
        self.orig_frame, self.enh_frame = orig, enh
        import cv2
        oh, ow = orig.shape[:2]; eh, ew = enh.shape[:2]
        # fit both to same display height
        disp_h = 300
        def fit(im):
            h, w = im.shape[:2]
            return cv2.resize(im, (int(w * disp_h / h), disp_h))
        a, b = fit(orig), fit(enh)
        cmp = __import__("numpy").hstack([a, b])
        self.preview_lbl.setPixmap(_bgr_to_pixmap(cmp).scaled(
            self.preview_lbl.width() - 8, self.preview_lbl.height() - 8,
            Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self._log(f"[预览] 原 {ow}x{oh} → 增强 {ew}x{eh} (左原/右增强)")

    def on_preview_fail(self, tb):
        self._log("[预览失败]\n" + tb[-800:])

    # ---------- batch run ----------
    def _make_cfg(self, inp, out):
        mult = 1
        if self.interp.currentText() == "x2": mult = 2
        elif self.interp.currentText() == "x4": mult = 4
        bi = self.backend.currentIndex()
        backend = "cugan" if bi == 1 else ("model" if bi == 2 else "esrgan")
        model_backend = self.model_backend.currentText()
        # collect enabled plugin ids: HDR/stabilize toggles + list selections
        enabled = []
        plugin_params = {}
        if backend == "model":
            if not any(p.startswith("builtin.onnx") or "onnx" in p.lower() for p in enabled):
                enabled.append("builtin.onnx_superres")
            # GUI drop-in controls -> onnx_superres plugin params (explicit wins)
            pp = {"native_scale": self.onnx_native.value(), "tile": self.onnx_tile.value()}
            model_file = self.onnx_model.text().strip()
            if model_file:
                pp["model_file"] = model_file
            plugin_params["builtin.onnx_superres"] = pp
        if self.hdr.isChecked():
            enabled.append("builtin.hdr")
        if self.stabilize.isChecked():
            enabled.append("builtin.stabilize")
        for it in self.plugin_list.selectedItems():
            pid = it.data(0x100)  # Qt.UserRole
            if pid and pid not in enabled:
                enabled.append(pid)
        return EnhanceConfig(
            input_path=inp, output_path=out or str(OUTPUT_DIR / "preview.mp4"),
            do_upscale=True, upscale_backend=backend,
            upscale_model=self.model.currentText(),
            outscale=self.scale.value(),
            do_face=self.face.isChecked(),
            do_interp=mult > 1, interp_mult=mult,
            codec=self.codec.currentText(),
            model_backend=model_backend,
            enabled_plugins=list(dict.fromkeys(enabled)),
            plugin_params=plugin_params,
        )

    def start_run(self):
        if not self.inputs:
            self._log("[错误] 先选择输入视频"); return
        if self.worker and self.worker.isRunning():
            self._log("[提示] 正在运行中…"); return
        cfgs = []
        for inp in self.inputs:
            base = os.path.splitext(os.path.basename(inp))[0]
            out = str(OUTPUT_DIR / f"{base}_enhanced.mp4")
            cfgs.append(self._make_cfg(inp, out))
        self.run_btn.setEnabled(False); self.abort_btn.setEnabled(True)
        self._log(f"开始批量: {len(cfgs)} 个, 后端={cfgs[0].upscale_backend}, "
                  f"人脸={cfgs[0].do_face}, 倍数={cfgs[0].outscale}")
        self.worker = Worker(cfgs)
        self.worker.job_started.connect(self.on_job)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished_ok.connect(self.on_done)
        self.worker.failed.connect(self.on_fail)
        self.worker.start()

    def abort_run(self):
        if self.worker:
            self.worker.abort(); self._log("[中止] 将在当前帧后停止")

    def on_job(self, i, total, name):
        self._log(f"[{i}/{total}] {name}")

    def on_progress(self, done, total, extra):
        if total:
            self.bar.setMaximum(total); self.bar.setValue(done)
        self.bar.setFormat(f"%p% {extra}")

    def on_done(self, out):
        self._log(f"[完成] 最后输出: {out}")
        self.bar.setValue(self.bar.maximum())
        self.run_btn.setEnabled(True); self.abort_btn.setEnabled(False)

    def on_fail(self, tb):
        self._log("[失败]\n" + tb)
        self.run_btn.setEnabled(True); self.abort_btn.setEnabled(False)


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
