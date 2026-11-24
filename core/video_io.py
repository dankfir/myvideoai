"""FFmpeg-based video probing and frame pipe helpers.

We avoid loading whole videos into RAM: frames stream through ffmpeg pipes.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterator, Optional

import numpy as np

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    n_frames: int
    duration: float
    codec: str
    has_audio: bool


def probe(path: str) -> VideoInfo:
    cmd = [
        FFPROBE, "-v", "error", "-print_format", "json",
        "-show_streams", "-show_format", str(path),
    ]
    out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
    data = json.loads(out)

    v = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
    if v is None:
        raise ValueError(f"No video stream in {path}")
    a = next((s for s in data["streams"] if s["codec_type"] == "audio"), None)

    fps_str = v.get("avg_frame_rate") or v.get("r_frame_rate") or "0/1"
    try:
        fps = float(Fraction(fps_str))
    except Exception:
        fps = 0.0
    if fps <= 0:
        fps = 30.0

    dur = float(data.get("format", {}).get("duration", 0.0) or 0.0)
    nb = int(v.get("nb_frames", 0) or 0)
    if nb <= 0 and dur > 0:
        nb = int(round(dur * fps))

    return VideoInfo(
        width=int(v["width"]),
        height=int(v["height"]),
        fps=fps,
        n_frames=nb,
        duration=dur,
        codec=v.get("codec_name", ""),
        has_audio=a is not None,
    )


def iter_frames(path: str, target_fps: Optional[float] = None,
                size: Optional[tuple[int, int]] = None) -> Iterator[np.ndarray]:
    """Yield BGR frames (H,W,3 uint8) streamed from ffmpeg.

    target_fps: resample fps (e.g. keep source fps). size: (w,h) resize.
    """
    vf = []
    if target_fps:
        vf.append(f"fps={target_fps}")
    if size:
        vf.append(f"scale={size[0]}:{size[1]}")
    cmd = [FFMPEG, "-v", "error", "-i", str(path)]
    if vf:
        cmd += ["-vf", ",".join(vf)]
    cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24", "-"]

    w, h = size if size else (None, None)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            bufsize=10**8)
    try:
        if w and h:
            frame_bytes = w * h * 3
            while True:
                raw = proc.stdout.read(frame_bytes)
                if len(raw) < frame_bytes:
                    break
                yield np.frombuffer(raw, np.uint8).reshape(h, w, 3)
        else:
            # need width/height from probe if not given -> caller should pass size
            raise ValueError("iter_frames requires size when piping raw frames")
    finally:
        proc.stdout.close()
        proc.wait()


class FrameWriter:
    """Encode an open-ended sequence of BGR frames with ffmpeg."""

    def __init__(self, path: str, width: int, height: int, fps: float,
                 codec: str = "libx264", crf: int = 18, preset: str = "medium",
                 pix_fmt: str = "yuv420p", audio_from: Optional[str] = None):
        self.width, self.height, self.fps = width, height, fps
        cmd = [
            FFMPEG, "-y", "-v", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", f"{fps}",
            "-i", "-",
        ]
        if audio_from:
            cmd += ["-i", str(audio_from), "-map", "0:v:0", "-map", "1:a:0", "-shortest"]
        cmd += [
            "-c:v", codec, "-crf", str(crf), "-preset", preset,
            "-pix_fmt", pix_fmt, "-movflags", "+faststart", str(path),
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                                     bufsize=10**8)

    def write(self, frame: np.ndarray):
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            import cv2  # lazy
            frame = cv2.resize(frame, (self.width, self.height))
        self.proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self):
        if self.proc.stdin:
            self.proc.stdin.close()
        err = self.proc.stderr.read().decode("utf-8", "ignore")
        self.proc.wait()
        if self.proc.returncode != 0:
            raise RuntimeError(f"ffmpeg encode failed:\n{err[-2000:]}")
