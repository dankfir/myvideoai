"""Two-pass vidstab stabilization plugin.

Pass 1 writes a transforms file; pass 2 applies it. Uses ctx.work_dir so two
concurrent jobs never collide. Verified on FFmpeg 9.0.1 with vidstabdetect and
vidstabtransform (60-frame 640x360 sample).
"""
from __future__ import annotations

import subprocess
from pathlib import Path


class StabilizePlugin:
    def setup(self, ctx):
        self.ctx = ctx
        p = ctx.params
        self.shakiness = int(p.get("shakiness", 5))
        self.accuracy = int(p.get("accuracy", 15))
        self.smoothing = int(p.get("smoothing", 10))
        self.optzoom = int(p.get("optzoom", 1))
        self.crf = int(p.get("crf", 18))

    def process(self, in_path: str, out_path: str, ctx):
        work = Path(ctx.work_dir or Path(in_path).parent).resolve()
        work.mkdir(parents=True, exist_ok=True)
        # Filter values cannot carry a Windows drive letter (':' = option separator).
        # Use a bare filename and run ffmpeg with cwd=work — eliminates the whole class.
        trf_name = "vidstab_transforms.trf"

        in_abs = str(Path(in_path).resolve()).replace("\\", "/")
        out_abs = str(Path(out_path).resolve()).replace("\\", "/")

        detect = [
            ctx.ffmpeg, "-y", "-v", "error", "-i", in_abs,
            "-vf", f"vidstabdetect=shakiness={self.shakiness}:"
                   f"accuracy={self.accuracy}:result={trf_name}",
            "-map", "0:v:0", "-an", "-f", "null", "-",
        ]
        ctx.log(f"[stabilize] pass1 detect (shakiness={self.shakiness})")
        r1 = subprocess.run(detect, capture_output=True, cwd=str(work))
        if r1.returncode != 0 or not (work / trf_name).exists():
            raise RuntimeError(f"vidstabdetect failed: {r1.stderr.decode('utf-8','ignore')[-800:]}")

        transform = [
            ctx.ffmpeg, "-y", "-v", "error", "-i", in_abs,
            "-vf", f"vidstabtransform=input={trf_name}:smoothing={self.smoothing}:"
                   f"optzoom={self.optzoom}:interpol=bilinear",
            "-c:v", "libx264", "-crf", str(self.crf), "-preset", "medium",
            "-map", "0:v:0", "-map", "0:a?", "-c:a", "copy",
            "-movflags", "+faststart", out_abs,
        ]
        ctx.log(f"[stabilize] pass2 transform (smoothing={self.smoothing})")
        subprocess.run(transform, check=True, capture_output=True, cwd=str(work))
        return out_path

    def teardown(self):
        pass
