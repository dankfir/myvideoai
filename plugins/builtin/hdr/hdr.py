"""SDR -> HDR10 plugin. Post-encode whole-file pass using libplacebo.

Verified command shape (RTX 4060, FFmpeg 9.0.1):
  -vf format=p010le,libplacebo=color_primaries=bt2020:color_trc=smpte2084:\
colorspace=bt2020nc:inverse_tonemapping=1:gamut_mode=0
  output: yuv420p10le / bt2020nc / smpte2084 / bt2020  (true HDR10 tagging)
"""
from __future__ import annotations

import subprocess
from pathlib import Path


class HDRPlugin:
    def setup(self, ctx):
        self.ctx = ctx
        p = ctx.params
        self.param = float(p.get("tonemapping_param", 0.0))
        self.gamut = int(p.get("gamut_mode", 0))
        self.crf = int(p.get("crf", 18))

    def process(self, in_path: str, out_path: str, ctx):
        vf = (
            "format=p010le,"
            f"libplacebo=color_primaries=bt2020:color_trc=smpte2084:"
            f"colorspace=bt2020nc:inverse_tonemapping=1:"
            f"tonemapping_param={self.param}:gamut_mode={self.gamut}"
        )
        cmd = [
            ctx.ffmpeg, "-y", "-v", "error", "-i", in_path,
            "-vf", vf,
            "-c:v", "libx264", "-crf", str(self.crf), "-preset", "medium",
            "-pix_fmt", "yuv420p10le",
            "-color_primaries", "bt2020", "-color_trc", "smpte2084",
            "-colorspace", "bt2020nc",
            "-map", "0:v:0", "-map", "0:a?", "-c:a", "copy",
            "-movflags", "+faststart", out_path,
        ]
        ctx.log(f"[hdr] inverse tone map -> HDR10 (param={self.param})")
        subprocess.run(cmd, check=True, capture_output=True)
        return out_path

    def teardown(self):
        pass
