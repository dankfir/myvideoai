"""Central configuration for myvideoai.

Holds model paths, default parameters, and runtime toggles.
Model weights are downloaded on first use into models/weights.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = ROOT / "models" / "weights"
OUTPUT_DIR = ROOT / "output"
SAMPLES_DIR = ROOT / "samples"
PLUGINS_DIR = ROOT / "plugins"
USER_PLUGINS_DIR = ROOT / "models" / "plugins"  # drop-in third-party plugins

WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Real-ESRGAN weight presets (name -> [mirror urls, tried in order]) -----
# Official weights (BSD-3). GitHub release is often unreachable in CN networks,
# so we list hf-mirror first (same files), original GitHub as fallback.
REALESRGAN_WEIGHTS = {
    "x4plus": [
        "https://hf-mirror.com/lllyasviel/Annotators/resolve/main/RealESRGAN_x4plus.pth",
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/RealESRGAN_x4plus.pth",
    ],
    "x4plus_anime": [
        "https://hf-mirror.com/lllyasviel/Annotators/resolve/main/RealESRGAN_x4plus_anime_6B.pth",
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    ],
    "x2plus": [
        "https://hf-mirror.com/lllyasviel/Annotators/resolve/main/RealESRGAN_x2plus.pth",
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    ],
}

# --- Real-CUGAN weight (bilibili/ailab, MIT) ------------------------------
# mirrors: primary hf-mirror (CN-friendly), original huggingface fallback.
CUGAN_WEIGHTS = {
    "cugan_4x": [
        "https://hf-mirror.com/smnorini/Real_CUGAN_4x/resolve/main/Real_CUGAN_4x.pth",
        "https://huggingface.co/smnorini/Real_CUGAN_4x/resolve/main/Real_CUGAN_4x.pth",
    ],
}

# --- GFPGAN face restoration weights (TencentARC, Apache-2.0) --------------
GFPGAN_WEIGHTS = {
    "GFPGANv1.4.pth": "https://hf-mirror.com/nlightcho/gfpgan_v14/resolve/main/GFPGANv1.4.pth",
    "detection_Resnet50_Final.pth": "https://hf-mirror.com/nlightcho/gfpgan_v14/resolve/main/detection_Resnet50_Final.pth",
    "parsing_parsenet.pth": "https://hf-mirror.com/nlightcho/gfpgan_v14/resolve/main/parsing_parsenet.pth",
}

# --- RIFE weight (from hzwer/Practical-RIFE, MIT) --------------------------
RIFE_WEIGHTS = {
    "rife46": [
        "https://hf-mirror.com/hzwer/RIFE/resolve/main/RIFE_HDv3.pth",
        "https://github.com/hzwer/Practical-RIFE/raw/main/train_log/RIFE_HDv3.pth",
    ],
}

# Upscaler backends: esrgan (photoreal) / cugan (anime/line-art)
UPSCALER_BACKENDS = ("esrgan", "cugan")


@dataclass
class EnhanceConfig:
    """Parameters for one enhancement job."""

    input_path: str = ""
    output_path: str = ""

    # upscaling
    do_upscale: bool = True
    upscale_backend: str = "esrgan"  # esrgan (photo) | cugan (anime)
    upscale_model: str = "x4plus"   # key of REALESRGAN_WEIGHTS
    outscale: float = 2.0           # desired output scale relative to source (1,2,4...)
    tile: int = 0                   # 0 = auto by VRAM; >0 fixed tile size for low VRAM
    cugan_tile_mode: int = 0        # RealCUGAN tile mode 0-4 (higher = less VRAM)

    # face restoration
    do_face: bool = False
    face_weight: float = 0.5        # GFPGAN strength

    # interpolation
    do_interp: bool = False
    interp_mult: int = 2            # 2 => double fps, 4 => x4 fps
    interp_model: str = "rife46"

    # encoding
    crf: int = 18
    preset: str = "medium"
    codec: str = "libx264"          # libx264 (cpu) or h264_nvenc (gpu)
    pix_fmt: str = "yuv420p"

    # plugins — features themselves are plugins (dogfooding the contract)
    # Each id maps to plugins/<...>/plugin.json. Empty = pure AI pipeline.
    enabled_plugins: list = field(default_factory=list)
    plugin_params: dict = field(default_factory=dict)  # {plugin_id: {param: value}}

    # runtime
    device: str = "cuda"            # device for BUILT-IN torch models (cuda/cpu)
    model_backend: str = "auto"     # backend for MODEL plugins: auto/cuda/directml/cpu
    half: bool = True               # fp16 inference
    temp_dir: str = ""

    extra: dict = field(default_factory=dict)
