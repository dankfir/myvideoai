"""GFPGAN face restoration wrapper (TencentARC, Apache-2.0).

Restores faces degraded by compression / low resolution. Placed AFTER upscaling
in the pipeline so it works at full output resolution. Uses local weights only
(no network) by pointing GFPGANer / facexlib at the shared weights directory
through an absolute path.
"""
from __future__ import annotations

import numpy as np

from .config import GFPGAN_WEIGHTS, WEIGHTS_DIR
from .weights import ensure_weights

# GFPGANer hard-codes model_rootpath='gfpgan/weights' (relative to CWD) and
# facexlib resolves detection/parsing models relative to the same CWD. Instead
# of mutating the process CWD (which breaks concurrent use), we temporarily
# swap gfpgan.utils.FaceRestoreHelper for a subclass that forces an absolute
# model_rootpath, then restore it.
GFPGAN_W = "GFPGANv1.4.pth"
DET_W = "detection_Resnet50_Final.pth"
PARSE_W = "parsing_parsenet.pth"


class FaceEnhancer:
    """Lazy-loaded GFPGANer. Enhances faces in a BGR frame (same resolution)."""

    def __init__(self, device: str = "cuda", weight: float = 0.5):
        self.device = device
        self.weight = weight
        self._restorer = None

    def _build(self):
        from gfpgan import GFPGANer
        import gfpgan.utils as gfpgan_utils

        root = WEIGHTS_DIR.resolve()
        for name in (GFPGAN_W, DET_W, PARSE_W):
            ensure_weights(GFPGAN_WEIGHTS[name], name, dest_dir=root, log=print)

        original_helper = gfpgan_utils.FaceRestoreHelper

        class _AbsoluteRootHelper(original_helper):
            """FaceRestoreHelper pinned to an absolute weights directory."""

            def __init__(self, *args, **kwargs):
                kwargs["model_rootpath"] = str(root)
                super().__init__(*args, **kwargs)

        gfpgan_utils.FaceRestoreHelper = _AbsoluteRootHelper
        try:
            self._restorer = GFPGANer(
                model_path=str(root / GFPGAN_W),
                upscale=1, arch="clean", channel_multiplier=2,
                bg_upsampler=None, device=self.device)
        finally:
            gfpgan_utils.FaceRestoreHelper = original_helper

    def enhance(self, frame_bgr: np.ndarray) -> np.ndarray:
        if self._restorer is None:
            self._build()
        # GFPGANer paths resolved at init; enhance itself does no file IO.
        _, _, out = self._restorer.enhance(
            frame_bgr, has_aligned=False, only_center_face=False, paste_back=True,
            weight=self.weight)
        return out if out is not None else frame_bgr
