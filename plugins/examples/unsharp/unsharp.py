"""Example third-party FRAME plugin: unsharp masking (OpenCV only).

Copy this folder, edit plugin.json, drop it into plugins/ — that's the whole
onboarding. No registration, no base-class import, no framework dependency.
"""
from __future__ import annotations

import cv2
import numpy as np


class UnsharpPlugin:
    def setup(self, ctx):
        self.amount = float(ctx.params.get("amount", 0.6))
        self.radius = float(ctx.params.get("radius", 1.2))
        ctx.log(f"[unsharp] amount={self.amount} radius={self.radius}")

    def process(self, frame: np.ndarray) -> np.ndarray:
        blur = cv2.GaussianBlur(frame, (0, 0), self.radius)
        out = cv2.addWeighted(frame, 1.0 + self.amount, blur, -self.amount, 0)
        return out

    def teardown(self):
        pass
