"""RIFE v4.6 (Real-time Intermediate Flow Estimation) frame interpolation.

Architecture implemented after hzwer/Practical-RIFE's
IFNet_HDv3 (MIT License): block0..block3 of IFBlocks with ResConv(+beta)
residual blocks, iterative flow refinement, and a learned fusion mask.
The ``warp`` operator from model/warplayer is inlined via grid_sample so the
package stays self-contained.

Upstream: https://github.com/hzwer/Practical-RIFE
Upstream license: MIT License
  Copyright (c) 2021 hzwer
  Full text: https://github.com/hzwer/Practical-RIFE/blob/master/LICENSE
  (The MIT license text is reproduced in THIRD_PARTY_NOTICES.md, which is
  included in this repository.)

Weights: flownet.pkl (RIFE v4.6, ~21 MB) auto-downloaded, loaded with the
"module." DataParallel prefix stripped.
"""
from __future__ import annotations

import numpy as np

from .config import RIFE_WEIGHTS, WEIGHTS_DIR
from .weights import ensure_weights


# --------------------------------------------------------------------------
# RIFE v4.6 IFNet (self-contained, matches flownet.pkl state_dict)
# --------------------------------------------------------------------------
def _build_rife():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def warp(img, flow):
        """Inlined model/warplayer.warp using differentiable grid_sample."""
        b, _, h, w = img.shape
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, device=img.device, dtype=flow.dtype),
            torch.arange(w, device=img.device, dtype=flow.dtype),
            indexing="ij",
        )
        gx = grid_x.unsqueeze(0) + flow[:, 0]
        gy = grid_y.unsqueeze(0) + flow[:, 1]
        gx = 2.0 * gx / max(w - 1, 1) - 1.0
        gy = 2.0 * gy / max(h - 1, 1) - 1.0
        grid = torch.stack((gx, gy), dim=-1)
        return F.grid_sample(img, grid, mode="bilinear", padding_mode="border",
                             align_corners=True)

    def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.LeakyReLU(0.2, True),
        )

    class ResConv(nn.Module):
        def __init__(self, c, dilation=1):
            super().__init__()
            self.conv = nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation, groups=1)
            self.beta = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
            self.relu = nn.LeakyReLU(0.2, True)

        def forward(self, x):
            return self.relu(self.conv(x) * self.beta + x)

    class IFBlock(nn.Module):
        def __init__(self, in_planes, c=64):
            super().__init__()
            self.conv0 = nn.Sequential(
                conv(in_planes, c // 2, 3, 2, 1),
                conv(c // 2, c, 3, 2, 1),
            )
            self.convblock = nn.Sequential(
                *[ResConv(c) for _ in range(8)]
            )
            self.lastconv = nn.Sequential(
                nn.ConvTranspose2d(c, 4 * 6, 4, 2, 1),
                nn.PixelShuffle(2),
            )

        def forward(self, x, flow=None, scale=1):
            x = F.interpolate(x, scale_factor=1. / scale, mode="bilinear",
                              align_corners=False)
            if flow is not None:
                flow = F.interpolate(flow, scale_factor=1. / scale, mode="bilinear",
                                     align_corners=False) * 1. / scale
                x = torch.cat((x, flow), 1)
            feat = self.convblock(self.conv0(x))
            tmp = self.lastconv(feat)
            tmp = F.interpolate(tmp, scale_factor=scale, mode="bilinear",
                                align_corners=False)
            flow = tmp[:, :4] * scale
            mask = tmp[:, 4:5]
            return flow, mask

    class IFNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.block0 = IFBlock(7, c=192)
            self.block1 = IFBlock(8 + 4, c=128)
            self.block2 = IFBlock(8 + 4, c=96)
            self.block3 = IFBlock(8 + 4, c=64)

        def forward(self, img0, img1, timestep=0.5, scale_list=(8, 4, 2, 1),
                    ensemble=False):
            if not torch.is_tensor(timestep):
                timestep = (img0[:, :1].clone() * 0 + 1) * timestep
            else:
                timestep = timestep.repeat(1, 1, img0.shape[2], img0.shape[3])
            warped_img0 = img0
            warped_img1 = img1
            flow = None
            mask = None
            block = [self.block0, self.block1, self.block2, self.block3]
            merged = None
            for i in range(4):
                if flow is None:
                    flow, mask = block[i](
                        torch.cat((img0[:, :3], img1[:, :3], timestep), 1),
                        None, scale=scale_list[i])
                    if ensemble:
                        f1, m1 = block[i](
                            torch.cat((img1[:, :3], img0[:, :3], 1 - timestep), 1),
                            None, scale=scale_list[i])
                        flow = (flow + torch.cat((f1[:, 2:4], f1[:, :2]), 1)) / 2
                        mask = (mask + (-m1)) / 2
                else:
                    f0, m0 = block[i](
                        torch.cat((warped_img0[:, :3], warped_img1[:, :3],
                                   timestep, mask), 1),
                        flow, scale=scale_list[i])
                    if ensemble:
                        f1, m1 = block[i](
                            torch.cat((warped_img1[:, :3], warped_img0[:, :3],
                                       1 - timestep, -mask), 1),
                            torch.cat((flow[:, 2:4], flow[:, :2]), 1),
                            scale=scale_list[i])
                        f0 = (f0 + torch.cat((f1[:, 2:4], f1[:, :2]), 1)) / 2
                        m0 = (m0 + (-m1)) / 2
                    flow = flow + f0
                    mask = mask + m0
                warped_img0 = warp(img0, flow[:, :2])
                warped_img1 = warp(img1, flow[:, 2:4])
                merged = (warped_img0, warped_img1)
            mask = torch.sigmoid(mask)
            return merged[0] * mask + merged[1] * (1 - mask)

    return IFNet


class RIFEInterpolator:
    """Interpolate a frame between img0 and img1 by a timestep in (0,1)."""

    def __init__(self, model_key: str = "rife46", device: str = "cuda", half: bool = True):
        self.model_key = model_key
        self.device = device
        self.half = half and device == "cuda"
        self._net = None

    def _build(self):
        import torch
        IFNet = _build_rife()
        net = IFNet()
        wpath = ensure_weights(RIFE_WEIGHTS[self.model_key],
                               f"{self.model_key}.pth",
                               dest_dir=WEIGHTS_DIR, log=print)
        state = torch.load(str(wpath), map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        clean = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
        missing, unexpected = net.load_state_dict(clean, strict=False)
        net.eval().to(self.device)
        if self.half:
            net.half()
        self._net = net
        print(f"[rife] loaded {self.model_key} (params {len(clean)}, "
              f"missing {len(missing)}, unexpected {len(unexpected)})")

    def interpolate(self, img0_bgr: np.ndarray, img1_bgr: np.ndarray,
                    timestep: float) -> np.ndarray:
        import torch
        if self._net is None:
            self._build()
        def to_tensor(bgr):
            rgb = bgr[:, :, ::-1].astype(np.float32) / 255.0
            x = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).unsqueeze(0)
            return x.to(self.device, dtype=torch.float16 if self.half else torch.float32)
        # pad to multiple of 32 (RIFE downsamples 8x with stride-2 convs)
        with torch.no_grad():
            x0 = to_tensor(img0_bgr)
            x1 = to_tensor(img1_bgr)
            h, w = x0.shape[-2:]
            ph, pw = (-h) % 32, (-w) % 32
            if ph or pw:
                x0 = torch.nn.functional.pad(x0, (0, pw, 0, ph), mode="replicate")
                x1 = torch.nn.functional.pad(x1, (0, pw, 0, ph), mode="replicate")
            out = self._net(x0, x1, float(timestep))
            if ph or pw:
                out = out[..., :h, :w]
        out = out.clamp(0, 1)[0].float().cpu().numpy().transpose(1, 2, 0)
        return (out[:, :, ::-1] * 255.0).round().astype(np.uint8)
