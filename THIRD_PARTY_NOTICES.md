# Third-Party Notices

This file records the licenses and attribution for third-party code, models and
runtime dependencies used by **myvideoai**.

`myvideoai` itself is licensed under the Apache License 2.0 — see
[`LICENSE`](LICENSE). Copyright 2026 the myvideoai authors.

The license texts reproduced below are verbatim copies of the license files
shipped inside the corresponding distributions. Where an upstream project ships
its license text inside its own distribution, that copy is the authoritative
artifact for this notice.

---

## 1. Source code incorporated into this repository

### 1.1 Real-CUGAN (UpCunet architecture)

* Upstream: <https://github.com/bilibili/ailab/tree/main/Real-CUGAN>
* License: MIT License — Copyright (c) 2022 bilibili
* License text: <https://github.com/bilibili/ailab/blob/main/Real-CUGAN/LICENSE>,
  reproduced in section 6 of this file. Real-CUGAN ships no NOTICE file.
* Provenance: `myvideoai/core/cugan_arch.py` is a readable re-formatting of the
  UpCunet network used by Real-CUGAN (upstream module `upcunet_v3.py` in the
  same repository). Layer topology and parameter names are kept identical so
  that the officially released `RealCUGAN_4x.pth` weight loads directly without
  key remapping.
* Modifications relative to the upstream `upcunet_v3.py`: the Pro / cache
  variants and their code paths were dropped (upstream `cache_mode` /
  `self.pro` / `forward_fast_rough` / `forward_gap_sync`); upstream's
  `os._exit(233)` guards on bad tile config were removed; the tile-mode dispatch
  was reduced to the subset used here; compact statements were reformatted and
  NumPy conversion made explicit (`np.ascontiguousarray` in `np2tensor`).
* Distributed under the MIT License, preserving the upstream copyright notice.

### 1.2 RIFE / IFNet_HDv3 (frame interpolation)

* Upstream: <https://github.com/hzwer/Practical-RIFE>
* License: MIT License — Copyright (c) 2021 hzwer
* License text: reproduced in section 3.1 of this file, and available upstream
  at <https://github.com/hzwer/Practical-RIFE/blob/master/LICENSE>
* `myvideoai/core/interpolator.py` is a PyTorch re-implementation of the
  IFNet_HDv3 interpolation network. The warp operator is inlined via
  `grid_sample`. The released `RIFE_HDv3.pth` weight (RIFE v4.6, ~21 MB) is
  downloaded on first use; the upstream `module.` DataParallel prefix is
  stripped when the state dict is loaded.
* The MIT license text is reproduced in `THIRD_PARTY_NOTICES.md`, which is
  included in this repository.

### 1.3 Packages used through their public APIs

The following packages are **not vendored into this repository**. They are
installed as regular dependencies (see `requirements.txt`) and are used through
their public APIs only.

| Package | Upstream | License | Notes |
| --- | --- | --- | --- |
| Real-ESRGAN | <https://github.com/xinntao/Real-ESRGAN> | BSD 3-Clause — Copyright (c) 2021, Xintao Wang | `myvideoai/core/upscaler.py` uses the official `realesrgan` package and the officially released weights. |
| BasicSR | <https://github.com/XPixelGroup/BasicSR> | Apache License 2.0 | `myvideoai/core/upscaler.py` uses the official `basicsr` package. |
| GFPGAN | <https://github.com/TencentARC/GFPGAN> | Apache License 2.0 — Copyright (C) 2021 THL A29 Limited, a Tencent company | `myvideoai/core/face_enhancer.py` is a wrapper around `GFPGANer` / `facexlib`; only local weights are used (no network access at runtime). |
| facexlib | <https://github.com/xinntao/facexlib> | MIT License — Copyright (c) 2020 Xintao Wang | Face detection / parsing models used by GFPGAN. See the license inconsistency note in section 4. |
| ONNX Runtime | <https://github.com/microsoft/onnxruntime> | MIT License — Copyright (c) Microsoft Corporation | Runs the ONNX Super-Resolution model plugin. |
| onnxruntime-directml | <https://github.com/microsoft/onnxruntime> | MIT License | DirectML execution provider for Windows GPUs without CUDA. |
| OpenCV (opencv-python / opencv-python-headless) | <https://github.com/opencv/opencv> | Apache 2.0 per wheel metadata; the bundled `cv2/LICENSE.txt` is MIT — see section 4 | Frame and image IO. |
| PySide6 / Qt 6 | <https://doc.qt.io/qtforpython-6/> | LGPL-3.0 | GUI front-end. Not vendored; installed as a normal dependency. |
| ONNX | <https://github.com/onnx/onnx> | Apache License 2.0 | Model format tooling. |
| NumPy | <https://github.com/numpy/numpy> | BSD 3-Clause — Copyright (c) 2005-2024, NumPy Developers | Array math. |
| SciPy | <https://github.com/scipy/scipy> | BSD 3-Clause | Scientific computing helpers. |
| Pillow | <https://github.com/python-pillow/Pillow> | HPND | Image IO helpers. |
| tqdm | <https://github.com/tqdm/tqdm> | MIT License — Copyright (c) 2013 Noah Y and contributors | Progress bars. |
| torch / torchvision | <https://github.com/pytorch/pytorch> | BSD 3-Clause | Deep learning runtime; installed separately (see `requirements.txt`). |
| ffmpeg / ffprobe | <https://ffmpeg.org/> | LGPL-2.1+ (or GPL, depending on the build) | External binaries required for decoding, interpolation-free muxing and encoding. Invoked as a subprocess. |

Transitive dependencies pulled in by the packages above keep their own
licenses. The most common ones observed in a typical installation are
`future` (MIT), `lmdb` (OLDAP-2.8), `yapf` (Apache-2.0), `addict` (unknown /
unrestricted per its metadata), `scikit-image` (per-file licenses) and
`packaging` / `typing-extensions` (BSD-2-Clause / Apache-2.0). Consult the
metadata of each installed distribution for the authoritative text.

---

## 2. Model weights and assets

**No model weights are redistributed in this repository.** All weights are
downloaded on first use into the shared weights directory
(`myvideoai/models/weights/`, excluded from version control) by
`myvideoai/core/weights.py`, which fetches from a mirror first and falls back to
the original upstream host.

| Model | Upstream | License | Files |
| --- | --- | --- | --- |
| Real-ESRGAN | <https://github.com/xinntao/Real-ESRGAN> | BSD 3-Clause | `RealESRGAN_x4plus.pth` (release `v0.2.5.0`), `RealESRGAN_x4plus_anime_6B.pth` (release `v0.2.2.4`), `RealESRGAN_x2plus.pth` (release `v0.2.1`) |
| Real-CUGAN | <https://github.com/bilibili/ailab/tree/main/Real-CUGAN> | MIT | `RealCUGAN_4x.pth` |
| GFPGAN | <https://github.com/TencentARC/GFPGAN> | Apache 2.0 | `GFPGANv1.4.pth`, `detection_Resnet50_Final.pth`, `parsing_parsenet.pth` |
| RIFE | <https://github.com/hzwer/Practical-RIFE> | MIT | `RIFE_HDv3.pth` |

If you redistribute the source together with any of these weight files, you
must comply with the corresponding license (MIT notice for Real-CUGAN,
Apache 2.0 notice for GFPGAN, BSD 3-Clause notice for Real-ESRGAN, MIT notice
for RIFE) and with the terms of the model licenses themselves.

Sample media shipped in `myvideoai/samples/` are generated locally by
`myvideoai/tests/make_sample.py` (FFmpeg `lavfi` synthetic sources) and contain
no third-party content.

---

## 3. Full license texts

### 3.1 MIT License — RIFE / Practical-RIFE (hzwer)

```
MIT License

Copyright (c) 2021 hzwer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### 3.2 MIT License — facexlib (Xintao Wang)

```
MIT License

Copyright (c) 2020 Xintao Wang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### 3.3 MIT License — ONNX Runtime (Microsoft Corporation)

```
MIT License

Copyright (c) Microsoft Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### 3.4 MIT License — OpenCV wheel (`cv2/LICENSE.txt`)

```
MIT License

Copyright (c) Olli-Pekka Heinisuo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### 3.5 MIT License — embedded component of GFPGAN (Kim Seonghyeon)

GFPGAN ships this MIT-licensed component inside its own `LICENSE` file
(lines 184–204 of `gfpgan-1.3.8.dist-info/LICENSE`).

```
MIT License

Copyright (c) 2019 Kim Seonghyeon

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### 3.6 BSD 3-Clause License — Real-ESRGAN (Xintao Wang)

```
BSD 3-Clause License

Copyright (c) 2021, Xintao Wang
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### 3.7 BSD 3-Clause License — components bundled with GFPGAN

GFPGAN's `LICENSE` file lists the following third-party components under the
BSD 3-Clause license:

1. **torchvision** — Copyright (c) Soumith Chintala 2016, All rights reserved.
2. **torch** — Copyright (c) 2016- Facebook, Inc (Adam Paszke); Copyright (c)
   2014- Facebook, Inc (Soumith Chintala); Copyright (c) 2011-2014 Idiap
   Research Institute (Ronan Collobert); Copyright (c) 2012-2014 DeepMind
   Technologies (Koray Kavukcuoglu); Copyright (c) 2011-2012 NEC Laboratories
   America (Koray Kavukcuoglu); Copyright (c) 2011-2013 NYU (Clement
   Farabet); Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert,
   Leon Bottou, Iain Melvin, Jason Weston); Copyright (c) 2006 Idiap Research
   Institute (Samy Bengio); Copyright (c) 2001-2004 Idiap Research Institute
   (Ronan Collobert, Johnny Mariethoz).
3. **numpy** — Copyright (c) 2005-2020, NumPy Developers, All rights reserved.

The BSD 3-Clause redistribution conditions, documentation requirement,
no-endorsement clause and disclaimer apply to those components, with the text
reproduced in section 3.6 above. GFPGAN additionally lists further third-party
components under "BSD 3-Clause License and Other Licenses of the Third-Party
Components therein"; consult the upstream `LICENSE` file for that complete
list.

---

## 4. Known license inconsistencies in upstream distributions

These are recorded as-is so that the attribution in this file stays faithful to
the artifacts that were actually installed and inspected:

* **facexlib** — the bundled `facexlib-0.2.5.dist-info/LICENSE` file is the MIT
  License (Copyright (c) 2020 Xintao Wang), while the wheel `METADATA` states
  `License: Apache License 2.0`. This notice follows the license file, which is
  the authoritative artifact shipped with the distribution.
* **OpenCV** — the `opencv-python-headless` wheel metadata states
  `License: Apache 2.0`, while the bundled `cv2/LICENSE.txt` contains the MIT
  License (Copyright (c) Olli-Pekka Heinisuo) and `cv2/LICENSE-3RD-PARTY.txt`
  lists additional third-party terms. Both are reproduced here rather than
  picking one and hiding the other.
* **GFPGAN** — the project is Apache 2.0 "except for the third-party
  components listed below" (MIT and BSD-3-Clause). Both variants are
  reproduced in sections 3.5 and 3.7.
* **PySide6 / Qt 6** — the PySide6 6.12.0 wheel metadata declares the SPDX
  expression `LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only`. This project uses
  PySide6 under the LGPL-3.0 option. PySide6 is an unmodified external
  dependency; it is neither vendored nor statically linked into this
  repository. The full LGPL-3.0 text is available at
  <https://www.gnu.org/licenses/lgpl-3.0.html>.

---

## 5. Packaging versions

`requirements.txt` requests `basicsr==1.4.2` and `opencv-python>=4.8`. The
environment used for development and regression testing carried BasicSR
1.3.1 and `opencv-python-headless` 4.14.0.94. The licenses are identical
(Apache 2.0 and MIT respectively); only the packaging differs. Both variants
are usable — the code depends on the public APIs of these packages, not on
distribution-specific packaging.

---

## 6. Vendored architecture source

`myvideoai/core/cugan_arch.py` reproduces the Real-CUGAN UpCunet network
(upstream `upcunet_v3.py`). Its full MIT notice, `Copyright (c) 2022 bilibili`,
is reproduced below.

```
MIT License

Copyright (c) 2022 bilibili

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
