"""Device helpers shared by the pipeline and model plugins.

Kept in its own module so plugins (loaded standalone via spec_from_file_location)
can import it without pulling in the pipeline and creating a cycle.
"""
from __future__ import annotations


def auto_tile(device: str = "cuda") -> int:
    """Pick a super-resolution tile size that fits available VRAM.

    256 / 384 / 512 for <=6GB / <=10GB / more. Non-CUDA devices use the
    conservative 256 because shared system memory is not measured here.
    """
    try:
        if device == "cuda":
            import torch
            _, total = torch.cuda.mem_get_info()
            gb = total / (1024 ** 3)
        else:
            gb = 0
    except Exception:
        gb = 0
    return 256 if gb <= 6 else (384 if gb <= 10 else 512)
