"""Backend auto-detection for model plugins.

Three backends, best-first: cuda (torch), directml (onnxruntime-directml),
cpu (always-available fallback). Detection is best-effort: any import failure
degrades gracefully and never raises, so the pipeline runs on any machine.

A backend choice carries two capability hints:
- ``device`` — for torch-based model plugins ('cuda' | 'cpu')
- ``ort_provider`` — for onnxruntime-based plugins (provider name string)
"""
from __future__ import annotations

from dataclasses import dataclass

CUDA = "cuda"
DIRECTML = "directml"
CPU = "cpu"
ALL_BACKENDS = (CUDA, DIRECTML, CPU)


@dataclass(frozen=True)
class BackendChoice:
    name: str             # cuda | directml | cpu
    device: str           # torch device: 'cuda' | 'cpu'
    ort_provider: str     # onnxruntime ExecutionProvider name
    note: str = ""        # human-readable detection source


def _probe_cuda_torch() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _probe_ort_providers() -> list[str]:
    try:
        import onnxruntime as ort
        return [str(p) for p in ort.get_available_providers()]
    except Exception:
        return []


def detect() -> list[BackendChoice]:
    """Return available backends, best quality first. Always includes CPU.

    ``ort_provider`` always names a provider the *installed* onnxruntime build
    actually supports (directml build has no CUDAExecutionProvider), so ONNX
    plugins never hit a "provider not available" warning/fallback.
    """
    out: list[BackendChoice] = []
    provs = _probe_ort_providers()
    cuda_torch = _probe_cuda_torch()
    if cuda_torch:
        # For ONNX models, pick a provider this ort build really has.
        if "CUDAExecutionProvider" in provs:
            ort_p = "CUDAExecutionProvider"
        elif "DmlExecutionProvider" in provs:
            ort_p = "DmlExecutionProvider"
        else:
            ort_p = "CPUExecutionProvider"
        out.append(BackendChoice(CUDA, "cuda", ort_p, "torch.cuda available"))
    if "DmlExecutionProvider" in provs:
        out.append(BackendChoice(DIRECTML, "cpu", "DmlExecutionProvider",
                                 "onnxruntime-directml (cross-GPU)"))
    if "CUDAExecutionProvider" in provs and not any(c.name == CUDA for c in out):
        out.append(BackendChoice(CUDA, "cuda", "CUDAExecutionProvider",
                                 "onnxruntime-gpu"))
    out.append(BackendChoice(CPU, "cpu", "CPUExecutionProvider", "always available"))
    return out


def pick(preferred: str = "auto") -> BackendChoice:
    """Pick a backend. ``auto`` → best available; explicit name → validate or
    gracefully degrade to CPU with a note (never raises on unavailable)."""
    avail = detect()
    if preferred and preferred != "auto":
        for c in avail:
            if c.name == preferred:
                return c
        return BackendChoice(CPU, "cpu", "CPUExecutionProvider",
                             f"requested '{preferred}' unavailable → degraded to cpu")
    return avail[0]


def describe() -> str:
    """One-line summary for logs: ``cuda(torch.cuda available), directml(...), cpu(...)``."""
    return ", ".join(f"{c.name}({c.note})" for c in detect())
