# SPDX-License-Identifier: Apache-2.0
"""Single source of truth for "which accelerator backend is this process on".

Three states:
  * "cuda" — an NVIDIA GPU via CUDA (torch's normal case).
  * "rocm" — an AMD GPU via ROCm. AMD's ROCm torch build aliases torch.cuda.*
    to HIP, so torch.cuda.is_available()/.device("cuda")/.synchronize() all
    keep working unmodified; torch.version.hip (vs .cuda) is what tells the
    two apart.
  * "cpu"  — no GPU (or a GPU whose backend we don't recognise). DEMO-FALLBACK
    mode kicks in above this, same as always.

RAPIDS (cudf/cuml/rmm) has no ROCm port and XGBoost has no official ROCm
wheel, so on "rocm" the tokenizer/preprocessing/UMAP stage runs on a CPU
shim (see rapids_shim.py) and XGBoost heads train with device="cpu"; only the
torch decoder embedding step actually uses the AMD GPU. xgb_device() below is
the one place that decision is made.
"""

from __future__ import annotations

from typing import Literal, Optional

Backend = Literal["cuda", "rocm", "cpu"]

_backend: Optional[Backend] = None
_rapids_available: Optional[bool] = None


def backend() -> Backend:
    """Detect once, cache for the process lifetime (mirrors gpu.py's
    _configured latch — the underlying hardware doesn't change mid-process)."""
    global _backend
    if _backend is not None:
        return _backend
    try:
        import torch
        if torch.cuda.is_available():
            if getattr(torch.version, "hip", None):
                _backend = "rocm"
            else:
                _backend = "cuda"
        else:
            _backend = "cpu"
    except Exception:                                              # noqa: BLE001
        _backend = "cpu"
    return _backend


def device_name() -> Optional[str]:
    """torch's name for device 0, or None off-GPU. Best-effort — never raises."""
    if backend() == "cpu":
        return None
    try:
        import torch
        return torch.cuda.get_device_name(0)
    except Exception:                                              # noqa: BLE001
        return None


def xgb_device() -> str:
    """"cuda" only on real NVIDIA hardware; "cpu" everywhere else (ROCm has no
    official XGBoost wheel, so this is the CPU-fallback decision for the
    fraud-head training step)."""
    return "cuda" if backend() == "cuda" else "cpu"


def rapids_available() -> bool:
    """True only when the real cudf can be imported (never the shim)."""
    global _rapids_available
    if _rapids_available is not None:
        return _rapids_available
    try:
        import cudf                                                # noqa: F401
        _rapids_available = not getattr(cudf, "__tfm_demo_shim__", False)
    except Exception:                                              # noqa: BLE001
        _rapids_available = False
    return _rapids_available
