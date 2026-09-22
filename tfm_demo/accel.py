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

RAPIDS (cudf/cuml/rmm) has no ROCm port, so on "rocm" the tokenizer /
preprocessing / UMAP stage runs on a CPU shim (see rapids_shim.py).

XGBoost is a different story: AMD maintains a HIP port of it
(github.com/AMD-Ecosystem/xgboost, built with -DUSE_HIP=ON — see
deploy/build_xgboost_rocm.sh), whose device string is still "cuda" because HIP
is a source-level translation of the CUDA API. Mainline XGBoost's PyPI wheel is
CPU-only on AMD hardware, so which one is installed decides whether the fraud
heads train on the GPU. That is *installed-build* state, not hardware state, so
xgb_device()/xgb_status() below probe the build rather than assume: they read
xgboost.build_info()["USE_HIP"|"USE_CUDA"] and then (on ROCm) smoke-train two
rounds in a throwaway subprocess, so a HIP build compiled for the wrong gfx
arch degrades to CPU instead of killing the export. The answer is surfaced in
the server log, /api/status and the UI header — see xgb_status().
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Dict, Literal, Optional

Backend = Literal["cuda", "rocm", "cpu"]

_backend: Optional[Backend] = None
_rapids_available: Optional[bool] = None
_xgb_status: Optional[Dict] = None


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


# ---------------------------------------------------------------------------
# XGBoost: which device the three fraud heads actually train on
# ---------------------------------------------------------------------------

# Two rounds on 64 rows: enough to allocate on the device, build a histogram
# and run a prediction, cheap enough to pay for at warmup. Run in a subprocess
# because a HIP build whose compiled gfx targets don't include this card can
# abort the process rather than raise (same reasoning as gpu.py's
# host_copy_canary).
_XGB_GPU_PROBE = (
    "import numpy as np, xgboost as xgb; "
    "X = np.random.rand(64, 4); y = (X[:, 0] > 0.5).astype(int); "
    "m = xgb.XGBClassifier(n_estimators=2, max_depth=2, tree_method='hist', "
    "device='cuda'); m.fit(X, y); m.predict_proba(X); print('xgb-gpu ok')"
)


def xgb_build_info() -> Dict:
    """xgboost.build_info() (its compile-time flags: USE_CUDA, USE_HIP,
    USE_OPENMP, ...), or {} when XGBoost is absent or too old to have it."""
    try:
        import xgboost
        info = xgboost.build_info()
        return info if isinstance(info, dict) else {}
    except Exception:                                              # noqa: BLE001
        return {}


def xgb_version() -> Optional[str]:
    try:
        import xgboost
        return str(xgboost.__version__)
    except Exception:                                              # noqa: BLE001
        return None


def _probe_xgb_gpu(timeout: int = 180) -> Optional[str]:
    """None when a two-round GPU fit succeeds, else a short failure string."""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _XGB_GPU_PROBE],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"probe timed out after {timeout}s"
    except Exception as exc:                                       # noqa: BLE001
        return f"probe could not run: {exc}"
    if proc.returncode == 0:
        return None
    desc = f"exit {proc.returncode}"
    if proc.returncode < 0:
        desc += f" (killed by signal {-proc.returncode})"
    tail = [ln for ln in (proc.stderr or "").strip().splitlines() if ln.strip()][-3:]
    if tail:
        desc += ": " + " / ".join(tail)
    return desc


def _compute_xgb_status() -> Dict:
    """{device, gpu, build, detail} — the one place the fraud heads' device is
    decided, and the same dict the log line, /api/status and the UI read."""
    bk = backend()
    version = xgb_version()
    info = xgb_build_info()
    # AMD's fork defines both keys; mainline defines USE_CUDA only. Treat a
    # missing key as False, but a missing *build_info* (very old XGBoost) as
    # "unknown" so the NVIDIA path keeps its historical behaviour.
    use_cuda = bool(info.get("USE_CUDA", False))
    use_hip = bool(info.get("USE_HIP", False))
    build = "hip" if use_hip else "cuda" if use_cuda else "cpu" if info else "unknown"

    def out(device: str, detail: str) -> Dict:
        return {"device": device, "gpu": device != "cpu", "backend": bk,
                "build": build, "version": version, "detail": detail}

    forced = os.environ.get("DEMO_XGB_DEVICE", "").strip().lower()
    if forced in ("cuda", "gpu"):
        return out("cuda", "forced by DEMO_XGB_DEVICE")
    if forced == "cpu":
        return out("cpu", "forced by DEMO_XGB_DEVICE=cpu")

    if version is None:
        return out("cpu", "xgboost is not installed")
    if bk == "cpu":
        return out("cpu", "no GPU detected — heads train on CPU")

    if bk == "cuda":
        if build == "cpu":
            return out("cpu", f"xgboost {version} is a CPU-only build "
                              "(reinstall requirements-gpu.txt)")
        # "unknown" (no build_info) keeps the historical NVIDIA behaviour.
        return out("cuda", f"NVIDIA CUDA build (xgboost {version})")

    # ---- ROCm -------------------------------------------------------------
    if not use_hip:
        return out("cpu", f"xgboost {version} has no ROCm/HIP support — heads "
                          "train on CPU (install the AMD build: "
                          "deploy/build_xgboost_rocm.sh)")
    if os.environ.get("DEMO_XGB_GPU_PROBE", "1") == "0":
        return out("cuda", f"AMD ROCm/HIP build (xgboost {version}, probe skipped)")
    failure = _probe_xgb_gpu()
    if failure:
        return out("cpu", f"AMD ROCm/HIP build (xgboost {version}) failed its GPU "
                          f"probe — falling back to CPU: {failure}")
    return out("cuda", f"AMD ROCm/HIP build (xgboost {version})")


def xgb_status(refresh: bool = False) -> Dict:
    """Cached {device, gpu, backend, build, version, detail}. The subprocess
    probe inside runs at most once per process."""
    global _xgb_status
    if _xgb_status is None or refresh:
        _xgb_status = _compute_xgb_status()
    return dict(_xgb_status)


def xgb_device() -> str:
    """"cuda" when the installed XGBoost can really train on this GPU (NVIDIA
    CUDA build on CUDA, AMD HIP build on ROCm — HIP keeps the "cuda" device
    string), "cpu" otherwise. See xgb_status() for the why."""
    return xgb_status()["device"]


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
