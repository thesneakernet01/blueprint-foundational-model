# SPDX-License-Identifier: Apache-2.0
"""One-time GPU memory configuration so REAL mode fits a small card — the
deployed CML cluster schedules a 24 GB L4, not the 80 GB A100/H100 the
blueprint assumes.

The checkpoint itself is tiny (~56 MB); what kills smaller cards is allocator
fragmentation — cuDF/RMM, CuPy, torch and XGBoost each carving a private arena
out of the same device — plus cuDF holding full temporal splits resident. So:

  * RMM pool allocator — one growable pool (1 GiB initial, $RMM_INITIAL_POOL_BYTES
    to override) that cuDF, CuPy and (best-effort) XGBoost all share.
  * cuDF spilling — cold device buffers page to host RAM under pressure instead
    of raising OOM. Must be enabled before the first frame is created.
  * torch expandable segments — keeps the torch caching allocator from pinning
    fragmented blocks it can't reuse (env must be set before torch's first CUDA
    allocation, so we setdefault it here).

Idempotent and safe with no GPU stack installed (the demo-fallback path).
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Optional

from .config import log

# numba's CUDA driver layer has two backends: its own ctypes shim (the default
# in numba-cuda <=0.14) and one routed through the cuda-python bindings. The
# ctypes shim probes the driver for versioned symbols and blindly prefers
# `<name>_v2`; r580-branch (CUDA 13) drivers — which the NeMo 25.09 base image
# targets — newly export cuCtxGetDevice_v2/cuCtxSynchronize_v2 with DIFFERENT
# signatures, so the shim calls them with the old argument list and segfaults
# on numba's very first context attach (NVIDIA/numba-cuda#173, fixed by #185
# only in numba-cuda >=0.8.1 — far past cudf 24.12's <0.0.18 window, so a
# matrix-correct pin still crashes). Every cuDF device->host copy
# (.to_numpy()/.to_pandas() on non-null numerics, via values_host ->
# numba.cuda.as_cuda_array) takes that path, so the first host copy killed the
# server. The cuda-python binding uses properly versioned typedefs and is
# immune — it's how RMM and torch were driving the same GPU all along — so
# route numba through it too. numba reads the env var at first import (cudf
# imports it) and the choice is frozen from then on; this module is imported
# by both engine warmup and run_export ahead of any cuDF use, so a
# module-level setdefault here is early enough.
if os.environ.get("DEMO_NUMBA_NV_BINDING", "1") != "0":
    os.environ.setdefault("NUMBA_CUDA_USE_NVIDIA_BINDING", "1")

_configured = False


def configure_gpu_memory() -> None:
    """Call before the process's first cuDF/CuPy allocation. Idempotent —
    both Engine warmup and run_export call this; whichever runs first wins,
    and we never reinitialize RMM while earlier allocations may be live."""
    global _configured
    if _configured:
        return
    _configured = True

    # Each knob is individually defeatable (set to 0) so a native crash can be
    # bisected from the CML application's environment without a code change.
    use_pool = os.environ.get("DEMO_RMM_POOL", "1") != "0"
    use_spill = os.environ.get("DEMO_CUDF_SPILL", "1") != "0"
    use_expandable = os.environ.get("DEMO_TORCH_EXPANDABLE", "1") != "0"

    # torch reads this when its CUDA caching allocator initialises (first
    # allocation), so a setdefault here is early enough in practice.
    if use_expandable:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    try:
        import cudf
        import rmm
    except Exception as exc:                                       # noqa: BLE001
        log.info("GPU memory config skipped (no RAPIDS stack): %s", exc)
        return

    _patch_cudf_compat(cudf)

    try:
        if use_pool:
            pool = int(os.environ.get("RMM_INITIAL_POOL_BYTES", str(2**30)))
            rmm.reinitialize(pool_allocator=True, initial_pool_size=pool)
            try:
                import cupy
                from rmm.allocators.cupy import rmm_cupy_allocator
                cupy.cuda.set_allocator(rmm_cupy_allocator)
            except Exception:                                      # noqa: BLE001
                pass                            # CuPy keeps its own pool; non-fatal
        if use_spill:
            cudf.set_option("spill", True)
        log.info("GPU memory config: pool=%s spill=%s torch_expandable=%s",
                 use_pool, use_spill, use_expandable)
    except Exception as exc:                                       # noqa: BLE001
        log.warning("GPU memory config failed (continuing on defaults): %s", exc)


def _patch_cudf_compat(cudf) -> None:
    """Shim cuDF APIs the blueprint uses but cudf 24.12 lacks, so the fetched
    `src/` stays verbatim. Currently: TimedeltaProperties.total_seconds()
    (added in cudf 25.02; the tokenizer's financial_pipeline.py calls it to
    derive time_delta_s). No-op once the real method exists."""
    props = cudf.core.series.TimedeltaProperties
    if hasattr(props, "total_seconds"):
        return

    def total_seconds(self):
        # int64 nanosecond ticks -> float seconds; nulls survive both casts.
        return self.series.astype("timedelta64[ns]").astype("int64") / 1e9

    props.total_seconds = total_seconds
    log.info("cuDF compat: shimmed TimedeltaProperties.total_seconds()")


def gpu_stack_versions() -> str:
    """One-line version report of the packages on the device->host copy path.
    Emitted into the export log so a remote crash report pins the installed
    matrix without shell access to the box."""
    from importlib import metadata
    parts = []
    for name in ("cudf-cu12", "numba", "numba-cuda", "cuda-python", "rmm-cu12", "torch"):
        try:
            parts.append(f"{name} {metadata.version(name)}")
        except Exception:                                          # noqa: BLE001
            parts.append(f"{name} ?")
    parts.append("nv_binding=" + os.environ.get("NUMBA_CUDA_USE_NVIDIA_BINDING", "0"))
    return " · ".join(parts)


# A non-null numeric to_pandas() takes cuDF's values_host fast path through
# numba.cuda — the exact call chain that segfaults when the numba driver shim
# is broken (see the module comment).
_CANARY_CODE = ("import cudf; "
                "assert cudf.Series([1, 2, 3]).to_pandas().sum() == 6; "
                "print('host-copy ok')")


def host_copy_canary(timeout: int = 300) -> Optional[str]:
    """Probe the cuDF device->host copy path in a throwaway subprocess.

    A native crash (SIGSEGV) on this path in the export worker thread takes the
    whole server down; in a child process it costs nothing. Returns None when
    the path is healthy, else a short human-readable failure description (exit
    code / signal + the tail of the child's stderr)."""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _CANARY_CODE],
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
    tail = (proc.stderr or "").strip().splitlines()[-12:]
    if tail:
        desc += "\n" + "\n".join(tail)
    return desc
