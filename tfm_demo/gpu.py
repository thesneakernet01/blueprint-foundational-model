# SPDX-License-Identifier: Apache-2.0
"""One-time GPU memory configuration so REAL mode fits a 48 GB L40S (not just
an 80 GB A100/H100).

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

from .config import log

_configured = False


def configure_gpu_memory() -> None:
    """Call before the process's first cuDF/CuPy allocation. Idempotent —
    both Engine warmup and run_export call this; whichever runs first wins,
    and we never reinitialize RMM while earlier allocations may be live."""
    global _configured
    if _configured:
        return
    _configured = True

    # torch reads this when its CUDA caching allocator initialises (first
    # allocation), so a setdefault here is early enough in practice.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    try:
        import cudf
        import rmm
    except Exception as exc:                                       # noqa: BLE001
        log.info("GPU memory config skipped (no RAPIDS stack): %s", exc)
        return

    try:
        pool = int(os.environ.get("RMM_INITIAL_POOL_BYTES", str(2**30)))
        rmm.reinitialize(pool_allocator=True, initial_pool_size=pool)
        cudf.set_option("spill", True)
        try:
            import cupy
            from rmm.allocators.cupy import rmm_cupy_allocator
            cupy.cuda.set_allocator(rmm_cupy_allocator)
        except Exception:                                          # noqa: BLE001
            pass                                # CuPy keeps its own pool; non-fatal
        log.info("RMM pool (%d MiB initial) + cuDF spill enabled", pool // 2**20)
    except Exception as exc:                                       # noqa: BLE001
        log.warning("GPU memory config failed (continuing on defaults): %s", exc)
