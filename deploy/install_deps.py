# SPDX-License-Identifier: Apache-2.0
"""AMP setup task: install the Python deps for the demo.

Layers, all installed here:
  * requirements-demo.txt — the thin web layer (fastapi/uvicorn/pydantic/joblib).
  * requirements-gpu.txt  — the NVIDIA / GPU stack the engine needs for REAL
    mode (cudf/cuml/cupy, torch/transformers, xgboost/scikit-learn). CUDA itself
    is assumed already present in the runtime; these are only the Python wheels.
  * requirements-rocm.txt — the AMD ROCm equivalent, installed instead of the
    NVIDIA stack when an AMD GPU is detected (see _accel_target()). No RAPIDS
    (no ROCm port) — tfm_demo/rapids_shim.py stands in with pandas/scikit-learn
    at runtime; see that module and tfm_demo/accel.py for the detection this
    mirrors.

torch is pinned to a CUDA 12 build on the NVIDIA path (see requirements-gpu.txt).
Before installing that layer we purge any CUDA 13 packages a previous
unpinned-torch install left behind, so re-runs converge on a clean cu12-only
environment. (Not applicable to the ROCm path.)

Set SKIP_GPU_DEPS=1 to install just the web layer (e.g. a laptop with no GPU,
where the engine falls back to DEMO-FALLBACK mode anyway). Set DEMO_ACCEL=cuda
or DEMO_ACCEL=rocm to force which GPU layer installs instead of auto-detecting.

This script exits normally (no os._exit): CML runs Job scripts inside the
engine's own process, and a hard low-level exit bypasses the harness's
completion handling and gets reported as an abnormal "status 1". The big GPU
install is just slow (multi-GB wheels) — let it finish and return cleanly.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

# `__file__` is undefined when this runs as a Cloudera notebook cell; fall back
# to the working directory (the project root by CML convention).
try:
    _ROOT = Path(__file__).resolve().parent.parent
except NameError:
    _ROOT = Path.cwd()

DEMO_REQS = _ROOT / "requirements-demo.txt"
GPU_REQS = _ROOT / "requirements-gpu.txt"
ROCM_REQS = _ROOT / "requirements-rocm.txt"

# CUDA-13 packages an unpinned `torch` pulls in. Our cu12 torch pin brings its
# own *-cu12 deps, so these are orphans once torch is downgraded. Listed by the
# exact distribution names (the unsuffixed nvidia-* ones are the cu13 packaging;
# the cu12 equivalents are nvidia-*-cu12 and are NOT in this set).
CUDA13_PURGE = [
    "cuda-toolkit",
    "nvidia-cublas",
    "nvidia-cuda-cupti",
    "nvidia-cuda-nvrtc",
    "nvidia-cuda-runtime",
    "nvidia-cufft",
    "nvidia-cufile",
    "nvidia-curand",
    "nvidia-cusolver",
    "nvidia-cusparse",
    "nvidia-nvjitlink",
    "nvidia-nvtx",
]


def _pip(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    # stdin=DEVNULL so a build-from-sdist can never block waiting on a prompt.
    return subprocess.run(
        [sys.executable, "-m", "pip", "--no-input", *args],
        stdin=subprocess.DEVNULL,
        check=check,
    )


def _installed() -> set[str]:
    out = subprocess.run(
        [sys.executable, "-m", "pip", "list", "--format=freeze"],
        stdin=subprocess.DEVNULL, capture_output=True, text=True, check=True,
    ).stdout
    return {line.split("==", 1)[0].lower() for line in out.splitlines() if "==" in line}


def _purge_cuda13() -> None:
    """Remove leftover CUDA-13 wheels so the env converges on cu12-only."""
    installed = _installed()
    targets = sorted(
        {p for p in installed if p.endswith("-cu13")}
        | {p for p in CUDA13_PURGE if p in installed}
    )
    if not targets:
        print("No CUDA 13 leftovers to purge.", flush=True)
        return
    print(f"Purging CUDA 13 leftovers: {', '.join(targets)}", flush=True)
    _pip("uninstall", "-y", *targets, check=False)  # don't fail the Job on this


def _accel_target() -> str:
    """"cuda" or "rocm" — which requirements-*.txt to install. $DEMO_ACCEL
    overrides outright (same "individually defeatable" convention as
    tfm_demo/gpu.py's DEMO_RMM_POOL etc.); otherwise auto-detect from what's
    on the host, defaulting to "cuda" (today's only behavior) when neither an
    NVIDIA nor an AMD GPU is detectable, so existing CUDA deployments are
    unaffected by this branch existing at all."""
    override = os.environ.get("DEMO_ACCEL", "").strip().lower()
    if override in ("cuda", "rocm"):
        return override
    if shutil.which("rocminfo") or Path("/dev/kfd").exists():
        return "rocm"
    return "cuda"


def main() -> None:
    _pip("install", "-r", str(DEMO_REQS))
    print("Demo (web-layer) dependencies installed.", flush=True)

    if os.environ.get("SKIP_GPU_DEPS"):
        print("SKIP_GPU_DEPS set — skipping the GPU stack.", flush=True)
    else:
        accel = _accel_target()
        if accel == "rocm":
            _pip("install", "-r", str(ROCM_REQS))
            print("AMD/ROCm dependencies installed.", flush=True)
        else:
            _purge_cuda13()
            _pip("install", "-r", str(GPU_REQS))
            print("NVIDIA/GPU dependencies installed.", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
