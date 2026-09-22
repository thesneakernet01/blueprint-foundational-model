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

On the ROCm path the XGBoost that requirements-rocm.txt installs from PyPI is
CPU-only, so _ensure_rocm_xgboost() then tries to replace it with AMD's HIP
build (the one that reports USE_HIP in xgboost.build_info() and trains the
fraud heads on the AMD GPU). Point it at a prebuilt wheel with
$DEMO_XGBOOST_ROCM_WHEEL (a URL or path) or an index with
$DEMO_XGBOOST_ROCM_INDEX, or set $DEMO_XGBOOST_ROCM_BUILD=1 to compile it from
source via deploy/build_xgboost_rocm.sh. With none of those set the CPU wheel
is left in place and the heads train on CPU — the app says which, so this
degrades visibly rather than silently.

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
ROCM_XGB_BUILD = _ROOT / "deploy" / "build_xgboost_rocm.sh"

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


def _xgb_build_info() -> dict:
    """xgboost.build_info() from a subprocess (this installer must keep running
    even when the just-installed xgboost cannot import). {} on any failure."""
    import json
    code = "import json, xgboost; print(json.dumps(xgboost.build_info()))"
    try:
        r = subprocess.run([sys.executable, "-c", code], stdin=subprocess.DEVNULL,
                           capture_output=True, text=True, timeout=120)
        return json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 else {}
    except Exception:                                              # noqa: BLE001
        return {}


def _ensure_rocm_xgboost() -> None:
    """Replace the CPU-only PyPI xgboost with AMD's HIP build, when we're told
    where to get one. Never fails the Job: without a GPU-capable XGBoost the
    heads simply train on CPU (tfm_demo/accel.py detects and reports that)."""
    if _xgb_build_info().get("USE_HIP"):
        print("XGBoost already has ROCm/HIP support (USE_HIP) — leaving it alone.",
              flush=True)
        return

    wheel = os.environ.get("DEMO_XGBOOST_ROCM_WHEEL", "").strip()
    index = os.environ.get("DEMO_XGBOOST_ROCM_INDEX", "").strip()
    build = os.environ.get("DEMO_XGBOOST_ROCM_BUILD", "").strip()

    if wheel:
        print(f"Installing AMD XGBoost wheel: {wheel}", flush=True)
        _pip("install", "--force-reinstall", "--no-deps", wheel, check=False)
    elif index:
        print(f"Installing AMD XGBoost from index: {index}", flush=True)
        _pip("install", "--force-reinstall", "--index-url", index, "xgboost", check=False)
    elif build and build != "0":
        print(f"Building AMD XGBoost from source ({ROCM_XGB_BUILD}) — this takes a while.",
              flush=True)
        subprocess.run(["bash", str(ROCM_XGB_BUILD)], stdin=subprocess.DEVNULL, check=False)
    else:
        print("XGBoost on this ROCm host is the CPU-only PyPI build, so the fraud "
              "heads will train on CPU. Set DEMO_XGBOOST_ROCM_WHEEL / "
              "DEMO_XGBOOST_ROCM_INDEX, or DEMO_XGBOOST_ROCM_BUILD=1 to compile "
              "AMD's HIP build (deploy/build_xgboost_rocm.sh).", flush=True)
        return

    info = _xgb_build_info()
    if info.get("USE_HIP"):
        print("AMD/ROCm XGBoost active — fraud heads will train on the AMD GPU.",
              flush=True)
    else:
        print("WARNING: XGBoost still reports no HIP support "
              f"(build_info={info or 'unavailable'}) — heads will train on CPU.",
              flush=True)


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
            _ensure_rocm_xgboost()
        else:
            _purge_cuda13()
            _pip("install", "-r", str(GPU_REQS))
            print("NVIDIA/GPU dependencies installed.", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
