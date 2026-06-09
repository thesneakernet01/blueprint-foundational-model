# SPDX-License-Identifier: Apache-2.0
"""AMP setup task: install the Python deps for the demo.

Two layers, both installed here:
  * requirements-demo.txt — the thin web layer (fastapi/uvicorn/pydantic/joblib).
  * requirements-gpu.txt   — the NVIDIA / GPU stack the engine needs for REAL
    mode (cudf/cuml/cupy, torch/transformers, xgboost/scikit-learn). CUDA itself
    is assumed already present in the runtime; these are only the Python wheels.

Set SKIP_GPU_DEPS=1 to install just the web layer (e.g. a laptop with no GPU,
where the engine falls back to DEMO-FALLBACK mode anyway).

NOTE on the hard exit: after a big GPU install some CUDA/RAPIDS wheels leave a
lingering non-daemon thread or background helper alive in the interpreter, so a
plain return can leave the CML Job process running forever even though the work
is done. We flush and os._exit(0) to guarantee the Job terminates as soon as the
installs succeed.
"""

import os
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


def _pip_install(reqs: Path) -> None:
    # stdin=DEVNULL so a build-from-sdist can never block waiting on a prompt.
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-input", "-r", str(reqs)],
        stdin=subprocess.DEVNULL,
        check=True,
    )


def main() -> None:
    _pip_install(DEMO_REQS)
    print("Demo (web-layer) dependencies installed.", flush=True)

    if os.environ.get("SKIP_GPU_DEPS"):
        print("SKIP_GPU_DEPS set — skipping the NVIDIA/GPU stack.", flush=True)
    else:
        _pip_install(GPU_REQS)
        print("NVIDIA/GPU dependencies installed.", flush=True)


if __name__ == "__main__":
    main()
    # Force-terminate: the work is done and committed to disk; don't let a
    # lingering library thread keep the Job process alive (see module docstring).
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
