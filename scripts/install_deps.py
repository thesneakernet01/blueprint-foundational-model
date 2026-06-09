# SPDX-License-Identifier: Apache-2.0
"""AMP setup task: install the Python deps for the demo.

Two layers, both installed here:
  * requirements-demo.txt — the thin web layer (fastapi/uvicorn/pydantic/joblib).
  * requirements-gpu.txt   — the NVIDIA / GPU stack the engine needs for REAL
    mode (cudf/cuml/cupy, torch/transformers, xgboost/scikit-learn). CUDA itself
    is assumed already present in the runtime; these are only the Python wheels.

Set SKIP_GPU_DEPS=1 to install just the web layer (e.g. a laptop with no GPU,
where the engine falls back to DEMO-FALLBACK mode anyway).
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
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(reqs)],
        check=True,
    )


if __name__ == "__main__":
    _pip_install(DEMO_REQS)
    print("Demo (web-layer) dependencies installed.")

    if os.environ.get("SKIP_GPU_DEPS"):
        print("SKIP_GPU_DEPS set — skipping the NVIDIA/GPU stack.")
    else:
        _pip_install(GPU_REQS)
        print("NVIDIA/GPU dependencies installed.")
