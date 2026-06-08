# SPDX-License-Identifier: Apache-2.0
"""AMP setup task: install the thin demo-layer Python deps.

The heavy NVIDIA stack (torch, transformers, cudf, cuml, xgboost) is expected to
already live in the project's ML Runtime / NeMo container; this only adds the web
layer (fastapi/uvicorn/pydantic/joblib) on top.
"""

import subprocess
import sys
from pathlib import Path

# `__file__` is undefined when this runs as a Cloudera notebook cell; fall back
# to the working directory (the project root by CML convention).
try:
    _ROOT = Path(__file__).resolve().parent.parent
except NameError:
    _ROOT = Path.cwd()
REQS = _ROOT / "requirements-demo.txt"

if __name__ == "__main__":
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(REQS)],
        check=True,
    )
    print("Demo dependencies installed.")
