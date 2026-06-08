# SPDX-License-Identifier: Apache-2.0
"""AMP setup task: install the thin demo-layer Python deps.

The heavy NVIDIA stack (torch, transformers, cudf, cuml, xgboost) is expected to
already live in the project's ML Runtime / NeMo container; this only adds the web
layer (fastapi/uvicorn/pydantic/joblib) on top.
"""

import subprocess
import sys
from pathlib import Path

REQS = Path(__file__).resolve().parent.parent / "requirements-demo.txt"

if __name__ == "__main__":
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(REQS)],
        check=True,
    )
    print("Demo dependencies installed.")
