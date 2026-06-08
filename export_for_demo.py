# SPDX-License-Identifier: Apache-2.0
"""
Export demo artifacts — CLI shim.
=================================

The export is now primarily triggered from the UI (which runs it on the GPU
backend via POST /api/export and streams progress). This script is the optional
command-line equivalent of that same code path.

The implementation lives in `tfm_demo/export.py` (`run_export`). Run this ONCE
inside the NeMo container, from the blueprint repo root, AFTER notebooks 04
(embeddings) and 05 (XGBoost):

    cd <blueprint repo root>
    python tfm-demo/export_for_demo.py

It writes everything into `tfm-demo/demo_artifacts/` and prints the lift numbers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make the demo package importable when run as `python tfm-demo/export_for_demo.py`.
# `__file__` is undefined in a Cloudera notebook cell; fall back to the cwd.
try:
    _ROOT = Path(__file__).resolve().parent
except NameError:
    _ROOT = Path.cwd()
sys.path.insert(0, str(_ROOT))
from tfm_demo.export import run_export  # noqa: E402


if __name__ == "__main__":
    summary = run_export(progress=print)
    print("\n" + json.dumps(summary["lift"], indent=2))
