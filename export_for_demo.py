# SPDX-License-Identifier: Apache-2.0
"""
Export demo artifacts — CLI shim.

The UI now triggers this via POST /api/export; this is the CLI equivalent,
wrapping `run_export` from `tfm_demo/export.py`. Run once, from the repo root,
after notebooks 04 (embeddings) and 05 (XGBoost):

    python export_for_demo.py

Writes to `demo_artifacts/` and prints the lift numbers.
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
