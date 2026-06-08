# SPDX-License-Identifier: Apache-2.0
"""
Transaction Foundation Model — Live Demo Backend (entrypoint)
============================================================

Thin entrypoint. The implementation is the modular `tfm_demo/` package:

    tfm_demo/config.py    paths, model dims, column views, env helpers
    tfm_demo/defaults.py  built-in summary/examples before export_for_demo runs
    tfm_demo/schemas.py   the Txn request model
    tfm_demo/engine.py    the inference engine (REAL + DEMO-FALLBACK modes)
    tfm_demo/app.py       FastAPI app factory + /api/* routes
    tfm_demo/server.py    uvicorn entrypoint (Cloudera-aware host/port)

This service is a pure JSON API; the UI is the standalone React SPA in
`frontend/`, which calls /api/* cross-origin. See README_DEMO.md.

Run:
    pip install -r requirements-demo.txt
    python app.py                  # or: uvicorn app:app
"""

from __future__ import annotations

# `app` is what ASGI servers import (uvicorn app:app); `engine` is re-exported
# for tests / introspection.
from tfm_demo import app, engine  # noqa: F401
from tfm_demo.server import main

if __name__ == "__main__":
    main()
