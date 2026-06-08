# SPDX-License-Identifier: Apache-2.0
"""Transaction Foundation Model — live fraud-inference demo backend.

Modular package behind the thin root `app.py` entrypoint:

    config.py    paths, model dims, column views, env helpers (shared with export)
    defaults.py  built-in summary/examples for the pre-export UI
    schemas.py   the Txn request model
    engine.py    the inference engine (REAL + DEMO-FALLBACK)
    app.py       FastAPI app factory + /api/* routes
    server.py    uvicorn entrypoint (Cloudera-aware host/port)
"""

from .app import app, create_app, engine

__all__ = ["app", "create_app", "engine"]
