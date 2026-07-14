# SPDX-License-Identifier: Apache-2.0
"""FastAPI application: wiring, CORS, and the /api/* routes. The model work
lives in `Engine` (engine.py); this module is just the HTTP surface.
"""

from __future__ import annotations

import faulthandler
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI

# The GPU stack (RAPIDS/numba/torch) can die with a native signal (SIGSEGV/
# SIGABRT) that produces no Python traceback — the backend just vanishes with
# e.g. exit -11 and the export log goes silent. faulthandler prints every
# thread's Python stack to stderr on those signals, so the CML app log shows
# exactly which line was executing.
faulthandler.enable(file=sys.stderr, all_threads=True)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import nexus
from .config import MODEL_DIR, cors_origins
from .engine import Engine
from .jobs import ExportManager, PrepManager
from .schemas import DataConfig, NexusConfig, Txn
from .settings import get_data_settings, save_data_settings

# Single process-wide engine; warmed up on startup by the lifespan hook.
engine = Engine()
# Background export runner; reloads `engine` on success.
exporter = ExportManager(engine)
# Background TabFormer -> storage data load (subprocess wrapper).
preparer = PrepManager()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    engine.warmup()
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="TFM Live Demo", lifespan=lifespan)

    # The React SPA is served separately (its own origin), so it calls this API
    # cross-origin. $CORS_ORIGINS is a comma-separated allowlist; default "*" is
    # fine for a demo. Set it to the SPA's URL in a locked-down deployment.
    origins = cors_origins()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials="*" not in origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/status")
    def status() -> JSONResponse:
        return JSONResponse({
            "mode": engine.mode,
            "gpu": engine.gpu,
            "detail": engine.detail,
            "model_dir": str(MODEL_DIR),
            "nexus": nexus.mode(),
        })

    @app.get("/api/summary")
    def summary() -> JSONResponse:
        return JSONResponse(engine.summary)

    @app.get("/api/examples")
    def examples() -> JSONResponse:
        return JSONResponse(engine.examples)

    @app.get("/api/umap")
    def umap() -> JSONResponse:
        return JSONResponse(engine.umap_background)

    @app.post("/api/score")
    def score(txn: Txn) -> JSONResponse:
        return JSONResponse(engine.score(txn.to_txn()))

    @app.post("/api/export")
    def start_export() -> JSONResponse:
        """Run the artifact export on the backend (GPU). Returns immediately;
        poll /api/export/status for streaming progress. 409 if already running."""
        started = exporter.start()
        return JSONResponse(
            {"started": started, **exporter.status()},
            status_code=202 if started else 409,
        )

    @app.get("/api/export/status")
    def export_status() -> JSONResponse:
        return JSONResponse(exporter.status())

    # ---- data target (settings come from the UI's Data dialog) -------------
    def _masked_settings() -> dict:
        """Settings for the UI: the VAST secret never leaves the server —
        replaced by a secret_set flag."""
        s = get_data_settings()
        s["vast"] = {**s["vast"], "secret_key": "",
                     "secret_set": bool(s["vast"]["secret_key"])}
        return s

    @app.get("/api/data")
    def data_settings() -> JSONResponse:
        """Current storage target. No connectivity probe — GETs stay fast;
        POST (save) is what tests the connection."""
        return JSONResponse(_masked_settings())

    @app.post("/api/data")
    def data_configure(cfg: DataConfig) -> JSONResponse:
        """Save the storage target, then probe it and report per-split row
        counts. A cold CDW warehouse can take a while to wake — the UI shows a
        spinner for the duration."""
        from . import storage  # lazy: pulls in the backend client stack

        try:
            save_data_settings(cfg.model_dump())
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=422)
        return JSONResponse({**_masked_settings(), "check": storage.check()})

    # ---- NEXUS head mode (settable from the Build-artifacts dialog) --------
    @app.get("/api/nexus")
    def nexus_settings() -> JSONResponse:
        return JSONResponse(nexus.settings())

    @app.post("/api/nexus")
    def nexus_configure(cfg: NexusConfig) -> JSONResponse:
        """Persist the NEXUS head mode. Takes effect immediately — mode is
        read per export run and per scoring request, no restart needed."""
        try:
            nexus.save_mode(cfg.mode)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=422)
        return JSONResponse(nexus.settings())

    @app.post("/api/data/prepare")
    def start_prepare() -> JSONResponse:
        """Download TabFormer and (re)load the splits into storage. Returns
        immediately; poll /api/data/prepare/status. 409 if already running."""
        started = preparer.start()
        return JSONResponse(
            {"started": started, **preparer.status()},
            status_code=202 if started else 409,
        )

    @app.get("/api/data/prepare/status")
    def prepare_status() -> JSONResponse:
        return JSONResponse(preparer.status())

    @app.get("/")
    def root() -> JSONResponse:
        """Health / info root. The UI is the standalone React SPA in frontend/,
        which calls this service's /api/* endpoints cross-origin."""
        return JSONResponse({
            "service": "TFM Live Demo API",
            "mode": engine.mode,
            "gpu": engine.gpu,
            "endpoints": ["/api/status", "/api/summary", "/api/examples",
                          "/api/umap", "/api/score", "/api/data"],
        })

    return app


# Module-level app so `uvicorn tfm_demo.app:app` (and the root app.py shim) work.
app = create_app()
