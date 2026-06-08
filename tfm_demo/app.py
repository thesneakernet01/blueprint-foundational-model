# SPDX-License-Identifier: Apache-2.0
"""FastAPI application: wiring, CORS, and the /api/* routes. The model work
lives in `Engine` (engine.py); this module is just the HTTP surface.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import MODEL_DIR, cors_origins
from .engine import Engine
from .jobs import ExportManager
from .schemas import Txn

# Single process-wide engine; warmed up on startup by the lifespan hook.
engine = Engine()
# Background export runner; reloads `engine` on success.
exporter = ExportManager(engine)


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

    @app.get("/")
    def root() -> JSONResponse:
        """Health / info root. The UI is the standalone React SPA in frontend/,
        which calls this service's /api/* endpoints cross-origin."""
        return JSONResponse({
            "service": "TFM Live Demo API",
            "mode": engine.mode,
            "gpu": engine.gpu,
            "endpoints": ["/api/status", "/api/summary", "/api/examples",
                          "/api/umap", "/api/score"],
        })

    return app


# Module-level app so `uvicorn tfm_demo.app:app` (and the root app.py shim) work.
app = create_app()
