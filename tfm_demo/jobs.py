# SPDX-License-Identifier: Apache-2.0
"""Background export job manager.

The export trains XGBoost heads / PCA / UMAP on the GPU and takes minutes, so the
API runs it in a daemon thread and the UI polls for progress. On success the
engine is re-warmed so the new artifacts go live (REAL mode + fresh metrics)
without a server restart.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional

from .config import log


class ExportManager:
    def __init__(self, engine) -> None:
        self.engine = engine
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self.state: str = "idle"          # idle | running | done | error
        self.log: List[str] = []
        self.summary: Optional[Dict] = None
        self.error: Optional[str] = None
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None

    def start(self) -> bool:
        """Kick off an export. Returns False if one is already running."""
        with self._lock:
            if self.state == "running":
                return False
            self.state = "running"
            self.log = []
            self.summary = None
            self.error = None
            self.started_at = time.time()
            self.finished_at = None
        self._thread = threading.Thread(target=self._run, name="tfm-export", daemon=True)
        self._thread.start()
        return True

    def _emit(self, msg: str) -> None:
        log.info("[export] %s", msg)
        self.log.append(msg)

    def _run(self) -> None:
        try:
            # Imported lazily — pulls in cudf/xgboost/torch only when an export runs.
            from .export import run_export

            self.summary = run_export(progress=self._emit)
            self._emit("Reloading artifacts into the live engine ...")
            self.engine.warmup()
            self._emit(f"Engine reloaded — mode: {self.engine.mode}")
            self.state = "done"
        except Exception as exc:                       # noqa: BLE001
            self.error = str(exc)
            self._emit(f"ERROR: {exc}")
            self.state = "error"
            log.exception("export failed")
        finally:
            self.finished_at = time.time()

    def status(self) -> Dict:
        elapsed = None
        if self.started_at is not None:
            end = self.finished_at or time.time()
            elapsed = round(end - self.started_at, 1)
        return {
            "state": self.state,
            "log": self.log,
            "summary": self.summary,
            "error": self.error,
            "elapsed_sec": elapsed,
            "engine_mode": self.engine.mode,
        }
