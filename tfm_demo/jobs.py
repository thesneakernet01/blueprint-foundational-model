# SPDX-License-Identifier: Apache-2.0
"""Background job managers for the two long-running UI actions.

Both run in a daemon thread and stream a line log the UI polls:
  * ExportManager — trains XGBoost heads / PCA / UMAP on the GPU (minutes); on
    success the engine is re-warmed so the new artifacts go live without a
    server restart.
  * PrepManager — downloads TabFormer and loads the temporal splits into the
    UI-configured storage target (Impala database or VAST S3), by running
    scripts/prepare_data.py in a subprocess (keeps the multi-GB pandas chunks
    out of the server process and makes a native crash non-fatal to the API).
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import threading
import time
from typing import Dict, List, Optional

from .config import PROJECT_ROOT, log
from .resources import sample as sample_resources


def _jsonsafe(obj):
    """Coerce a value tree into something Starlette's JSONResponse (allow_nan=
    False) can serialise: numpy scalars -> python, NaN/Inf -> None. Without this
    a single NaN/Inf or numpy float32/int from a build result 500s the status
    poll, leaving the UI stuck on the build dialog (see export summary `lift`)."""
    if isinstance(obj, dict):
        return {k: _jsonsafe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonsafe(v) for v in obj]
    item = getattr(obj, "item", None)            # numpy scalar -> python scalar
    if callable(item) and getattr(obj, "ndim", None) == 0:
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None                              # NaN / +-Inf are not JSON-compliant
    return obj


class JobManager:
    """One-at-a-time background job with a streaming line log + status poll."""

    name = "job"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self.state: str = "idle"          # idle | running | done | error
        self.log: List[str] = []
        self.summary: Optional[Dict] = None
        self.error: Optional[str] = None
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None

    def start(self) -> bool:
        """Kick off the job. Returns False if one is already running."""
        with self._lock:
            if self.state == "running":
                return False
            self.state = "running"
            self.log = []
            self.summary = None
            self.error = None
            self.started_at = time.time()
            self.finished_at = None
        self._thread = threading.Thread(
            target=self._run, name=f"tfm-{self.name}", daemon=True)
        self._thread.start()
        return True

    def _emit(self, msg: str) -> None:
        log.info("[%s] %s", self.name, msg)
        self.log.append(msg)

    def _work(self) -> Optional[Dict]:
        raise NotImplementedError

    def _run(self) -> None:
        try:
            self.summary = self._work()
            self.state = "done"
        except Exception as exc:                       # noqa: BLE001
            self.error = str(exc)
            self._emit(f"ERROR: {exc}")
            self.state = "error"
            log.exception("%s failed", self.name)
        finally:
            self.finished_at = time.time()

    def status(self) -> Dict:
        elapsed = None
        if self.started_at is not None:
            end = self.finished_at or time.time()
            elapsed = round(end - self.started_at, 1)
        return _jsonsafe({
            "state": self.state,
            "log": list(self.log),               # snapshot: the job thread mutates this
            "summary": self.summary,
            "error": self.error,
            "elapsed_sec": elapsed,
            # Live CPU/RAM/GPU snapshot so the dialogs can show meters while
            # the job runs (the UI polls this endpoint anyway).
            "resources": sample_resources(),
        })


class ExportManager(JobManager):
    name = "export"

    def __init__(self, engine) -> None:
        super().__init__()
        self.engine = engine

    def _work(self) -> Optional[Dict]:
        # Imported lazily — pulls in cudf/xgboost/torch only when an export runs.
        from .export import run_export

        summary = run_export(progress=self._emit)
        self._emit("Reloading artifacts into the live engine ...")
        self.engine.warmup()
        self._emit(f"Engine reloaded — mode: {self.engine.mode}")
        return summary

    def status(self) -> Dict:
        return {**super().status(), "engine_mode": self.engine.mode}


class PrepManager(JobManager):
    name = "prepare"

    def _work(self) -> Optional[Dict]:
        from . import storage

        if not storage.configured():
            raise RuntimeError(
                "Configure the storage target first (Data dialog)."
            )
        script = PROJECT_ROOT / "scripts" / "prepare_data.py"
        # PREP_FORCE=1: a UI click means "load/refresh the data", so re-ingest
        # even when the tables already exist.
        env = {**os.environ, "PREP_FORCE": "1"}
        self._emit(f"Starting data preparation (download + split + load into "
                   f"{storage.target()}) ...")
        proc = subprocess.Popen(
            [sys.executable, str(script)],
            cwd=str(PROJECT_ROOT), env=env, text=True, bufsize=1,
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                self._emit(line)
        code = proc.wait()
        if code != 0:
            raise RuntimeError(f"prepare_data.py exited with status {code} — see log above.")
        # Fresh table counts for the dialog's summary panel.
        return storage.check()
