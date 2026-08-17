# SPDX-License-Identifier: Apache-2.0
"""Run history + progressive training budget for the Model Lifecycle dashboard.

Every successful export appends one record here, so the UI can chart AUC/AP
improvement across training runs and correlate them with Model Registry
versions. Like tfm_demo/settings.py, the store is a small JSON file in the
project root shared across the CML processes (API server, jobs, the
export_for_demo.py CLI shim). It deliberately lives OUTSIDE demo_artifacts/ —
that directory is overwritten by every export and wiped to force clean
rebuilds, while the run history must survive both.

The budget schedule is the demo's "improvement over time" driver: each run is
granted more embedded rows per split and more boosting rounds than the last,
plateauing at the full budget. Seeds are fixed throughout the export, so after
a reset() the same tiers reproduce the same metrics (and re-hit the per-tier
embedding cache), making the demo replayable.
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

from .config import PROJECT_ROOT

_PATH = PROJECT_ROOT / ".runs_history.json"
_LOCK = threading.Lock()

# (embed_max_per_split, xgb n_estimators scale) per run; the last tier repeats
# for every run beyond the schedule.
_TIERS = [
    (4000, 0.35),
    (8000, 0.55),
    (12000, 0.75),
    (16000, 0.90),
    (20000, 1.00),
]

# $EMBED_MAX_PER_SPLIT stays the hard ceiling (same default as export.EMBED_MAX,
# read here directly so importing runs never pulls the export module's deps).
_EMBED_CEILING = int(os.environ.get("EMBED_MAX_PER_SPLIT", "20000"))

_EMPTY_REGISTRY = {
    "registered": False, "model_version": None, "version_id": None,
    "registered_at": None, "deployed": False, "deployed_at": None,
}


def _read() -> Dict:
    try:
        data = json.loads(_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {"version": 1, "runs": []}
    if not isinstance(data, dict) or not isinstance(data.get("runs"), list):
        return {"version": 1, "runs": []}
    return data


def _write(data: Dict) -> None:
    tmp = _PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(_PATH)


def history() -> List[Dict]:
    """All recorded runs, oldest first. Tolerates a missing/corrupt file."""
    return _read()["runs"]


def next_run_index() -> int:
    return len(history()) + 1


def budget_for(run_index: int) -> Dict:
    """The budget a given 1-based run index is granted."""
    i = min(run_index - 1, len(_TIERS) - 1)
    embed_max, scale = _TIERS[i]
    return {"tier": i, "embed_max": min(embed_max, _EMBED_CEILING),
            "xgb_scale": scale}


def next_budget() -> Dict:
    return budget_for(next_run_index())


def schedule() -> List[Dict]:
    """The full tier schedule, for the dashboard's empty-state preview."""
    return [budget_for(i + 1) for i in range(len(_TIERS))]


def append_run(record: Dict) -> Dict:
    """Assign the run index/id and persist the record. Returns the stored copy."""
    with _LOCK:
        data = _read()
        n = len(data["runs"]) + 1
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        record = {
            "run": n,
            "run_id": f"r{n}-{stamp}",
            "registry": dict(_EMPTY_REGISTRY),
            **record,
        }
        data["runs"].append(record)
        _write(data)
    return record


def latest() -> Optional[Dict]:
    runs = history()
    return runs[-1] if runs else None


def update_registry(run_id: str, patch: Dict) -> Optional[Dict]:
    """Merge a patch into one run's registry block (registered/deployed state).
    Returns the updated record, or None if the run_id is unknown (e.g. history
    was reset while a registry job ran) — callers treat that as non-fatal."""
    with _LOCK:
        data = _read()
        for rec in data["runs"]:
            if rec.get("run_id") == run_id:
                rec["registry"] = {**_EMPTY_REGISTRY, **rec.get("registry", {}),
                                   **patch}
                _write(data)
                return rec
    return None


def reset() -> None:
    """Clear the history so the demo replays from tier 0."""
    with _LOCK:
        _write({"version": 1, "runs": []})


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def elapsed(t0: float) -> float:
    return round(time.time() - t0, 1)
