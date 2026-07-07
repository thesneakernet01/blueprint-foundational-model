# SPDX-License-Identifier: Apache-2.0
"""UI-configurable runtime settings, persisted as JSON in the project root.

The Impala connection name and target database are entered in the UI (the
"Data" dialog) and must be visible to BOTH the API server (training/export
reads) and the data-prep script (writes) — on CML those run as separate
processes sharing the project filesystem, so a small JSON file is the simplest
shared store. Env vars ($IMPALA_CONNECTION_NAME / $IMPALA_DATABASE) seed the
defaults for a fresh project; a UI save overrides them from then on.
"""

from __future__ import annotations

import json
import os
import re
import threading
from typing import Dict

from .config import PROJECT_ROOT

_PATH = PROJECT_ROOT / ".impala_settings.json"
_LOCK = threading.Lock()

# Interpolated into SQL as `db`.`table` — restrict to Impala identifier chars
# so a UI value can never break out of the backticks.
_DB_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def get_impala_settings() -> Dict[str, str]:
    """{'connection': str, 'database': str} — empty strings when unset."""
    data: Dict[str, str] = {}
    try:
        data = json.loads(_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return {
        "connection": data.get("connection") or os.environ.get("IMPALA_CONNECTION_NAME", ""),
        "database": data.get("database") or os.environ.get("IMPALA_DATABASE", ""),
    }


def save_impala_settings(connection: str, database: str) -> Dict[str, str]:
    """Validate and persist; returns the saved settings. Raises ValueError on a
    database name that isn't a plain Impala identifier."""
    connection = connection.strip()
    database = database.strip()
    if database and not _DB_RE.match(database):
        raise ValueError(
            f"invalid Impala database name {database!r} — use letters, digits "
            "and underscores only"
        )
    with _LOCK:
        tmp = _PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"connection": connection, "database": database}, indent=2))
        tmp.replace(_PATH)
    return {"connection": connection, "database": database}


def impala_configured() -> bool:
    s = get_impala_settings()
    return bool(s["connection"] and s["database"])
