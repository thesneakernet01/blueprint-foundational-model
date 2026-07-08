# SPDX-License-Identifier: Apache-2.0
"""UI-configurable runtime settings, persisted as JSON in the project root.

The Data dialog configures WHERE the training splits live — either an Impala
database (via a CML data connection) or a VAST S3 bucket written directly with
boto3 — and the settings must be visible to BOTH the API server (training/
export reads) and the data-prep script (writes) — on CML those run as separate
processes sharing the project filesystem, so a small JSON file is the simplest
shared store. Env vars seed the defaults for a fresh project ($IMPALA_* /
$VAST_*, same names as scripts/vast_probe.py); a UI save overrides them from
then on.

The file may hold the VAST secret key, so it is written 0600.
"""

from __future__ import annotations

import json
import os
import re
import threading
from typing import Dict

from .config import PROJECT_ROOT

_PATH = PROJECT_ROOT / ".data_settings.json"
_LEGACY_PATH = PROJECT_ROOT / ".impala_settings.json"   # pre-VAST installs
_LOCK = threading.Lock()

BACKENDS = ("impala", "vast")

# Interpolated into SQL as `db`.`table` — restrict to Impala identifier chars
# so a UI value can never break out of the backticks.
_DB_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _read_file() -> Dict:
    for path in (_PATH, _LEGACY_PATH):
        try:
            data = json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        if path is _LEGACY_PATH:
            # Old flat {connection, database} shape.
            data = {"impala": data}
        return data if isinstance(data, dict) else {}
    return {}


def get_data_settings() -> Dict:
    """Full data-target settings; every field is a string ('' when unset).

    {'backend': 'impala'|'vast',
     'impala': {'connection', 'database'},
     'vast': {'endpoint', 'bucket', 'prefix', 'access_key', 'secret_key'}}
    """
    data = _read_file()
    impala = data.get("impala") or {}
    vast = data.get("vast") or {}
    env = os.environ.get
    backend = data.get("backend") or env("DATA_BACKEND") or (
        "vast" if env("VAST_BUCKET") else "impala")
    if backend not in BACKENDS:
        backend = "impala"
    return {
        "backend": backend,
        "impala": {
            "connection": impala.get("connection") or env("IMPALA_CONNECTION_NAME", ""),
            "database": impala.get("database") or env("IMPALA_DATABASE", ""),
        },
        "vast": {
            "endpoint": vast.get("endpoint") or env("VAST_ENDPOINT", ""),
            "bucket": vast.get("bucket") or env("VAST_BUCKET", ""),
            "prefix": (vast.get("prefix") or env("VAST_PATH", "")).strip("/"),
            "access_key": vast.get("access_key") or env("VAST_ACCESS_KEY", ""),
            "secret_key": vast.get("secret_key") or env("VAST_SECRET_KEY", ""),
        },
    }


def save_data_settings(cfg: Dict) -> Dict:
    """Validate and persist the Data-dialog settings; returns the saved view.

    An empty vast.secret_key means "keep the stored one" so the UI never has to
    echo the secret back. Raises ValueError on a bad backend or database name.
    """
    backend = (cfg.get("backend") or "impala").strip()
    if backend not in BACKENDS:
        raise ValueError(f"unknown storage backend {backend!r} — use one of {BACKENDS}")

    impala = {k: str((cfg.get("impala") or {}).get(k, "")).strip()
              for k in ("connection", "database")}
    if impala["database"] and not _DB_RE.match(impala["database"]):
        raise ValueError(
            f"invalid Impala database name {impala['database']!r} — use letters, "
            "digits and underscores only"
        )

    vast = {k: str((cfg.get("vast") or {}).get(k, "")).strip()
            for k in ("endpoint", "bucket", "prefix", "access_key", "secret_key")}
    vast["prefix"] = vast["prefix"].strip("/")

    with _LOCK:
        if not vast["secret_key"]:                    # keep the stored secret
            vast["secret_key"] = (_read_file().get("vast") or {}).get("secret_key", "")
        payload = {"backend": backend, "impala": impala, "vast": vast}
        tmp = _PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.chmod(0o600)                              # may hold the secret key
        tmp.replace(_PATH)
        _LEGACY_PATH.unlink(missing_ok=True)
    return payload


# --------------------------------------------------------------------------- #
# per-backend views
# --------------------------------------------------------------------------- #
def get_impala_settings() -> Dict[str, str]:
    """{'connection': str, 'database': str} — empty strings when unset."""
    return get_data_settings()["impala"]


def get_vast_settings() -> Dict[str, str]:
    """{'endpoint', 'bucket', 'prefix', 'access_key', 'secret_key'}."""
    return get_data_settings()["vast"]


def impala_configured() -> bool:
    s = get_impala_settings()
    return bool(s["connection"] and s["database"])


def vast_configured() -> bool:
    s = get_vast_settings()
    return bool(s["endpoint"] and s["bucket"] and s["access_key"] and s["secret_key"])
