# SPDX-License-Identifier: Apache-2.0
"""Static configuration: filesystem paths, model dimensions, the column views
the two model families expect, and small environment helpers.

This module is the single source of truth for these constants — both the live
API engine and the offline `export_for_demo.py` exporter import from here so the
demo and the notebook stay in lockstep.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# ---- paths -----------------------------------------------------------------
# Two supported layouts:
#   * nested   — tfm_demo/ lives inside the cloned blueprint repo, next to its
#                src/, models/ and data/ (REPO_ROOT = the blueprint root).
#   * standalone — the demo is its own project (e.g. a CML AMP installed from a
#                git URL); src/, models/ and data/ sit INSIDE the project.
# So we resolve src/model/data against the project first, then the parent repo,
# and let env vars override outright. (Previously these were hard-wired to the
# parent, which on CML resolved to /home/… and missed the in-project data.)
# In a Cloudera notebook/interactive session the module may be exec'd as a cell,
# where `__file__` is undefined — fall back to the working directory (the project
# root, by CML convention) in that case.
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_ROOT = Path.cwd()
REPO_ROOT = PROJECT_ROOT.parent                       # the cloned blueprint repo
ARTIFACTS = PROJECT_ROOT / "demo_artifacts"


def _resolve(env_var: str, *candidates: Path) -> Path:
    """$env_var if set; else the first candidate that exists; else candidate[0]."""
    override = os.environ.get(env_var)
    if override:
        return Path(override).expanduser().resolve()
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


# $MODEL_DIR / $DATA_DIR override; otherwise prefer in-project, then parent repo.
MODEL_DIR = _resolve(
    "MODEL_DIR",
    PROJECT_ROOT / "models" / "decoder-foundation-model",
    REPO_ROOT / "models" / "decoder-foundation-model",
)
DATA_DIR = _resolve("DATA_DIR", PROJECT_ROOT / "data", REPO_ROOT / "data")

# ---- model dims ------------------------------------------------------------
MAX_LENGTH = 128
MERCHANT_HASH_SIZE = 2000
PCA_DIM = 64

# Make the blueprint's src/ importable from either layout.
for _p in (PROJECT_ROOT, REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ---- column views ----------------------------------------------------------
# Order matters: the tokenizer and the raw-feature preprocessor were each fit on
# these exact column orders in notebooks 04 / 05.
TOKENIZER_COLS = ["Amount", "Merchant Name", "MCC", "Year", "Month", "Day",
                  "Time", "Card", "Use Chip", "Zip", "Merchant State", "User"]
RAW_FEATURE_COLS = ["User", "Card", "Year", "Month", "Day", "Hour", "Amount",
                    "Use Chip", "Merchant Name", "Merchant City",
                    "Merchant State", "Zip", "MCC"]
FRAUD_COL = "Is Fraud?"

# ---- logging ---------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("tfm-demo")


# ---- environment helpers ---------------------------------------------------
def cors_origins() -> list[str]:
    """Comma-separated CORS allowlist from $CORS_ORIGINS (default '*')."""
    return [o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",") if o.strip()]


def server_host_port() -> tuple[str, int]:
    """Resolve (host, port) for uvicorn.

    Cloudera ML (CML/CDSW) injects CDSW_APP_PORT and expects the app to bind it
    on 127.0.0.1; the platform's reverse proxy exposes the public URL. Outside
    CML we fall back to $PORT, then 8000, on $HOST (default 0.0.0.0).
    """
    in_cml = "CDSW_APP_PORT" in os.environ
    port = int(os.environ.get("CDSW_APP_PORT") or os.environ.get("PORT") or 8000)
    host = "127.0.0.1" if in_cml else os.environ.get("HOST", "0.0.0.0")
    return host, port
