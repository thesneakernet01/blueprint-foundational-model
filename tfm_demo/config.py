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
# tfm_demo/ lives inside the demo project root (the `tfm-demo/` folder), which
# itself sits inside the cloned blueprint repo so we can import its `src/` and
# load the checkpoint under `models/`.
# In a Cloudera notebook/interactive session the module may be exec'd as a cell,
# where `__file__` is undefined — fall back to the working directory (the project
# root, by CML convention) in that case.
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_ROOT = Path.cwd()
REPO_ROOT = PROJECT_ROOT.parent                       # the cloned blueprint repo
ARTIFACTS = PROJECT_ROOT / "demo_artifacts"
MODEL_DIR = REPO_ROOT / "models" / "decoder-foundation-model"

# ---- model dims ------------------------------------------------------------
MAX_LENGTH = 128
MERCHANT_HASH_SIZE = 2000
PCA_DIM = 64

# Make the blueprint's src/ importable.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
