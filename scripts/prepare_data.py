# SPDX-License-Identifier: Apache-2.0
"""Generate the TabFormer temporal-split parquets the export needs.

The dataset is NOT in the blueprint git repo — notebook 01 downloads it
(~2.4 GB transactions.tgz from IBM Box) and writes the temporal splits to
`data/TabFormer/temporal_split/` (train.parquet, val_eval.parquet,
test_eval.parquet). This script reproduces that by fetching NB01 from the
blueprint and running it headlessly with nbclient, trimmed to the data-prep
cells (it stops before the notebook's XGBoost baseline / EDA tail, which the
demo doesn't need).

NB01 uses `PROJECT_ROOT = Path(".").resolve()`, so we execute it with the
project root as the working directory and the data lands where config.DATA_DIR
points. Idempotent: skips when the splits already exist. GPU + outbound HTTPS to
IBM Box are required; the download can be flaky (the notebook says to retry).

Run:  python scripts/prepare_data.py
"""

from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

try:
    _ROOT = Path(__file__).resolve().parent.parent
except NameError:
    _ROOT = Path.cwd()
sys.path.insert(0, str(_ROOT))
from tfm_demo.config import DATA_DIR, PROJECT_ROOT  # noqa: E402

OWNER = "NVIDIA-AI-Blueprints"
REPO = "transaction-foundation-model"
REF = os.environ.get("TFM_BLUEPRINT_REF", "main")
NB01_URL = f"https://raw.githubusercontent.com/{OWNER}/{REPO}/{REF}/01_dataset_baseline.ipynb"

TEMPORAL_DIR = DATA_DIR / "TabFormer" / "temporal_split"
REQUIRED = ["train.parquet", "val_eval.parquet", "test_eval.parquet"]
# Last data-prep cell in NB01 prints this; we drop everything after it.
STOP_MARKER = "Saved val_eval/test_eval"


def _already_present() -> bool:
    return all((TEMPORAL_DIR / f).exists() for f in REQUIRED)


def _trim(nb):
    """Keep cells up to and including the one that writes the eval subsets."""
    kept = []
    for cell in nb.cells:
        kept.append(cell)
        if cell.cell_type == "code" and STOP_MARKER in "".join(cell.source):
            break
    else:
        print("prepare_data: WARNING — stop marker not found; running full NB01")
        return nb
    nb.cells = kept
    return nb


def _execute(nb) -> None:
    """Run the notebook to completion.

    CML runs this Job inside the engine's ipykernel, which already has a running
    asyncio loop; NotebookClient.execute() calls run_until_complete() and would
    raise "This event loop is already running". Run it in a worker thread, which
    starts with no running loop, and re-raise any error on the main thread.
    """
    import threading
    from nbclient import NotebookClient

    holder: dict = {}

    def _run() -> None:
        try:
            NotebookClient(
                nb,
                timeout=int(os.environ.get("PREP_CELL_TIMEOUT", "5400")),
                kernel_name="python3",
                resources={"metadata": {"path": str(PROJECT_ROOT)}},
            ).execute()
        except BaseException as exc:                       # noqa: BLE001
            holder["err"] = exc

    t = threading.Thread(target=_run)
    t.start()
    t.join()
    if "err" in holder:
        raise holder["err"]


def main() -> None:
    if _already_present():
        print(f"prepare_data: temporal splits already present in {TEMPORAL_DIR}")
        return

    import nbformat

    print(f"prepare_data: fetching NB01 from {NB01_URL}")
    nb = nbformat.reads(urllib.request.urlopen(NB01_URL, timeout=60).read().decode(),
                        as_version=4)
    nb = _trim(nb)

    # NB01 resolves paths from the working directory; run it at the project root
    # so data/ lands under config.DATA_DIR. Force a non-interactive mpl backend.
    os.environ.setdefault("MPLBACKEND", "Agg")
    print(f"prepare_data: running {len(nb.cells)} cells "
          f"(downloads ~2.4 GB from IBM Box — this takes a while) ...")
    _execute(nb)

    missing = [f for f in REQUIRED if not (TEMPORAL_DIR / f).exists()]
    if missing:
        raise RuntimeError(
            f"prepare_data: NB01 finished but {missing} are missing in {TEMPORAL_DIR}"
        )
    print(f"prepare_data: temporal splits ready in {TEMPORAL_DIR}")


if __name__ == "__main__":
    main()
