# SPDX-License-Identifier: Apache-2.0
"""Generate the TabFormer temporal splits and load them into storage.

The dataset is in no git repo: NB01 downloads it (~2.4 GB transactions.tgz from
IBM Box). NB01 splits it with cuDF over the full ~24M rows, which OOMs a single
modest GPU — so we reproduce its logic here in plain, chunked pandas (CPU,
bounded memory):

  * download + extract card_transaction.v1.csv (same IBM Box source as NB01),
  * temporal split by date cutoffs at 80% / 90% cumulative rows (NB01's rule),
  * stratified ~100K val_eval / test_eval subsets (NB01's eval workflow),
  * load the splits train / val_eval / test_eval into the configured storage
    backend — Impala tables or Parquet objects on VAST S3 (tfm_demo/storage.py)
    — with the raw transaction columns (Amount as "$…", Time as "HH:MM",
    Is Fraud? as Yes/No) — exactly what tfm_demo/export.py reads back.

The storage target comes from the UI's Data dialog (stored via
tfm_demo/settings.py; $IMPALA_* / $VAST_* env vars seed the defaults). When
it's unconfigured this script prints guidance and exits 0, so the AMP setup
job doesn't fail a fresh deployment where the UI hasn't run yet.

Disk-frugal: the big tgz/CSV go to a scratch dir ($PREP_SCRATCH, default the
system temp), not the project volume, and the scratch is removed at the end.
Train is capped ($PREP_TRAIN_CAP, default 1M). Idempotent — skips when the
tables are already populated — unless $PREP_FORCE=1 (the UI's "Load data"
button forces, so a click always re-ingests). No GPU required.

Run:  python scripts/prepare_data.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from collections import Counter
from pathlib import Path

try:
    _ROOT = Path(__file__).resolve().parent.parent
except NameError:
    _ROOT = Path.cwd()
sys.path.insert(0, str(_ROOT))
from tfm_demo.config import DATA_DIR  # noqa: E402
from tfm_demo import storage  # noqa: E402

# Same shared file NB01 pulls from IBM Box.
DOWNLOAD_URL = (
    "https://ibm.ent.box.com/index.php"
    "?rm=box_download_shared_file"
    "&shared_name=mhrtz6xiknblqznoi9h4f390scoqustt"
    "&file_id=f_770766751708"
)

# Raw transaction columns the export consumes (drop "Errors?").
USECOLS = ["User", "Card", "Year", "Month", "Day", "Time", "Amount", "Use Chip",
           "Merchant Name", "Merchant City", "Merchant State", "Zip", "MCC", "Is Fraud?"]
DATE_COLS = ["Year", "Month", "Day"]

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
EVAL_SAMPLES = int(os.environ.get("PREP_EVAL_SAMPLES", "100000"))
TRAIN_CAP = int(os.environ.get("PREP_TRAIN_CAP", "1000000"))
CHUNK = int(os.environ.get("PREP_CHUNK_ROWS", "2000000"))
SEED = 42

SCRATCH = Path(os.environ.get("PREP_SCRATCH") or tempfile.gettempdir()) / "tfm_tabformer"
TGZ_PATH = SCRATCH / "transactions.tgz"
CSV_PATH = SCRATCH / "card_transaction.v1.csv"
FORCE = os.environ.get("PREP_FORCE", "") == "1"


def _download() -> None:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    if not CSV_PATH.exists():
        if not TGZ_PATH.exists():
            last = None
            for attempt in range(1, 4):                   # IBM Box can be flaky
                try:
                    print(f"prepare_data: downloading transactions.tgz (attempt {attempt}) ...")
                    urllib.request.urlretrieve(DOWNLOAD_URL, TGZ_PATH)
                    break
                except Exception as exc:                   # noqa: BLE001
                    last = exc
                    print(f"  download failed: {exc}")
                    TGZ_PATH.unlink(missing_ok=True)
            else:
                raise RuntimeError(f"prepare_data: download failed after 3 tries: {last}")
        # IBM Box sometimes serves an HTML error page instead of the gzip.
        with open(TGZ_PATH, "rb") as fh:
            if fh.read(2) != b"\x1f\x8b":
                TGZ_PATH.unlink(missing_ok=True)
                raise RuntimeError(
                    "prepare_data: downloaded file is not a gzip (IBM Box likely "
                    "returned an error page). Retry, or download transactions.tgz "
                    f"manually into {SCRATCH}."
                )
        print("prepare_data: extracting transactions.tgz ...")
        with tarfile.open(TGZ_PATH, "r:gz") as tar:
            tar.extractall(path=SCRATCH)
        TGZ_PATH.unlink(missing_ok=True)                  # free ~2.4 GB immediately
    if not CSV_PATH.exists():
        # Some archives nest the csv; find it.
        found = next(SCRATCH.rglob("card_transaction.v1.csv"), None)
        if found is None:
            raise RuntimeError(f"prepare_data: card_transaction.v1.csv not found under {SCRATCH}")
        found.replace(CSV_PATH)


def _actual_cols(pd, want):
    """Map desired (stripped) column names to the CSV's actual header names."""
    header = pd.read_csv(CSV_PATH, nrows=0)
    by_stripped = {c.strip(): c for c in header.columns}
    missing = [w for w in want if w not in by_stripped]
    if missing:
        raise RuntimeError(f"prepare_data: CSV missing columns {missing}; "
                           f"header is {list(header.columns)}")
    return [by_stripped[w] for w in want]


def _read_chunks(pd, actual, want):
    for chunk in pd.read_csv(CSV_PATH, usecols=actual, chunksize=CHUNK):
        chunk.columns = [c.strip() for c in chunk.columns]
        yield chunk[want]


def _date_int(df):
    return df["Year"].astype(int) * 10000 + df["Month"].astype(int) * 100 + df["Day"].astype(int)


def _diag(tag: str) -> None:
    """Print peak RSS + free disk so the job log reveals resource pressure."""
    bits = [f"prepare_data[{tag}]"]
    try:
        import resource  # ru_maxrss is KiB on Linux
        bits.append(f"rss={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f}GB")
    except Exception:  # noqa: BLE001
        pass
    for name, p in (("scratch", SCRATCH), ("data", DATA_DIR)):
        try:
            target = p if p.exists() else p.parent
            bits.append(f"{name}_free={shutil.disk_usage(target).free / 1e9:.1f}GB")
        except Exception:  # noqa: BLE001
            pass
    print("  " + " · ".join(bits), flush=True)


def main() -> None:
    if not storage.configured():
        print("prepare_data: no storage target configured yet — set it in the "
              "UI's Data dialog (or $IMPALA_CONNECTION_NAME/$IMPALA_DATABASE, "
              "or $VAST_ENDPOINT/$VAST_BUCKET/$VAST_ACCESS_KEY/$VAST_SECRET_KEY"
              ") and re-run. Nothing to do.")
        return
    target = storage.target()
    if not FORCE:
        ready, detail = storage.splits_ready()
        if ready:
            print(f"prepare_data: splits already loaded ({target}: {detail}) "
                  "— set PREP_FORCE=1 to re-ingest.")
            return

    import numpy as np
    import pandas as pd

    _download()
    date_actual = _actual_cols(pd, DATE_COLS)
    use_actual = _actual_cols(pd, USECOLS)

    _diag("after download")

    # ---- pass 1: per-date row counts -> temporal cutoffs ------------------
    print("prepare_data: scanning dates for temporal cutoffs ...", flush=True)
    counts: Counter = Counter()
    for chunk in _read_chunks(pd, date_actual, DATE_COLS):
        for d, c in _date_int(chunk).value_counts().items():
            counts[int(d)] += int(c)
    total = sum(counts.values())
    train_cut = test_cut = None
    cum = 0
    for d in sorted(counts):
        cum += counts[d]
        if train_cut is None and cum >= TRAIN_RATIO * total:
            train_cut = d
        if cum >= (TRAIN_RATIO + VAL_RATIO) * total:
            test_cut = d
            break
    train_total = sum(c for d, c in counts.items() if d < train_cut)
    val_total = sum(c for d, c in counts.items() if train_cut <= d < test_cut)
    test_total = sum(c for d, c in counts.items() if d >= test_cut)
    # train: keep ALL fraud + subsample normals to the cap; val/test: random
    # subsample to ~EVAL_SAMPLES (preserves the natural fraud rate for eval).
    keep_norm = min(1.0, TRAIN_CAP / max(1, train_total))
    keep_val = min(1.0, EVAL_SAMPLES / max(1, val_total))
    keep_test = min(1.0, EVAL_SAMPLES / max(1, test_total))
    print(f"prepare_data: {total:,} rows · cutoffs {train_cut}/{test_cut} · "
          f"keep norm={keep_norm:.3f} val={keep_val:.3f} test={keep_test:.3f}", flush=True)

    # ---- pass 2: split + subsample in-stream (bounded memory) -------------
    rng = np.random.default_rng(SEED)
    train_parts, val_parts, test_parts = [], [], []

    def _is_fraud(df):
        return df["Is Fraud?"].astype(str).str.lower().eq("yes")

    def _sub(df, p):
        return df if p >= 1.0 or not len(df) else df[rng.random(len(df)) < p]

    for chunk in _read_chunks(pd, use_actual, USECOLS):
        di = _date_int(chunk)
        tr = chunk[di < train_cut]
        if len(tr):
            f = _is_fraud(tr)
            train_parts.append(pd.concat([tr[f], _sub(tr[~f], keep_norm)]))
        val_parts.append(_sub(chunk[(di >= train_cut) & (di < test_cut)], keep_val))
        test_parts.append(_sub(chunk[di >= test_cut], keep_test))

    train_df = pd.concat(train_parts, ignore_index=True); del train_parts
    val_eval = pd.concat(val_parts, ignore_index=True); del val_parts
    test_eval = pd.concat(test_parts, ignore_index=True); del test_parts
    _diag("after split")

    def emit(msg: str) -> None:
        print(msg, flush=True)

    print(f"prepare_data: loading splits into {target} ...", flush=True)
    # write_splits overlaps the three uploads on the VAST backend (each split
    # is an independent object PUT); Impala loads stay sequential inside it.
    storage.write_splits(
        {"train": train_df, "val": val_eval, "test": test_eval}, progress=emit)

    # The export's embedding cache is row-position-keyed against these tables —
    # a re-ingest invalidates it, so drop this target's cache dir.
    shutil.rmtree(DATA_DIR / "embeddings" / storage.cache_key(), ignore_errors=True)

    def rate(df):
        return df["Is Fraud?"].astype(str).str.lower().eq("yes").mean() * 100

    print(f"prepare_data: wrote train={len(train_df):,} ({rate(train_df):.3f}% fraud)  "
          f"val_eval={len(val_eval):,} ({rate(val_eval):.3f}%)  "
          f"test_eval={len(test_eval):,} ({rate(test_eval):.3f}%) -> {target}", flush=True)

    shutil.rmtree(SCRATCH, ignore_errors=True)            # drop the big CSV/scratch


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception:                                     # surface the real error
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
