# SPDX-License-Identifier: Apache-2.0
"""Generate the TabFormer temporal-split parquets the export needs.

The dataset is in no git repo: NB01 downloads it (~2.4 GB transactions.tgz from
IBM Box) and writes the temporal splits to data/TabFormer/temporal_split/. NB01
does this with cuDF over the full ~24M rows, which OOMs a single modest GPU — so
we reproduce its logic here in plain, chunked pandas (CPU, bounded memory):

  * download + extract card_transaction.v1.csv (same IBM Box source as NB01),
  * temporal split by date cutoffs at 80% / 90% cumulative rows (NB01's rule),
  * stratified ~100K val_eval / test_eval subsets (NB01's eval workflow),
  * write train.parquet, val_eval.parquet, test_eval.parquet with the raw
    transaction columns (Amount as "$…", Time as "HH:MM", Is Fraud? as Yes/No) —
    exactly what tfm_demo/export.py reads.

The train split is capped ($PREP_TRAIN_CAP, default 1M rows) since the export
balances/subsamples it anyway; this keeps the parquet and the export tractable.
Idempotent; honors config.DATA_DIR ($DATA_DIR). No GPU required.

Run:  python scripts/prepare_data.py
"""

from __future__ import annotations

import os
import sys
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path

try:
    _ROOT = Path(__file__).resolve().parent.parent
except NameError:
    _ROOT = Path.cwd()
sys.path.insert(0, str(_ROOT))
from tfm_demo.config import DATA_DIR  # noqa: E402

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

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
EVAL_SAMPLES = int(os.environ.get("PREP_EVAL_SAMPLES", "100000"))
TRAIN_CAP = int(os.environ.get("PREP_TRAIN_CAP", "1000000"))
CHUNK = int(os.environ.get("PREP_CHUNK_ROWS", "2000000"))
SEED = 42

TAB_DIR = DATA_DIR / "TabFormer"
RAW_DIR = TAB_DIR / "raw"
CSV_PATH = RAW_DIR / "card_transaction.v1.csv"
TGZ_PATH = TAB_DIR / "transactions.tgz"
TEMPORAL_DIR = TAB_DIR / "temporal_split"
REQUIRED = ["train.parquet", "val_eval.parquet", "test_eval.parquet"]


def _download() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if CSV_PATH.exists():
        return
    if not TGZ_PATH.exists():
        last = None
        for attempt in range(1, 4):                       # IBM Box can be flaky
            try:
                print(f"prepare_data: downloading transactions.tgz (attempt {attempt}) ...")
                urllib.request.urlretrieve(DOWNLOAD_URL, TGZ_PATH)
                break
            except Exception as exc:                       # noqa: BLE001
                last = exc
                print(f"  download failed: {exc}")
                TGZ_PATH.unlink(missing_ok=True)
        else:
            raise RuntimeError(f"prepare_data: download failed after 3 tries: {last}")
    print("prepare_data: extracting transactions.tgz ...")
    with tarfile.open(TGZ_PATH, "r:gz") as tar:
        tar.extractall(path=RAW_DIR)
    if not CSV_PATH.exists():
        raise RuntimeError(f"prepare_data: {CSV_PATH} not found after extraction")


def _date_int(df):
    return df["Year"].astype(int) * 10000 + df["Month"].astype(int) * 100 + df["Day"].astype(int)


def main() -> None:
    if all((TEMPORAL_DIR / f).exists() for f in REQUIRED):
        print(f"prepare_data: temporal splits already present in {TEMPORAL_DIR}")
        return

    import numpy as np
    import pandas as pd

    _download()

    # ---- pass 1: per-date row counts -> temporal cutoffs ------------------
    print("prepare_data: scanning dates for temporal cutoffs ...")
    counts: Counter = Counter()
    for chunk in pd.read_csv(CSV_PATH, usecols=["Year", "Month", "Day"], chunksize=CHUNK):
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
    keep_p = min(1.0, TRAIN_CAP / max(1, train_total))
    print(f"prepare_data: {total:,} rows · cutoffs {train_cut}/{test_cut} · "
          f"train≈{train_total:,} (keep {keep_p:.3f})")

    # ---- pass 2: split, cap train, collect val/test -----------------------
    rng = np.random.default_rng(SEED)
    train_parts, val_parts, test_parts = [], [], []
    for chunk in pd.read_csv(CSV_PATH, usecols=USECOLS, chunksize=CHUNK):
        di = _date_int(chunk)
        tr = chunk[di < train_cut]
        if keep_p < 1.0 and len(tr):
            tr = tr[rng.random(len(tr)) < keep_p]
        train_parts.append(tr)
        val_parts.append(chunk[(di >= train_cut) & (di < test_cut)])
        test_parts.append(chunk[di >= test_cut])

    train_df = pd.concat(train_parts, ignore_index=True); del train_parts
    val_df = pd.concat(val_parts, ignore_index=True); del val_parts
    test_df = pd.concat(test_parts, ignore_index=True); del test_parts

    def stratified(df, n):
        if n >= len(df):
            return df.reset_index(drop=True)
        is_fraud = df["Is Fraud?"].astype(str).str.lower().eq("yes")
        frac = n / len(df)
        return pd.concat([
            df[is_fraud].sample(frac=frac, random_state=SEED),
            df[~is_fraud].sample(frac=frac, random_state=SEED),
        ]).sample(frac=1, random_state=SEED).reset_index(drop=True)

    val_eval = stratified(val_df, EVAL_SAMPLES)
    test_eval = stratified(test_df, EVAL_SAMPLES)

    TEMPORAL_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(TEMPORAL_DIR / "train.parquet", index=False)
    val_eval.to_parquet(TEMPORAL_DIR / "val_eval.parquet", index=False)
    test_eval.to_parquet(TEMPORAL_DIR / "test_eval.parquet", index=False)

    def rate(df):
        return df["Is Fraud?"].astype(str).str.lower().eq("yes").mean() * 100

    print(f"prepare_data: wrote train={len(train_df):,} ({rate(train_df):.3f}% fraud)  "
          f"val_eval={len(val_eval):,} ({rate(val_eval):.3f}%)  "
          f"test_eval={len(test_eval):,} ({rate(test_eval):.3f}%) -> {TEMPORAL_DIR}")


if __name__ == "__main__":
    main()
