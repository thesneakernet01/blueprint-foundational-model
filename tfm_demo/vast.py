# SPDX-License-Identifier: Apache-2.0
"""VAST S3 data layer — the temporal splits live as Parquet objects on VAST,
written and read directly with boto3 (no Impala/s3a in the path).

Why this exists: the CDW Impala warehouse could not be pointed at VAST — its
Java s3a client gets a bare "400 null" from the endpoint on every table-storage
call, while the exact same S3 API sequence succeeds via boto3 (proven by
scripts/vast_probe.py). So this backend skips the warehouse entirely.

Layout: one object per split at  s3://<bucket>/<prefix>/<table>.parquet  with
the table names shared with the Impala backend (train / val_eval / test_eval).
Columns keep their original TabFormer names (Parquet is fine with spaces and
'?'), and Parquet preserves row order — so unlike Impala no row_id column or
snake_case mapping is needed; a frame reads back exactly as written, which is
what the row-position-keyed embedding cache requires.

The row count is stamped on each object as S3 user metadata (x-amz-meta-rows)
so check() can report table sizes from a HeadObject instead of downloading.

VAST specifics (same as the probe): path-style addressing, sigv4, region
literal "vast" ($VAST_REGION overrides), and an internal-CA TLS cert — so
certificate verification is OFF unless $VAST_VERIFY_SSL=1 (or it names a CA
bundle file).
"""

from __future__ import annotations

import io
import math
import os
import threading
import time
from typing import Callable, Dict, Optional, Sequence

from .config import log
from .settings import get_vast_settings

# split key (as used by export.py) -> object stem; names match impala.SPLIT_TABLES
SPLIT_TABLES = {"train": "train", "val": "val_eval", "test": "test_eval"}

# Parquet row-group size: the unit of the bounded-memory chunked read in
# read_split_cudf (one row group of pandas/arrow on the host at a time).
ROW_GROUP_ROWS = int(os.environ.get("VAST_ROW_GROUP_ROWS", "250000"))
_ROWS_META = "rows"

# Upload transfer tuning. previewhub is a high-latency WAN endpoint, so
# throughput comes from parts in flight, not from a single stream: files above
# the part size are split into part-size chunks PUT concurrently. Anything at
# or below one part goes as a single PUT — multipart's initiate/complete
# round-trips are pure overhead there (boto3's 8 MB default turned an 11 MB
# upload into a 2-part multipart: all the overhead, no parallelism).
# Part size is clamped to S3's 5 MiB minimum for non-final parts; smaller
# parts survive flaky/slow paths better (each finishes before proxy send
# timeouts — the live failure was an SSLEOFError mid-part), bigger parts have
# less per-request overhead.
UPLOAD_PART_MB = max(5, int(os.environ.get("VAST_UPLOAD_PART_MB", "8")))
UPLOAD_CONCURRENCY = int(os.environ.get("VAST_UPLOAD_CONCURRENCY", "8"))
# Whole-upload retries: the endpoint has been seen closing TLS mid-part
# (SSLEOFError) after botocore's own per-request retries are spent, so each
# split upload restarts from the buffered Parquet bytes with backoff.
UPLOAD_RETRIES = int(os.environ.get("VAST_UPLOAD_RETRIES", "3"))

Progress = Optional[Callable[[str], None]]


# --------------------------------------------------------------------------- #
# connection
# --------------------------------------------------------------------------- #
def _verify():
    v = os.environ.get("VAST_VERIFY_SSL", "")
    if v in ("", "0", "false", "no"):
        return False
    return True if v in ("1", "true", "yes") else v    # else: a CA bundle path


def _client():
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:
        raise RuntimeError(
            "boto3 is not installed — run the 'Install dependencies' job "
            "(pip install boto3) to use the VAST storage backend."
        ) from exc

    s = get_vast_settings()
    missing = [k for k in ("endpoint", "bucket", "access_key", "secret_key") if not s[k]]
    if missing:
        raise RuntimeError(
            "VAST storage is not fully configured (missing: "
            f"{', '.join(missing)}) — open the Data dialog in the UI (or set "
            "$VAST_ENDPOINT / $VAST_BUCKET / $VAST_ACCESS_KEY / $VAST_SECRET_KEY)."
        )
    if not _verify():
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    client = boto3.client(
        "s3",
        endpoint_url=s["endpoint"],
        aws_access_key_id=s["access_key"],
        aws_secret_access_key=s["secret_key"],
        region_name=os.environ.get("VAST_REGION", "vast"),
        config=Config(
            s3={"addressing_style": "path"},           # required by VAST
            signature_version="s3v4",
            retries={"max_attempts": 5, "mode": "standard"},
            # The pool must fit every concurrent part upload plus the parallel
            # per-split writers (storage.write_splits), or urllib3 serialises
            # them again behind pool checkouts.
            max_pool_connections=max(10, UPLOAD_CONCURRENCY + len(SPLIT_TABLES)),
            # The path to the endpoint is slow and proxied: keep idle-looking
            # long transfers alive and don't declare a stall until well past a
            # slow part's real duration.
            tcp_keepalive=True,
            connect_timeout=30,
            read_timeout=300,
        ),
        verify=_verify(),
    )
    return client, s["bucket"]


def _key(table: str) -> str:
    prefix = get_vast_settings()["prefix"]
    return f"{prefix}/{table}.parquet" if prefix else f"{table}.parquet"


def target() -> str:
    """Human-readable description of where the splits land."""
    s = get_vast_settings()
    path = f"{s['bucket']}/{s['prefix']}" if s["prefix"] else s["bucket"]
    return f"s3://{path} @ {s['endpoint'] or '<no endpoint>'}"


# --------------------------------------------------------------------------- #
# write (prepare_data)
# --------------------------------------------------------------------------- #
def write_split(df, split: str, progress: Progress = None) -> int:
    """Serialise a pandas frame to Parquet and upload it as the split's object.
    Returns the row count written."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    emit = progress or (lambda m: log.info("[vast] %s", m))
    table = SPLIT_TABLES[split]
    client, bucket = _client()
    key = _key(table)

    emit(f"  {table}: encoding {len(df):,} rows as Parquet ...")
    buf = io.BytesIO()
    pq.write_table(
        pa.Table.from_pandas(df, preserve_index=False),
        buf, row_group_size=ROW_GROUP_ROWS, compression="snappy",
    )
    size = buf.tell()
    buf.seek(0)

    from boto3.s3.transfer import TransferConfig

    part = UPLOAD_PART_MB << 20
    parts = max(1, math.ceil(size / part))
    emit(f"  {table}: uploading {size / 1e6:.1f} MB to s3://{bucket}/{key} "
         f"({parts} part(s) × {UPLOAD_PART_MB} MB, "
         f"concurrency {min(parts, UPLOAD_CONCURRENCY)}) ...")
    sent = [0]
    lock = threading.Lock()                           # parts complete on worker threads

    def _cb(n: int) -> None:                          # upload progress, ~every 32 MB
        with lock:
            before = sent[0]
            sent[0] += n
            done = sent[0]
        if done // (32 << 20) != before // (32 << 20) or done >= size:
            emit(f"  {table}: {done / 1e6:.0f}/{size / 1e6:.0f} MB uploaded")

    for attempt in range(UPLOAD_RETRIES + 1):
        buf.seek(0)
        sent[0] = 0
        try:
            client.upload_fileobj(
                buf, bucket, key,
                ExtraArgs={"Metadata": {_ROWS_META: str(len(df))}},
                Callback=_cb,
                Config=TransferConfig(
                    # threshold > part size: a file of exactly one part stays a
                    # single PUT instead of a one-part multipart.
                    multipart_threshold=part + 1,
                    multipart_chunksize=part,
                    max_concurrency=UPLOAD_CONCURRENCY,
                ),
            )
            break
        except Exception as exc:                                   # noqa: BLE001
            if attempt == UPLOAD_RETRIES:
                raise
            delay = min(2 ** (attempt + 1), 30)
            emit(f"  {table}: upload failed (attempt {attempt + 1}/"
                 f"{UPLOAD_RETRIES + 1}): {_first_line(exc)} — retrying in {delay}s")
            time.sleep(delay)
    emit(f"  {table}: {len(df):,} rows written")
    return len(df)


# --------------------------------------------------------------------------- #
# read (export / training)
# --------------------------------------------------------------------------- #
def read_split_cudf(split: str, columns: Sequence[str]):
    """Fetch a split's Parquet object and return ONE cuDF frame (original
    column names, write order).

    The object is buffered on the host (tens of MB — far below what the ingest
    already holds), then decoded row-group by row-group with each chunk pushed
    to the GPU immediately, so decoded host memory stays one row group deep.
    """
    import cudf
    import pyarrow.parquet as pq

    client, bucket = _client()
    key = _key(SPLIT_TABLES[split])
    try:
        body = client.get_object(Bucket=bucket, Key=key)["Body"]
    except Exception as exc:                                       # noqa: BLE001
        raise RuntimeError(
            f"VAST split object s3://{bucket}/{key} is missing or unreadable "
            f"({_first_line(exc)}) — run the data load from the UI's Data "
            "dialog (or scripts/prepare_data.py) first."
        ) from exc
    buf = io.BytesIO(body.read())

    pf = pq.ParquetFile(buf)
    cols = list(columns)
    parts = []
    for g in range(pf.num_row_groups):
        tbl = pf.read_row_group(g, columns=cols).select(cols)
        parts.append(cudf.DataFrame.from_arrow(tbl))
        del tbl
    if not parts:
        raise RuntimeError(
            f"VAST split object s3://{bucket}/{key} is empty — re-run the "
            "data load."
        )
    return parts[0] if len(parts) == 1 else cudf.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------- #
# status / preflight
# --------------------------------------------------------------------------- #
def _first_line(exc: Exception) -> str:
    return str(exc).split("\n", 1)[0][:300]


def check() -> Dict:
    """Probe the bucket and report per-split row counts (from the object
    metadata stamped at write time). Never raises — errors land in 'error'."""
    settings = get_vast_settings()
    out: Dict = {
        "ok": False,
        "error": None,
        "endpoint": settings["endpoint"],
        "bucket": settings["bucket"],
        "prefix": settings["prefix"],
        "tables": {t: None for t in SPLIT_TABLES.values()},
    }
    try:
        client, bucket = _client()
        client.head_bucket(Bucket=bucket)
        for table in SPLIT_TABLES.values():
            try:
                head = client.head_object(Bucket=bucket, Key=_key(table))
            except Exception:                                      # noqa: BLE001
                continue                                           # missing split
            rows = (head.get("Metadata") or {}).get(_ROWS_META)
            out["tables"][table] = int(rows) if rows and rows.isdigit() else 0
        out["ok"] = True
    except Exception as exc:                                       # noqa: BLE001
        out["error"] = _first_line(exc)
    return out


def splits_ready() -> tuple[bool, str]:
    """(ready, detail) — ready when all three split objects exist with rows."""
    c = check()
    if not c["ok"]:
        return False, c["error"] or "VAST check failed"
    empty = [t for t, n in c["tables"].items() if not n]
    if empty:
        return False, (f"split object(s) {', '.join(empty)} missing or empty "
                       f"under {target()}")
    return True, ", ".join(f"{t}={n:,}" for t, n in c["tables"].items())
