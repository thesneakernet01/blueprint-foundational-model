# SPDX-License-Identifier: Apache-2.0
"""VAST S3 data layer — the temporal splits live as Parquet objects on the
"VAST" S3 target, written and read directly with boto3 (no Impala/s3a in the
path). The name is historical: the store behind it is now a MinIO bucket, and
this module is tuned for that — the original VAST Data endpoint's defects
(60 KiB upload-body cap, dropped multipart metadata, no trailing-checksum
support) and their workarounds (tiny-part uploads, `.rows` sidecars, stripped
checksums/Expect headers) are gone.

Layout: one object per split at  s3://<bucket>/<prefix>/<table>.parquet  with
the table names shared with the Impala backend (train / val_eval / test_eval).
Columns keep their original TabFormer names (Parquet is fine with spaces and
'?'), and Parquet preserves row order — so unlike Impala no row_id column or
snake_case mapping is needed; a frame reads back exactly as written, which is
what the row-position-keyed embedding cache requires.

The row count is stamped on each object as S3 user metadata (x-amz-meta-rows)
so check() can report table sizes from a HeadObject instead of downloading.
MinIO preserves user metadata on multipart uploads, so this works for every
object size.

MinIO specifics: path-style addressing (MinIO serves buckets on the path, not
virtual hosts), sigv4, region defaults to MinIO's "us-east-1" ($VAST_REGION
overrides). TLS verification is OFF unless $VAST_VERIFY_SSL=1 (or it names a
CA bundle file) — self-hosted MinIO frequently runs on an internal CA.
Trailing checksums (botocore >= 1.36's aws-chunked CRC32 default) stay ON for
end-to-end integrity — MinIO supports them; $VAST_TRAILING_CHECKSUMS=0 turns
them off for stores that don't.
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

# Transfer tuning, sized for MinIO on a fast link: big parts keep per-request
# overhead low, and a single stream usually saturates a LAN — the concurrency
# is there for high-latency paths and for stacking the three split uploads.
# Anything at or below one part goes as a single PUT: multipart's
# initiate/complete round-trips are pure overhead there. Part size is clamped
# to S3's 5 MiB minimum for non-final parts.
UPLOAD_PART_MB = max(5, int(os.environ.get("VAST_UPLOAD_PART_MB", "64")))
UPLOAD_CONCURRENCY = int(os.environ.get("VAST_UPLOAD_CONCURRENCY", "8"))
# Whole-upload retries on top of botocore's per-request retries: cheap
# insurance, the Parquet bytes are already buffered.
UPLOAD_RETRIES = int(os.environ.get("VAST_UPLOAD_RETRIES", "2"))

Progress = Optional[Callable[[str], None]]


# --------------------------------------------------------------------------- #
# connection
# --------------------------------------------------------------------------- #
def _verify():
    v = os.environ.get("VAST_VERIFY_SSL", "")
    if v in ("", "0", "false", "no"):
        return False
    return True if v in ("1", "true", "yes") else v    # else: a CA bundle path


def _env_off(name: str) -> bool:
    return os.environ.get(name, "").lower() in ("0", "false", "no")


def _client(read_timeout: int = 120, attempts: int = 5):
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
    opts = dict(
        s3={"addressing_style": "path"},               # MinIO wants path-style
        signature_version="s3v4",
        retries={"max_attempts": attempts, "mode": "standard"},
        # The pool must fit every concurrent part transfer plus the parallel
        # per-split writers (storage.write_splits), or urllib3 serialises
        # them again behind pool checkouts.
        max_pool_connections=max(10, UPLOAD_CONCURRENCY + len(SPLIT_TABLES)),
        tcp_keepalive=True,
        connect_timeout=10,
        read_timeout=read_timeout,
    )
    if _env_off("VAST_TRAILING_CHECKSUMS"):
        # Escape hatch for stores that can't parse aws-chunked trailing
        # checksums (the original VAST endpoint couldn't): plain signed bodies.
        opts["request_checksum_calculation"] = "when_required"
        opts["response_checksum_validation"] = "when_required"
    try:
        config = Config(**opts)
    except TypeError:
        # botocore < 1.36 doesn't know the checksum options — and doesn't
        # need them (it never sends trailing checksums).
        opts.pop("request_checksum_calculation", None)
        opts.pop("response_checksum_validation", None)
        config = Config(**opts)
    client = boto3.client(
        "s3",
        endpoint_url=s["endpoint"],
        aws_access_key_id=s["access_key"],
        aws_secret_access_key=s["secret_key"],
        region_name=os.environ.get("VAST_REGION", "us-east-1"),
        config=config,
        verify=_verify(),
    )

    # $VAST_EXPECT_100=0 strips botocore's `Expect: 100-continue` from upload
    # requests — only needed behind gateways that mishandle the interim
    # response (MinIO itself is fine with it).
    if _env_off("VAST_EXPECT_100"):
        for op in ("PutObject", "UploadPart"):
            client.meta.events.register(f"request-created.s3.{op}", _strip_expect)

    # One-time wire-format banner so the job log shows which client behavior
    # is deployed.
    if not _WIRE_LOGGED.is_set():
        _WIRE_LOGGED.set()
        import botocore
        log.info(
            "[vast] wire config: boto3 %s · botocore %s · trailing checksums: "
            "%s · part %d MB · concurrency %d",
            getattr(boto3, "__version__", "?"),
            getattr(botocore, "__version__", "?"),
            "off" if _env_off("VAST_TRAILING_CHECKSUMS") else "on",
            UPLOAD_PART_MB, UPLOAD_CONCURRENCY,
        )
    return client, s["bucket"]


_WIRE_LOGGED = threading.Event()


def _strip_expect(request, **_kwargs):
    if "Expect" in request.headers:
        del request.headers["Expect"]


def _transfer_config():
    from boto3.s3.transfer import TransferConfig

    part = UPLOAD_PART_MB << 20
    return TransferConfig(
        # threshold > part size: a file of exactly one part stays a single
        # PUT/GET instead of a 1-part multipart.
        multipart_threshold=part + 1,
        multipart_chunksize=part,
        max_concurrency=UPLOAD_CONCURRENCY,
    )


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

    meta = {_ROWS_META: str(len(df))}
    n_parts = max(1, math.ceil(size / (UPLOAD_PART_MB << 20)))
    emit(f"  {table}: uploading {size / 1e6:.1f} MB to s3://{bucket}/{key} "
         f"({n_parts} part(s) × {UPLOAD_PART_MB} MB, "
         f"concurrency {min(n_parts, UPLOAD_CONCURRENCY)}) ...")

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
                ExtraArgs={"Metadata": meta},
                Callback=_cb,
                Config=_transfer_config(),
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

    The object is buffered on the host via concurrent ranged GETs (tens of MB —
    far below what the ingest already holds), then decoded row-group by
    row-group with each chunk pushed to the GPU immediately, so decoded host
    memory stays one row group deep.
    """
    import cudf
    import pyarrow.parquet as pq

    client, bucket = _client()
    key = _key(SPLIT_TABLES[split])
    buf = io.BytesIO()
    try:
        client.download_fileobj(bucket, key, buf, Config=_transfer_config())
    except Exception as exc:                                       # noqa: BLE001
        raise RuntimeError(
            f"VAST split object s3://{bucket}/{key} is missing or unreadable "
            f"({_first_line(exc)}) — run the data load from the UI's Data "
            "dialog (or scripts/prepare_data.py) first."
        ) from exc
    buf.seek(0)

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
        client, bucket = _client(read_timeout=30)
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
