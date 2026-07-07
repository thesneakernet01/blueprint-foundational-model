# SPDX-License-Identifier: Apache-2.0
"""Impala data layer — the temporal splits live in Impala tables, not parquet.

Write side (scripts/prepare_data.py): each split becomes a Parquet-backed
Impala table (`<db>.train` / `val_eval` / `test_eval`) loaded with batched
multi-row INSERTs. Read side (tfm_demo/export.py): splits are SELECTed back in
chunks, built into pandas on the host, and concatenated into one cuDF frame on
the GPU.

Two connection paths:
  * CML Data Connection (the normal one) — `cml.data_v1.get_connection(name)`
    with the connection name from the UI settings; the CML runtime injects the
    workload credentials.
  * direct impyla — set $IMPALA_HOST (plus $IMPALA_PORT/_USER/_PASSWORD/_AUTH/
    _SSL/_HTTP) to bypass CML entirely, e.g. for testing outside the cluster.

Row order: Impala SELECTs have NO stable order, but the export pipeline caches
embeddings keyed by row position — so every row carries a `row_id` assigned at
write time and every read is `ORDER BY row_id`. Don't remove either half.

Column names: Impala identifiers can't hold spaces or '?', so the TabFormer
names are mapped to snake_case in the tables and mapped back on read — the rest
of the pipeline never sees the Impala names.
"""

from __future__ import annotations

import math
import os
import time
from contextlib import contextmanager
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .config import log
from .settings import get_impala_settings

# split key (as used by export.py) -> Impala table name
SPLIT_TABLES = {"train": "train", "val": "val_eval", "test": "test_eval"}

# original TabFormer column -> (impala column, impala type). Types mirror the
# dtypes pandas.read_csv produced for the old parquet files, so frames read
# back from Impala look the same to the tokenizer / feature engineering.
_SCHEMA = {
    "User": ("user", "BIGINT"),
    "Card": ("card", "BIGINT"),
    "Year": ("year", "INT"),
    "Month": ("month", "INT"),
    "Day": ("day", "INT"),
    "Time": ("time", "STRING"),           # "HH:MM"
    "Amount": ("amount", "STRING"),       # "$123.45" — parsed by _engineer()
    "Use Chip": ("use_chip", "STRING"),
    "Merchant Name": ("merchant_name", "BIGINT"),
    "Merchant City": ("merchant_city", "STRING"),
    "Merchant State": ("merchant_state", "STRING"),
    "Zip": ("zip", "DOUBLE"),             # float64 with NaN -> NULL
    "MCC": ("mcc", "INT"),
    "Is Fraud?": ("is_fraud", "STRING"),  # "Yes"/"No"
}
_ROW_ID = "row_id"

INSERT_ROWS = int(os.environ.get("IMPALA_INSERT_ROWS", "10000"))
FETCH_ROWS = int(os.environ.get("IMPALA_FETCH_ROWS", "250000"))
RETRIES = int(os.environ.get("IMPALA_RETRIES", "3"))     # per-statement, on write
# Optional explicit storage prefix for the split tables (e.g.
# "s3a://my_bucket/tfm_demo") — each table lands at <prefix>/<table>. Use when
# the warehouse's default external location is broken/misconfigured; leave
# empty to let the metastore place the tables.
TABLE_LOCATION = os.environ.get("IMPALA_TABLE_LOCATION", "").rstrip("/")

Progress = Optional[Callable[[str], None]]


# --------------------------------------------------------------------------- #
# connection
# --------------------------------------------------------------------------- #
def database() -> str:
    """The UI-configured target database; raises with guidance when unset."""
    db = get_impala_settings()["database"]
    if not db:
        raise RuntimeError(
            "No Impala database configured — open the Data dialog in the UI "
            "and set the CML data connection + database (or set "
            "$IMPALA_CONNECTION_NAME / $IMPALA_DATABASE)."
        )
    return db


def _connect():
    """Return a DB-API connection: direct impyla if $IMPALA_HOST is set, else
    the named CML Data Connection."""
    host = os.environ.get("IMPALA_HOST")
    if host:
        from impala.dbapi import connect  # type: ignore
        return connect(
            host=host,
            port=int(os.environ.get("IMPALA_PORT", "21050")),
            user=os.environ.get("IMPALA_USER"),
            password=os.environ.get("IMPALA_PASSWORD"),
            auth_mechanism=os.environ.get("IMPALA_AUTH", "NOSASL"),
            use_ssl=os.environ.get("IMPALA_SSL", "") == "1",
            use_http_transport=os.environ.get("IMPALA_HTTP", "") == "1",
            http_path=os.environ.get("IMPALA_HTTP_PATH", ""),
        )

    name = get_impala_settings()["connection"]
    if not name:
        raise RuntimeError(
            "No CML data connection configured — open the Data dialog in the "
            "UI and set the connection name (or set $IMPALA_CONNECTION_NAME)."
        )
    try:
        import cml.data_v1 as cmldata  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "cml.data_v1 is not available in this environment — CML Data "
            "Connections only work inside a CML runtime. For local testing "
            "set $IMPALA_HOST to connect with impyla directly."
        ) from exc
    return cmldata.get_connection(name).get_base_connection()


@contextmanager
def _cursor():
    conn = _connect()
    try:
        cur = conn.cursor()
        try:
            yield cur
        finally:
            cur.close()
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# SQL helpers
# --------------------------------------------------------------------------- #
def _bt(name: str) -> str:
    return f"`{name}`"


def _qualified(table: str) -> str:
    return f"{_bt(database())}.{_bt(table)}"


def _lit(v) -> str:
    """A Python value as an Impala SQL literal (None/NaN -> NULL)."""
    item = getattr(v, "item", None)                    # numpy scalar -> python
    if callable(item):
        v = v.item()
    if v is None:
        return "NULL"
    if isinstance(v, float):
        return repr(v) if math.isfinite(v) else "NULL"
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, int):
        return str(v)
    s = str(v).replace("\\", "\\\\").replace("'", "\\'")
    return f"'{s}'"


# --------------------------------------------------------------------------- #
# write (prepare_data)
# --------------------------------------------------------------------------- #
class _WriteSession:
    """One connection for a bulk load, with reconnect-and-retry per statement.

    An ingest issues ~100+ statements over one HS2 session across many minutes;
    transient server-side failures (HMS hiccups, warehouse autoscaling, idle
    proxies dropping the socket) must not kill the whole load. Retried INSERTs
    are guarded by a row-count probe so a batch whose rows actually landed
    before the error surfaced is skipped, never double-inserted (duplicate
    row_ids would silently break the ORDER BY row_id read contract)."""

    def __init__(self, emit: Callable[[str], None]) -> None:
        self._emit = emit
        self._conn = None
        self._cur = None

    def _cursor(self):
        if self._cur is None:
            self._conn = _connect()
            self._cur = self._conn.cursor()
        return self._cur

    def reset(self) -> None:
        for c in (self._cur, self._conn):
            try:
                if c is not None:
                    c.close()
            except Exception:                                      # noqa: BLE001
                pass
        self._cur = self._conn = None

    close = reset

    def _landed(self, guard: Tuple[str, int, int]) -> bool:
        """True if the guarded INSERT's rows are already in the table."""
        qualified, before, batch = guard
        cur = self._cursor()
        try:
            cur.execute(f"REFRESH {qualified}")   # fresh coordinator metadata
        except Exception:                                          # noqa: BLE001
            pass
        cur.execute(f"SELECT COUNT(*) FROM {qualified}")
        count = int(cur.fetchone()[0])
        if count == before + batch:
            return True
        if count == before:
            return False
        raise RuntimeError(
            f"{qualified} holds {count:,} rows mid-retry (expected {before:,} "
            f"or {before + batch:,}) — table is inconsistent, aborting; re-run "
            "the data load to rebuild it."
        )

    def execute(self, sql: str, count_guard: Optional[Tuple[str, int, int]] = None,
                retries: Optional[int] = None) -> None:
        retries = RETRIES if retries is None else retries
        for attempt in range(retries + 1):
            try:
                if count_guard and attempt and self._landed(count_guard):
                    self._emit("  (batch already landed on the server — not retrying)")
                    return
                self._cursor().execute(sql)
                return
            except Exception as exc:                               # noqa: BLE001
                self.reset()
                if attempt == retries:
                    raise
                self._emit(f"  Impala error (attempt {attempt + 1}/{retries + 1}), "
                           f"reconnecting: {_first_line(exc)}")
                time.sleep(min(2 ** (attempt + 1), 15))


def _first_line(exc: Exception) -> str:
    """Impala/HMS errors nest multi-KB S3 stack traces — keep logs readable."""
    return str(exc).split("\n", 1)[0][:300]


def write_split(df, split: str, progress: Progress = None) -> int:
    """(Re)create the split's table and load a pandas frame into it. The frame
    must carry exactly the original TabFormer columns (any order). Returns the
    row count written."""
    emit = progress or (lambda m: log.info("[impala] %s", m))
    table = SPLIT_TABLES[split]
    qualified = _qualified(table)
    cols = list(df.columns)
    missing = [c for c in cols if c not in _SCHEMA]
    if missing:
        raise ValueError(f"write_split: no Impala mapping for columns {missing}")

    ddl_cols = ", ".join(
        [f"{_bt(_ROW_ID)} BIGINT"]
        + [f"{_bt(_SCHEMA[c][0])} {_SCHEMA[c][1]}" for c in cols]
    )
    ins_cols = ", ".join([_bt(_ROW_ID)] + [_bt(_SCHEMA[c][0]) for c in cols])

    sess = _WriteSession(emit)
    try:
        try:
            sess.execute(f"CREATE DATABASE IF NOT EXISTS {_bt(database())}")
        except Exception as exc:                                   # noqa: BLE001
            # The database usually pre-exists; lacking CREATE rights is fine.
            log.info("[impala] CREATE DATABASE skipped: %s", exc)
        # EXTERNAL + external.table.purge: DROP still deletes the data, but
        # INSERTs bypass the Hive-ACID transaction path — CDP warehouses default
        # plain CREATE TABLE to insert-only transactional tables, whose per-
        # INSERT metastore transactions both throttle the load and die with
        # "TransactionException ... Broken pipe" on flaky HMS links.
        #
        # BUT external tables land in the warehouse's *external* storage
        # location, which some environments misconfigure (seen live: the
        # database's external dir pointed at a nonexistent S3 bucket and every
        # createTable RPC 404'd) — so when the external CREATE is rejected,
        # fall back to a managed table (its storage demonstrably works there)
        # and let the per-statement retry + count guard carry the ACID inserts.
        sess.execute(f"DROP TABLE IF EXISTS {qualified}")
        loc = f" LOCATION '{TABLE_LOCATION}/{table}'" if TABLE_LOCATION else ""
        try:
            sess.execute(
                f"CREATE EXTERNAL TABLE IF NOT EXISTS {qualified} ({ddl_cols}) "
                f"STORED AS PARQUET{loc} "
                "TBLPROPERTIES ('external.table.purge'='true')",
                retries=0,   # storage misconfig is deterministic — fail fast
            )
        except Exception as exc:                                   # noqa: BLE001
            emit(f"  external table rejected ({_first_line(exc)})")
            emit(f"  falling back to a managed table for {table} "
                 "(transactional inserts, slower)")
            sess.execute(
                f"CREATE TABLE IF NOT EXISTS {qualified} ({ddl_cols}) "
                "STORED AS PARQUET"
            )

        total = len(df)
        buf: List[str] = []
        written = 0

        def flush() -> None:
            nonlocal written
            if not buf:
                return
            sess.execute(
                f"INSERT INTO {qualified} ({ins_cols}) VALUES " + ",".join(buf),
                count_guard=(qualified, written, len(buf)),
            )
            written += len(buf)
            buf.clear()
            emit(f"  {table}: {written:,}/{total:,} rows written")

        for i, row in enumerate(df.itertuples(index=False, name=None)):
            buf.append("(" + ",".join([str(i)] + [_lit(v) for v in row]) + ")")
            if len(buf) >= INSERT_ROWS:
                flush()
        flush()
    finally:
        sess.close()
    return written


# --------------------------------------------------------------------------- #
# read (export / training)
# --------------------------------------------------------------------------- #
def read_split_cudf(split: str, columns: Sequence[str]):
    """SELECT a split (original column names, write order) into ONE cuDF frame.

    Rows come over in FETCH_ROWS chunks — each chunk becomes a small pandas
    frame that is pushed to the GPU immediately, so host memory holds one chunk,
    not the whole split."""
    import cudf
    import pandas as pd

    impala_cols = ", ".join(_bt(_SCHEMA[c][0]) for c in columns)
    table = SPLIT_TABLES[split]
    sql = (f"SELECT {impala_cols} FROM {_qualified(table)} "
           f"ORDER BY {_bt(_ROW_ID)}")

    parts = []
    with _cursor() as cur:
        cur.execute(sql)
        while True:
            rows = cur.fetchmany(FETCH_ROWS)
            if not rows:
                break
            pdf = pd.DataFrame(rows, columns=list(columns))
            parts.append(cudf.from_pandas(pdf))
            del pdf
    if not parts:
        raise RuntimeError(
            f"Impala table {database()}.{table} is empty — run the data load "
            "from the UI's Data dialog (or scripts/prepare_data.py) first."
        )
    return parts[0] if len(parts) == 1 else cudf.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------- #
# status / preflight
# --------------------------------------------------------------------------- #
def check() -> Dict:
    """Connect and report per-split row counts. Never raises — errors land in
    the 'error' field so the UI can render them."""
    settings = get_impala_settings()
    out: Dict = {
        "ok": False,
        "error": None,
        "connection": settings["connection"],
        "database": settings["database"],
        "tables": {t: None for t in SPLIT_TABLES.values()},
    }
    try:
        db = database()
        with _cursor() as cur:
            cur.execute(f"SHOW TABLES IN {_bt(db)}")
            existing = {r[0].lower() for r in cur.fetchall()}
            for table in SPLIT_TABLES.values():
                if table in existing:
                    cur.execute(f"SELECT COUNT(*) FROM {_qualified(table)}")
                    out["tables"][table] = int(cur.fetchone()[0])
        out["ok"] = True
    except Exception as exc:                                       # noqa: BLE001
        out["error"] = str(exc)
    return out


def splits_ready() -> tuple[bool, str]:
    """(ready, detail) — ready when all three split tables exist with rows."""
    c = check()
    if not c["ok"]:
        return False, c["error"] or "Impala check failed"
    empty = [t for t, n in c["tables"].items() if not n]
    if empty:
        return False, (f"table(s) {', '.join(empty)} missing or empty in "
                       f"database '{c['database']}'")
    return True, ", ".join(f"{t}={n:,}" for t, n in c["tables"].items())
