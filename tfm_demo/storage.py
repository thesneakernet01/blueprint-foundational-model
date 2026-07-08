# SPDX-License-Identifier: Apache-2.0
"""Storage-backend dispatch for the training splits.

The Data dialog picks WHERE the temporal splits live:
  * "impala" — tables in an Impala database (tfm_demo/impala.py)
  * "vast"   — Parquet objects written straight to VAST S3 with boto3
               (tfm_demo/vast.py), bypassing the warehouse's broken s3a path

Writers (scripts/prepare_data.py) and readers (tfm_demo/export.py) go through
this module only, so both backends stay interchangeable. Backend modules are
imported lazily — the impala path must not require boto3 and vice versa.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, Optional, Sequence

from .settings import (
    get_data_settings, get_impala_settings, impala_configured, vast_configured,
)

Progress = Optional[Callable[[str], None]]

# Table/object names are identical across backends.
SPLIT_TABLES = {"train": "train", "val": "val_eval", "test": "test_eval"}


def backend() -> str:
    return get_data_settings()["backend"]


def _mod():
    if backend() == "vast":
        from . import vast
        return vast
    from . import impala
    return impala


def configured() -> bool:
    return vast_configured() if backend() == "vast" else impala_configured()


def target() -> str:
    """Human-readable description of the active data target, for logs/UI."""
    if backend() == "vast":
        from . import vast
        return f"VAST {vast.target()}"
    s = get_impala_settings()
    return (f"Impala database '{s['database'] or '<unset>'}' "
            f"(connection '{s['connection'] or '<unset>'}')")


def cache_key() -> str:
    """Filesystem-safe identity of the data target, keying the row-position-
    bound embedding cache (data/embeddings/<key>/). Distinct targets must never
    share a key — cached sel/labels/embeddings only match the exact tables or
    objects they were built from."""
    if backend() == "vast":
        s = get_data_settings()["vast"]
        raw = f"vast_{s['bucket']}_{s['prefix']}"
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_")
    return get_impala_settings()["database"]


def write_split(df, split: str, progress: Progress = None) -> int:
    return _mod().write_split(df, split, progress=progress)


def write_splits(frames: Dict, progress: Progress = None) -> Dict[str, int]:
    """Write several splits ({split_key: frame}); returns {split_key: rows}.

    On VAST the writes run concurrently — each is an independent object upload
    to a high-latency endpoint, so overlapping them stacks the three transfer
    windows instead of queueing them (boto3 clients are thread-safe). Impala
    stays sequential: its writes are hundreds of INSERT statements against one
    coordinator, where parallel sessions just contend."""
    if backend() == "vast" and len(frames) > 1:
        from concurrent.futures import ThreadPoolExecutor
        mod = _mod()
        with ThreadPoolExecutor(max_workers=len(frames),
                                thread_name_prefix="vast-write") as pool:
            futures = {split: pool.submit(mod.write_split, df, split, progress=progress)
                       for split, df in frames.items()}
            return {split: f.result() for split, f in futures.items()}
    return {split: write_split(df, split, progress=progress)
            for split, df in frames.items()}


def read_split_cudf(split: str, columns: Sequence[str]):
    return _mod().read_split_cudf(split, columns)


def check() -> Dict:
    """Backend check() plus the fields the UI renders for either backend."""
    return {**_mod().check(), "backend": backend(), "target": target()}


def splits_ready() -> tuple[bool, str]:
    return _mod().splits_ready()
