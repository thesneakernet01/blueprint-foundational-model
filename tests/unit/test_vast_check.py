# SPDX-License-Identifier: Apache-2.0
"""vast.check() row-count fallback — stores that drop the x-amz-meta-rows
user metadata (the original VAST endpoint did, on multipart uploads) must
still report the split as present, via the Parquet footer."""

import io

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tfm_demo import vast


ROWS = 1234


def _parquet_bytes() -> bytes:
    buf = io.BytesIO()
    pq.write_table(pa.table({"Amount": [f"${i}.00" for i in range(ROWS)]}),
                   buf, row_group_size=100)
    return buf.getvalue()


class _Body:
    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data


class _FakeClient:
    """head_bucket/head_object/get_object over an in-memory object store."""

    def __init__(self, objects, metadata):
        self.objects = objects            # key -> bytes
        self.metadata = metadata          # key -> user-metadata dict
        self.get_calls = 0

    def head_bucket(self, Bucket):
        return {}

    def head_object(self, Bucket, Key):
        if Key not in self.objects:
            raise RuntimeError("404 NoSuchKey")
        return {"ContentLength": len(self.objects[Key]),
                "Metadata": self.metadata.get(Key, {})}

    def get_object(self, Bucket, Key, Range=None):
        self.get_calls += 1
        data = self.objects[Key]
        if Range:
            start, end = Range.removeprefix("bytes=").split("-")
            data = data[int(start):int(end) + 1]
        return {"Body": _Body(data)}


@pytest.fixture
def fake_settings(monkeypatch):
    monkeypatch.setattr(vast, "get_vast_settings", lambda: {
        "endpoint": "https://store.example", "bucket": "demo",
        "prefix": "tfm", "access_key": "k", "secret_key": "s",
    })


def _install(monkeypatch, client):
    monkeypatch.setattr(vast, "_client", lambda read_timeout=120, attempts=5:
                        (client, "demo"))


def test_check_uses_rows_metadata_without_fetching(fake_settings, monkeypatch):
    data = _parquet_bytes()
    keys = {f"tfm/{t}.parquet": data for t in vast.SPLIT_TABLES.values()}
    client = _FakeClient(keys, {k: {"rows": str(ROWS)} for k in keys})
    _install(monkeypatch, client)

    out = vast.check()
    assert out["ok"]
    assert out["tables"] == {t: ROWS for t in vast.SPLIT_TABLES.values()}
    assert client.get_calls == 0


def test_check_falls_back_to_parquet_footer(fake_settings, monkeypatch):
    """Metadata stripped by the store: counts still come from the footer."""
    data = _parquet_bytes()
    keys = {f"tfm/{t}.parquet": data for t in vast.SPLIT_TABLES.values()}
    client = _FakeClient(keys, {})
    _install(monkeypatch, client)

    out = vast.check()
    assert out["ok"]
    assert out["tables"] == {t: ROWS for t in vast.SPLIT_TABLES.values()}
    ready, detail = vast.splits_ready()
    assert ready, detail


def test_check_metadata_key_case_insensitive(fake_settings, monkeypatch):
    data = _parquet_bytes()
    keys = {f"tfm/{t}.parquet": data for t in vast.SPLIT_TABLES.values()}
    client = _FakeClient(keys, {k: {"Rows": str(ROWS)} for k in keys})
    _install(monkeypatch, client)

    out = vast.check()
    assert out["tables"] == {t: ROWS for t in vast.SPLIT_TABLES.values()}
    assert client.get_calls == 0


def test_check_non_parquet_object_reports_empty(fake_settings, monkeypatch):
    keys = {f"tfm/{t}.parquet": b"not parquet at all"
            for t in vast.SPLIT_TABLES.values()}
    client = _FakeClient(keys, {})
    _install(monkeypatch, client)

    out = vast.check()
    assert out["ok"]
    assert out["tables"] == {t: 0 for t in vast.SPLIT_TABLES.values()}
    ready, _ = vast.splits_ready()
    assert not ready


def test_check_missing_objects_still_missing(fake_settings, monkeypatch):
    client = _FakeClient({}, {})
    _install(monkeypatch, client)

    out = vast.check()
    assert out["ok"]
    assert out["tables"] == {t: None for t in vast.SPLIT_TABLES.values()}
    ready, detail = vast.splits_ready()
    assert not ready
    assert "missing" in detail
