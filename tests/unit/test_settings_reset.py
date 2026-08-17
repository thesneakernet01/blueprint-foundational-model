# SPDX-License-Identifier: Apache-2.0
"""reset_data_settings(): per-backend reset keeps the rest; "all" forgets
everything including the backend choice."""

import pytest

from tfm_demo import settings


CFG = {
    "backend": "vast",
    "impala": {"connection": "default-impala", "database": "tfm_demo"},
    "vast": {"endpoint": "https://s3.example", "bucket": "demo",
             "prefix": "tfm", "access_key": "k", "secret_key": "sec"},
}

ENV_VARS = ("DATA_BACKEND", "IMPALA_CONNECTION_NAME", "IMPALA_DATABASE",
            "VAST_ENDPOINT", "VAST_BUCKET", "VAST_PATH", "VAST_ACCESS_KEY",
            "VAST_SECRET_KEY")


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "_PATH", tmp_path / ".data_settings.json")
    monkeypatch.setattr(settings, "_LEGACY_PATH", tmp_path / ".impala_settings.json")
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    return tmp_path


def test_reset_impala_keeps_vast_and_backend(isolated):
    settings.save_data_settings(CFG)
    out = settings.reset_data_settings("impala")
    assert out["backend"] == "vast"
    assert out["impala"] == {"connection": "", "database": ""}
    assert out["vast"]["bucket"] == "demo"
    assert settings.get_vast_settings()["secret_key"] == "sec"


def test_reset_vast_drops_the_secret(isolated):
    settings.save_data_settings(CFG)
    out = settings.reset_data_settings("vast")
    assert out["backend"] == "vast"
    assert all(v == "" for v in out["vast"].values())
    assert out["impala"]["database"] == "tfm_demo"


def test_reset_all_forgets_the_backend_choice(isolated, monkeypatch):
    settings.save_data_settings(CFG)
    out = settings.reset_data_settings("all")
    assert not settings._PATH.exists()
    assert out["backend"] == "impala"          # fresh-project default, no env
    monkeypatch.setenv("VAST_BUCKET", "seeded")
    assert settings.get_data_settings()["backend"] == "vast"


def test_reset_all_removes_legacy_file(isolated):
    settings._LEGACY_PATH.write_text('{"connection": "c", "database": "d"}')
    settings.reset_data_settings("all")
    assert not settings._LEGACY_PATH.exists()


def test_reset_scope_reseeds_from_env(isolated, monkeypatch):
    settings.save_data_settings(CFG)
    monkeypatch.setenv("IMPALA_DATABASE", "seeded_db")
    out = settings.reset_data_settings("impala")
    assert out["impala"]["database"] == "seeded_db"


def test_reset_bad_scope_raises(isolated):
    with pytest.raises(ValueError):
        settings.reset_data_settings("everything")
